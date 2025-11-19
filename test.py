# streamlit_filters.py
# ATTENTION: Put your keys in .streamlit/secrets.toml or env vars, do NOT hardcode in production.
#
# .streamlit/secrets.toml example:
# SUPABASE_URL = "https://hzyzqmyabfqagcxdwjti.supabase.co"
# SUPABASE_KEY = "YOUR_ANON_KEY"
# PG_USER = "postgres"
# PG_PASS = "1234"   # rotate immediately; use service_role or env for server-side only
# PG_DB = "postgres"
# PG_HOST = "db.hzyzqmyabfqagcxdwjti.supabase.co"
# PG_PORT = "5432"
#
# This app will try SQLAlchemy first (force IPv4 if available), then fallback to REST.

import streamlit as st
import os
import socket
import pandas as pd
import requests
import traceback

st.set_page_config(page_title="Orders — SQL/REST with Advanced Filters", layout="wide")
st.title("Orders: try SQLAlchemy → fallback to Supabase REST")

# Load secrets / env
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", "https://hzyzqmyabfqagcxdwjti.supabase.co"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", ""))
PG_HOST = st.secrets.get("PG_HOST", os.environ.get("PG_HOST", "db.hzyzqmyabfqagcxdwjti.supabase.co"))
PG_PORT = int(st.secrets.get("PG_PORT", os.environ.get("PG_PORT", 5432)))
PG_DB = st.secrets.get("PG_DB", os.environ.get("PG_DB", "postgres"))
PG_USER = st.secrets.get("PG_USER", os.environ.get("PG_USER", "postgres"))
PG_PASS = st.secrets.get("PG_PASS", os.environ.get("PG_PASS", ""))  # rotate password ASAP

# Helper: force-resolve IPv4 (return None if not found)
def resolve_first_ipv4(host):
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_INET)
        return infos[0][4][0]
    except Exception:
        return None

# Try SQLAlchemy connection (returns dataframe or raises)
def try_sqlalchemy_fetch(sql, host, port, db, user, password, force_ipv4=True):
    from sqlalchemy import create_engine, text
    import urllib.parse

    ipv4 = resolve_first_ipv4(host) if force_ipv4 else None

    # quote password
    pw_q = urllib.parse.quote_plus(password) if password else ""
    # Build base connection URL (psycopg2 driver)
    conn_url = f"postgresql+psycopg2://{user}:{pw_q}@{host}:{port}/{db}"

    # connect_args: pass hostaddr if ipv4 resolved
    connect_args = {"sslmode": "require", "connect_timeout": 10}
    if ipv4:
        # psycopg2 accepts hostaddr
        connect_args["hostaddr"] = ipv4

    engine = create_engine(conn_url, connect_args=connect_args)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    return df

# Build REST query URL from filter parameters
def build_rest_url(base, table, filters, order, limit):
    # base already ends with /rest/v1
    q = f"{base}/{table}?select=*&limit={limit}"
    # append filters dict (filters should be list of tuples (field, op, val) like ("status","eq","delivered"))
    for field, op, val in filters:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            continue
        # encode values (simple): dates and numbers pass as-is, strings should be URL-safe
        from urllib.parse import quote
        q += f"&{quote(field)}={op}.{quote(str(val))}"
    if order:
        q += f"&order={order}"
    return q

# Sidebar filters (advanced)
st.sidebar.header("Filter options (Advanced)")
status = st.sidebar.selectbox("Status", options=["", "delivered", "cancelled", "pending", "shipped"], index=0)
date_from = st.sidebar.date_input("Order date from", value=None)
date_to = st.sidebar.date_input("Order date to", value=None)
min_ship = st.sidebar.number_input("Min shipping_cost", min_value=0.0, format="%.2f", value=0.0)
max_ship = st.sidebar.number_input("Max shipping_cost", min_value=0.0, format="%.2f", value=10000.0)
sort_field = st.sidebar.selectbox("Sort by", options=["order_date", "shipping_cost", "order_id"], index=0)
sort_dir = st.sidebar.selectbox("Sort direction", options=["asc", "desc"], index=1)
limit = st.sidebar.slider("Limit", 10, 1000, 200)

st.sidebar.markdown("---")
st.sidebar.write("Connection method debug")
st.sidebar.write("Resolved IPv4 for host:", resolve_first_ipv4(PG_HOST))
st.sidebar.write("SQL password present?", bool(PG_PASS))

# Main controls
use_sql_first = st.checkbox("Attempt direct SQL (SQLAlchemy) first", value=True)
table_input = st.text_input("Table (if in public schema, enter name only)", value="orders").strip()
table = table_input.split(".")[-1]  # normalize

run = st.button("Run query")

if run:
    # Construct SQL where clauses (sanitized simple)
    where_clauses = []
    params = {}
    if status:
        where_clauses.append("status = :status")
        params["status"] = status
    if date_from:
        where_clauses.append("order_date >= :date_from")
        params["date_from"] = date_from.isoformat()
    if date_to:
        where_clauses.append("order_date <= :date_to")
        params["date_to"] = date_to.isoformat()
    # shipping cost filters (use >= and <=)
    if min_ship is not None:
        where_clauses.append("shipping_cost >= :min_ship")
        params["min_ship"] = float(min_ship)
    if max_ship is not None and max_ship < 1e9:
        where_clauses.append("shipping_cost <= :max_ship")
        params["max_ship"] = float(max_ship)

    where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # Build final SQL (limit applied)
    sql = f"SELECT * FROM public.\"{table}\" {where_sql} ORDER BY {sort_field} {sort_dir} LIMIT {limit};"
    st.write("SQL attempt (for SQLAlchemy):")
    st.code(sql)

    df = None
    sql_error = None

    if use_sql_first:
        try:
            st.info("Trying SQLAlchemy (direct Postgres) — may fail due to IPv6/network).")
            df = try_sqlalchemy_fetch(sql, PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS, force_ipv4=True)
            st.success("SQLAlchemy fetch succeeded")
            st.dataframe(df)
        except Exception as e:
            sql_error = traceback.format_exc()
            st.error("SQLAlchemy fetch failed — falling back to REST")
            st.exception(e)

    if df is None:
        # Build REST filters for Supabase PostgREST
        st.info("Building REST query and using Supabase REST API (HTTPS)")
        rest_base = SUPABASE_URL.rstrip("/") + "/rest/v1"
        filters = []
        # status
        if status:
            filters.append(("status", "eq", status))
        # date_from / date_to use gte / lte (ISO)
        if date_from:
            filters.append(("order_date", "gte", date_from.isoformat()))
        if date_to:
            filters.append(("order_date", "lte", date_to.isoformat()))
        # shipping cost
        if min_ship is not None:
            filters.append(("shipping_cost", "gte", str(min_ship)))
        if max_ship is not None and max_ship < 1e9:
            filters.append(("shipping_cost", "lte", str(max_ship)))

        order_param = f"{sort_field}.{sort_dir}"
        rest_url = build_rest_url(rest_base, table, filters, order_param, limit)
        st.write("REST URL:", rest_url)

        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        try:
            r = requests.get(rest_url, headers=headers, timeout=30)
            st.write("Status code:", r.status_code)
            if r.status_code == 200:
                data = r.json()
                if not data:
                    st.info("REST returned zero rows (possible RLS or empty result).")
                else:
                    df = pd.DataFrame(data)
                    st.success(f"REST returned {len(df)} rows")
                    st.dataframe(df)
            else:
                st.error(f"REST request failed with status {r.status_code}")
                st.text(r.text[:2000])
        except Exception as e:
            st.error("REST request failed")
            st.exception(e)

    # Show SQL error if REST used as fallback
    if sql_error:
        st.subheader("SQLAlchemy error (for debugging)")
        st.code(sql_error)
