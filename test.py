# streamlit_orders_test.py
import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(page_title="Supabase ORDERS test", layout="wide")
st.title("Supabase REST — Orders (diagnostic)")

# Load secrets or fall back to environment variable
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://hzyzqmyabfqagcxdwjti.supabase.co")
API_KEY = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY", "")

def mask(k):
    if not k: return "(empty)"
    if len(k) <= 12: return k
    return k[:6] + "..." + k[-6:] + f" (len={len(k)})"

# Debug info so you can see exactly what Streamlit is loading
st.subheader("Debug — what key/paths Streamlit sees")
st.write("Working dir:", os.getcwd())
st.write("Python executable:", __import__("sys").executable)
st.write("SUPABASE_URL:", SUPABASE_URL)
st.write("SUPABASE_KEY (masked):", mask(API_KEY))
st.write("Contents of .streamlit folder (if present):")
import pathlib
p = pathlib.Path(".") / ".streamlit"
if p.exists():
    st.write([x.name for x in p.iterdir()])
else:
    st.write("No .streamlit folder here")

# Query UI
rest_base = SUPABASE_URL.rstrip("/") + "/rest/v1"
table = st.text_input("Table (schema.table)", value="orders")
limit = st.number_input("Limit", min_value=1, max_value=1000, value=50, step=10)

headers = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

if st.button("Fetch orders via REST"):
    if not API_KEY:
        st.error("No API key available to send. Set .streamlit/secrets.toml or SUPABASE_KEY env var.")
    else:
        url = f"{rest_base}/{table}?select=*&limit={limit}"
        st.write("Request URL:", url)
        with st.spinner("Requesting..."):
            try:
                r = requests.get(url, headers=headers, timeout=30)
            except Exception as e:
                st.error("HTTP request failed")
                st.exception(e)
            else:
                st.write("Status code:", r.status_code)
                if r.status_code == 200:
                    try:
                        data = r.json()
                        if not data:
                            st.info("Query succeeded but returned zero rows.")
                        else:
                            df = pd.DataFrame(data)
                            st.success(f"Returned {len(df)} rows")
                            st.dataframe(df)
                    except Exception as e:
                        st.error("Failed to parse JSON response")
                        st.text(r.text[:2000])
                else:
                    st.error("Request failed")
                    st.text(r.text[:2000])
