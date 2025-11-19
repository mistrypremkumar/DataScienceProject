import requests

# REST base from existing code: rest_base = SUPABASE_URL.rstrip("/") + "/rest/v1"
# SUPABASE_KEY is already defined in your app and used for headers

headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

def column_exists_rest(table: str, column: str, rest_base: str, headers: dict) -> bool:
    """
    Test whether 'column' exists on 'table' using PostgREST.
    Returns True if the column is accepted (exists), False if PostgREST returns 400 (column not found)
    """
    if not column or column.strip() == "":
        return False
    test_url = f"{rest_base}/{table}?select={column}&limit=1"
    try:
        r = requests.get(test_url, headers=headers, timeout=10)
    except Exception:
        # network error -> be conservative and return False so we don't add an invalid filter
        return False
    if r.status_code == 400:
        # column doesn't exist for this table
        return False
    # 200 or 204 etc = column exists (or returned empty rows but column accepted)
    return r.status_code == 200

# Example usage in your filter-building logic
# (Assuming 'table' variable is normalized to 'users' or 'orders', rest_base and headers exist)

filters = []
# Only add shipping_cost filters if the column exists for this table
if column_exists_rest(table, "shipping_cost", rest_base, headers):
    if min_ship is not None:
        filters.append(("shipping_cost", "gte", str(min_ship)))
    if max_ship is not None and max_ship < 1e9:
        filters.append(("shipping_cost", "lte", str(max_ship)))
else:
    st.info("Column 'shipping_cost' not present on table '" + table + "'. Skipping that filter.")
