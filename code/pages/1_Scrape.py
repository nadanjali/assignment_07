"""
1_Scrape.py — Part 1: scrape the rivals' menus into one dataset.

The owner picks the cafés, clicks Scrape, and sees what came back: how much was
scraped, every row in one table, and a download of the dataset.

The scraping itself runs in a separate program (code/scrape_menus.py) that
runner.py starts for you — Streamlit and Playwright do not share a process well.
This page only asks for it and shows the result.

Test it: pytest tests/test_pipeline.py -k scrape_page
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rivals import CAFES, MENUS_FILE, OUR_CAFE, run_scraper

# --- The page: the algorithm, step by step ------------------------------------------
#
# Write the code for each step below its comment. The tests
# (pytest tests/test_pipeline.py -k scrape_page) check the title, the widget keys,
# the metric labels and values, and that Scrape runs run_scraper() with the cafés
# that were picked. Run the whole app (open code/menu_app.py, then Run and Debug ->
# "Streamlit Run: Current File") and click "Scrape" in the sidebar.
#
# 1. HEADING
#    st.title: Salt City: Scrape the Rivals
#    st.markdown: one sentence, e.g. "Pick the cafés to scrape. Each run replaces
#    the saved dataset."
#
# 2. PICK AND SCRAPE
#    a. cafes <- st.multiselect labeled "Cafés", key "cafes", options list(CAFES),
#       every café selected by default.
#    b. A st.button labeled "Scrape", key "scrape". When it is clicked and at least
#       one café is picked:
#          result <- run_scraper(cafes), inside `with st.spinner(...)` — a browser
#                    visits every café, which takes a few seconds each.
#          result.returncode is 0 -> st.success with the LAST line of result.stdout
#                                    (the scraper's "Wrote ... rows" message)
#          otherwise              -> st.error saying the scraper failed, and
#                                    st.code(the end of result.stderr) — the
#                                    Python error, so you can fix it.
#
# 3. NOTHING TO SHOW YET?
#    If MENUS_FILE does not exist: st.info("No menus yet — ...") and st.stop().
#
# 4. THE DATASET
#    menus <- pd.read_csv(MENUS_FILE); theirs <- the rows whose cafe is not OUR_CAFE
#    Three st.metric cards side by side (st.columns(3)), all about `theirs`:
#       "Cafés": how many different cafés   "Items": how many different
#       (cafe, item) pairs                  "Rows": how many rows (one per size)
#    st.dataframe(menus, hide_index=True)
#    st.download_button "Download the dataset" with menus.to_csv(index=False),
#       file_name "menus.csv", mime "text/csv", key "download"

st.title("Salt City: Scrape the Rivals")
st.markdown("Pick the cafés to scrape. Each run replaces the saved dataset.")
cafes = st.multiselect("Cafés", options=list(CAFES), default=list(CAFES), key="cafes")
if st.button("Scrape", key="scrape"):
    if cafes:
        with st.spinner("Scraping the cafés..."):
            result = run_scraper(cafes)
        if result.returncode == 0:
            st.success(result.stdout.strip().splitlines()[-1])
        else:
            st.error("The scraper failed. See the error below:")
            st.code("\n".join(result.stderr.strip().splitlines()[-10:]))
    else:
        st.warning("Pick at least one café to scrape.")

if not os.path.exists(MENUS_FILE):
    st.info("No menus yet — scrape a café to build the dataset.")
    st.stop()

menus = pd.read_csv(MENUS_FILE)
theirs = menus[menus["cafe"] != OUR_CAFE]

cafes_column, items_column, rows_column = st.columns(3)
cafes_column.metric("Cafés", theirs["cafe"].nunique())
items_column.metric("Items", theirs[["cafe", "item"]].drop_duplicates().shape[0])
rows_column.metric("Rows", len(theirs))

st.dataframe(menus, hide_index=True)
st.download_button(
    "Download the dataset",
    menus.to_csv(index=False),
    file_name="menus.csv",
    mime="text/csv",
    key="download",
)
