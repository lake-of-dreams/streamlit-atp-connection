import streamlit as st

prep_page = st.Page("prep.py", title="Data Preparation", icon=":material/database:")
search_page = st.Page("search.py", title="Search", icon=":material/search:")

pg = st.navigation([prep_page, search_page])
st.set_page_config(page_title="Streamlit Oracle ATP Connection example", page_icon=":material/automation:")
pg.run()