import streamlit as st

def get_page_from_url():
    """Get the current page from URL query parameters"""
    return st.query_params.get("page", "landing")

def set_page_in_url(page):
    """Set the page in URL query parameters"""
    st.query_params["page"] = page