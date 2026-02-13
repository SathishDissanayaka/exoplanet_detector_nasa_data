import streamlit as st
from pages.landing import show_landing_page
from pages.login import show_login_page
from pages.signup import show_signup_page
from pages.predict import show_predict_page
from pages.train import show_train_page
from pages.history import show_history_page
from utils.auth import AuthManager
from models.model_manager import ModelManager
from utils.url_manager import get_page_from_url, set_page_in_url
import streamlit.components.v1 as components
import os
hh
# Load custom CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), 'styles', 'custom.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

hide_default_format = """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)
load_css()

# Configure page to use wide layout
st.set_page_config(
    page_title="Exoplanet Detection System",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state and restore user session if exists
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    
    # Initialize and auto-load pre-trained models
    with st.spinner("🔄 Loading pre-trained models from disk..."):
        st.session_state.model_manager = ModelManager(auto_load_models=True)
        
        loaded_count = sum(
            1 for model in st.session_state.model_manager.available_models.values() 
            if model['trained']
        )
        
        if loaded_count > 0:
            st.toast(f"✅ Loaded {loaded_count} pre-trained models!", icon="🎉")
        else:
            st.toast("⚠️ No pre-trained models found - you'll need to train them first", icon="⚠️")
    
    # Try to restore user session from Supabase
    current_user = AuthManager.get_current_user()
    if current_user:
        st.session_state.user = current_user
        # Set page from URL or default to predict
        page = get_page_from_url()
        st.session_state.page = page if page != "landing" else "predict"
    else:
        # Set page from URL or default to landing
        st.session_state.page = get_page_from_url()

if "page" not in st.session_state:
    st.session_state.page = "landing"

# Navigation sidebar
with st.sidebar:
    st.title("Navigation")
    
    if st.session_state.get("user"):
        # Navigation for logged-in users
        nav_items = [
            ("🔮 Predict", "predict", "auth"),
            ("📊 History", "history", "auth"),
            ("🎯 Train Model", "train", "auth")
        ]
        
        for label, page, section in nav_items:
            if st.button(label, key=f"nav_{section}_{page}", use_container_width=True):
                st.session_state.page = page
                set_page_in_url(page)
                st.rerun()
                
        if st.button("🚪 Logout", key="nav_auth_logout", type="secondary", use_container_width=True):
            AuthManager.sign_out()
            st.session_state.user = None
            st.session_state.page = "landing"
            set_page_in_url("landing")
            st.rerun()
            
    else:
        # Navigation for non-logged-in users
        nav_items = [
            ("Home ", "landing", "public"),
            ("Login", "login", "public"),
            ("Sign Up", "signup", "public")
        ]
        
        for label, page, section in nav_items:
            if st.button(label, key=f"nav_{section}_{page}", use_container_width=True):
                st.session_state.page = page
                set_page_in_url(page)
                st.rerun()

# Define protected routes
protected_routes = ["predict", "history", "train"]
public_routes = ["landing", "login", "signup"]

# Page routing
current_page = st.session_state.page

# Check if the current page is protected and user is not logged in
if current_page in protected_routes and not st.session_state.get("user"):
    st.warning("Please log in to access this page")
    st.session_state.page = "login"
    set_page_in_url("login")
    st.rerun()
else:
    # Route to appropriate page
    if current_page == "landing":
        show_landing_page()
    elif current_page == "login":
        show_login_page()
    elif current_page == "signup":
        show_signup_page()
    elif current_page == "predict":
        show_predict_page()
    elif current_page == "history":
        show_history_page()
    elif current_page == "train":
        show_train_page()
    else:
        st.error("Page not found")
        st.session_state.page = "landing"
        set_page_in_url("landing")
        st.rerun()
