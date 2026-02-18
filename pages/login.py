import streamlit as st
from utils.auth import AuthManager

from utils.url_manager import set_page_in_url

def show_login_page():
    st.title("🔐 Login to Exoplanet Detector")
    
    # Center the form
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        # Login form with improved styling
        with st.form("login_form"):
            st.markdown("### Welcome Back!")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            remember_me = st.checkbox("Remember me")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if not email or not password:
                    st.error("Please fill in all fields")
                else:
                    with st.spinner("Logging in..."):
                        result = AuthManager.sign_in(email, password)
                        if result["success"]:
                            st.session_state.user = result["user"]
                            st.session_state.page = "predict"
                            set_page_in_url("predict")
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error(f"Login failed: {result['error']}")
    
        # Sign up option with centered button
        st.markdown("---")
        st.write("Don't have an account yet?")
        if st.button("Create an Account", use_container_width=True):
            st.session_state.page = "signup"
            set_page_in_url("signup")
