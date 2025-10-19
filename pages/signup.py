import streamlit as st
from utils.auth import AuthManager
import re

def is_valid_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def is_strong_password(password: str) -> tuple[bool, str]:
    """
    Validate password strength
    Returns: (is_valid, message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, "Password is strong"

def show_signup_page():
    st.title("🚀 Create an Account")
    
    # Center the form
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        # Sign up form with improved styling
        with st.form("signup_form"):
            st.markdown("### Join Exoplanet Detector")
            st.write("Create an account to start detecting exoplanets!")
            
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
            
            # Terms and conditions
            agree = st.checkbox("I agree to the Terms and Conditions")
            
            submitted = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submitted:
                if not email or not password or not confirm_password:
                    st.error("Please fill in all fields")
                elif not is_valid_email(email):
                    st.error("Please enter a valid email address")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif not agree:
                    st.error("Please agree to the Terms and Conditions")
                else:
                    # Validate password strength
                    is_valid, msg = is_strong_password(password)
                    if not is_valid:
                        st.error(msg)
                    else:
                        # Attempt signup
                        with st.spinner("Creating your account..."):
                            result = AuthManager.sign_up(email, password)
                            if result["success"]:
                                st.success("Account created successfully! Please check your email to verify your account.")
                                st.session_state.page = "login"
                                st.rerun()
                            else:
                                st.error(f"Sign up failed: {result['error']}")
    
        # Login option with centered button
        st.markdown("---")
        st.write("Already have an account?")
        if st.button("Login Here", use_container_width=True):
            st.session_state.page = "login"