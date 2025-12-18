import os
import streamlit as st
import time

def check_authentication():
    """
    Handles Simple Email + Password Authentication.
    Returns True if the user is authenticated, False otherwise.
    """
    if "user_info" in st.session_state:
        return True

    # 1. Container for the login form
    login_container = st.container()

    with login_container:
        st.header("Bellwether Studio Login")
        
        # 2. Form for Email/Password
        with st.form("login_form"):
            email = st.text_input("Email Address", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="Enter access code")
            submit = st.form_submit_button("Sign In")
        
        if submit:
            # 3. Validation Logic
            # Check for valid email format (basic)
            if "@" not in email or "." not in email:
                st.error("Please enter a valid email address.")
                return False
            
            # Check password against End Variable (Simple Shared Secret)
            # Default password is 'bellwether' if not set
            correct_password = os.environ.get("APP_PASSWORD", "bellwether")
            
            if password == correct_password:
                st.success("Login successful!")
                # Store user info mimics the structure we used before for compatibility
                st.session_state["user_info"] = {
                    "email": email,
                    "name": email.split("@")[0].title(),
                    # No picture for simple auth, app handles it gracefully
                }
                time.sleep(0.5) # Short delay for UX
                st.rerun()
            else:
                st.error("Incorrect password / access code.")
                return False

    return False

def logout():
    """Logs the user out."""
    if "user_info" in st.session_state:
        del st.session_state["user_info"]
    st.rerun()
