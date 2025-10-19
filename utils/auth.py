from config.supabase import supabase
from typing import Dict, Any, Optional

class AuthManager:
    @staticmethod
    def sign_up(email: str, password: str) -> Dict[str, Any]:
        """Sign up a new user"""
        try:
            response = supabase.auth.sign_up({
                "email": email,
                "password": password
            })
            return {
                "success": True,
                "user": response.user,
                "session": response.session
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def sign_in(email: str, password: str) -> Dict[str, Any]:
        """Sign in an existing user"""
        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            return {
                "success": True,
                "user": response.user,
                "session": response.session
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def sign_out() -> Dict[str, Any]:
        """Sign out the current user"""
        try:
            supabase.auth.sign_out()
            return {
                "success": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def get_current_user() -> Optional[Dict[str, Any]]:
        """Get the current user's session"""
        try:
            session = supabase.auth.get_session()
            if session and session.user:
                return session.user
            return None
        except:
            return None

    @staticmethod
    def is_authenticated() -> bool:
        """Check if user is authenticated"""
        try:
            session = supabase.auth.get_session()
            return session is not None and session.user is not None
        except:
            return False