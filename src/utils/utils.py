import json
from typing import Optional, Any
# from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st
import hmac

# Import logging utilities
from .logging_utils import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


# Function to save feedback to a file
def save_feedback(feedback_text: str) -> None:
    """
    Save user feedback to Google Sheets.
    
    Args:
        feedback_text: The feedback text to save
        
    Raises:
        Exception: If feedback saving fails
    """
    logger.info("Attempting to save user feedback", 
                feedback_length=len(feedback_text))
    
    try:
        # Load the credentials from the secrets
        logger.debug("Loading GCP credentials from secrets")
        credentials_data = st.secrets["gcp"]["service_account_json"]
        
        with logger.performance_timer("credentials_parsing"):
            creds = json.loads(credentials_data, strict=False)
        
        logger.debug("GCP credentials loaded successfully")

        # Set up the Google Sheets API credentials
        scope = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
        # credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds, scope)
        # client = gspread.authorize(credentials)

        # Open the Google Sheet
        sheet_id = '1qnFzZZ7YI-9pXj3iAXafjRmC_EIQyK9gA98AjMv29DM'
        # sheet = client.open_by_key(sheet_id).worksheet("groupdebating")
        # sheet.append_row([feedback_text])
        
        # Note: Google Sheets functionality is currently commented out
        logger.info("Feedback save completed (Google Sheets integration disabled)", 
                   sheet_id=sheet_id)
        
    except KeyError as e:
        logger.error("Missing required secrets for feedback saving", 
                    error=e,
                    missing_key=str(e))
        raise Exception(f"Configuration error: Missing required secret {e}")
    
    except json.JSONDecodeError as e:
        logger.error("Failed to parse GCP credentials JSON", 
                    error=e)
        raise Exception("Configuration error: Invalid GCP credentials format")
    
    except Exception as e:
        logger.error("Unexpected error during feedback saving", 
                    error=e,
                    feedback_length=len(feedback_text))
        raise Exception(f"Failed to save feedback: {str(e)}")
       
    
# Password checking function
def check_password() -> bool:
    """
    Returns `True` if the user had the correct password.
    
    Returns:
        bool: True if password is correct, False otherwise
    """
    def password_entered() -> None:
        """Checks whether a password entered by the user is correct."""
        try:
            logger.debug("Password authentication attempt")
            
            if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store the password.
                logger.info("Password authentication successful")
            else:
                st.session_state["password_correct"] = False
                logger.warning("Password authentication failed - incorrect password")
                
        except KeyError as e:
            logger.error("Password authentication failed - missing configuration", 
                        error=e,
                        missing_key=str(e))
            st.session_state["password_correct"] = False
        except Exception as e:
            logger.error("Unexpected error during password authentication", 
                        error=e)
            st.session_state["password_correct"] = False

    # Check if already authenticated
    if st.session_state.get("password_correct", False):
        logger.debug("User already authenticated")
        return True

    # Show password input
    logger.debug("Displaying password input form")
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    
    # Show error if password was incorrect
    if "password_correct" in st.session_state:
        logger.debug("Displaying password error message")
        st.error("😕 Password incorrect")
    
    return False


