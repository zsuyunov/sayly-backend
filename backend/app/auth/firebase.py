import os
import json
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv

load_dotenv()

# Initialize Firebase Admin SDK only once
if not firebase_admin._apps:
    # Try to load from environment variables first (preferred for production)
    private_key_raw = os.getenv("FIREBASE_PRIVATE_KEY", "")
    # Handle private key formatting - replace literal \n with actual newlines
    # Also handle cases where it might be wrapped in quotes or have extra whitespace
    if private_key_raw:
        private_key_raw = private_key_raw.strip()
        # Remove surrounding quotes if present
        if (private_key_raw.startswith('"') and private_key_raw.endswith('"')) or \
           (private_key_raw.startswith("'") and private_key_raw.endswith("'")):
            private_key_raw = private_key_raw[1:-1]
        # Replace escaped newlines with actual newlines
        private_key = private_key_raw.replace("\\n", "\n")
    else:
        private_key = ""
    
    firebase_credentials = {
        "type": os.getenv("FIREBASE_TYPE", "service_account"),
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": private_key,
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": os.getenv("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
        "token_uri": os.getenv("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
        "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
        "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN", "googleapis.com"),
    }
    
    # Check if all required fields are present
    required_fields = ["project_id", "private_key", "client_email"]
    missing_fields = [field for field in required_fields if not firebase_credentials.get(field)]
    
    if not missing_fields:
        # Use credentials from environment variables
        cred = credentials.Certificate(firebase_credentials)
        firebase_admin.initialize_app(cred)
    else:
        # Fallback to file path if environment variables are not set
        service_account_path = os.getenv("FIREBASE_CREDENTIALS")
        if service_account_path:
            from pathlib import Path
            # Resolve relative paths relative to the backend directory
            if not os.path.isabs(service_account_path):
                # Remove ./ prefix if present
                service_account_path = service_account_path.lstrip('./')
                # Get the backend directory (parent of app directory)
                backend_dir = Path(__file__).parent.parent.parent
                service_account_path = str(backend_dir / service_account_path)
            
            # Verify the file exists
            if os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path) 
                firebase_admin.initialize_app(cred)
            else:
                raise FileNotFoundError(
                    f"Firebase service account file not found: {service_account_path}. "
                    f"Please check that the file exists or set all FIREBASE_* environment variables."
                )
        else:
            raise ValueError(
                f"Firebase credentials are missing. Please set FIREBASE_CREDENTIALS (file path) "
                f"or set all required FIREBASE_* environment variables. Missing fields: {missing_fields}"
            )


def get_firebase_auth():
    """Get Firebase Auth instance for verifying user identity tokens.
    
    This function returns the Firebase Auth module which provides methods
    to verify user identity tokens, including verify_id_token().
    
    Example usage:
        auth_service = get_firebase_auth()
        decoded_token = auth_service.verify_id_token(id_token)
    
    Returns:
        firebase_admin.auth: Firebase Auth module instance capable of 
            verifying user identity tokens via verify_id_token() method
    """
    return auth
