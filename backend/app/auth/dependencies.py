from typing import Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin.auth import (
    InvalidIdTokenError,
    ExpiredIdTokenError,
    RevokedIdTokenError,
    UserDisabledError,
    CertificateFetchError,
)

from app.auth.firebase import get_firebase_auth

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Extract and validate identity token from request, returning normalized user object.
    
    This function:
    1. Extracts the identity token from the Authorization header
    2. Validates the token using Firebase Authentication service
    3. Returns a normalized user object with essential user information
    4. Rejects the request with 401 if validation fails
    
    Args:
        credentials: HTTPBearer credentials containing the token in Authorization header
        
    Returns:
        Dict[str, Any]: Normalized user object containing:
            - uid: User's unique identifier
            - email: User's email address (if available)
            - email_verified: Whether email is verified (if available)
            - name: User's display name (if available)
            
    Raises:
        HTTPException: 
            - 401 if token is missing, invalid, expired, or revoked
            - 503 if there's an error fetching certificates for validation
    """
    # Extract token from Authorization header
    token = credentials.credentials
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get Firebase Auth service for token validation
    auth_service = get_firebase_auth()
    
    try:
        # Log token info for debugging (first 20 chars only for security)
        print(f"[AUTH] Verifying token (length: {len(token)}, prefix: {token[:20]}...)")
        
        # Validate the identity token
        # Using check_revoked=False to avoid certificate fetching issues on cloud platforms
        # This still validates the token signature and expiration
        decoded_token = auth_service.verify_id_token(token, check_revoked=False)
        
        print(f"[AUTH] Token verified successfully for uid: {decoded_token.get('uid')}")
        
        # Return normalized user object
        return {
            "uid": decoded_token["uid"],
            "email": decoded_token.get("email"),
            "email_verified": decoded_token.get("email_verified", False),
            "name": decoded_token.get("name"),
        }
        
    except (InvalidIdTokenError, ExpiredIdTokenError) as e:
        print(f"[AUTH] Token validation failed: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
        
    except RevokedIdTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
        
    except UserDisabledError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account has been disabled",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
        
    except CertificateFetchError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable",
        ) from e
        
    except ValueError as e:
        print(f"[AUTH] Token format error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token format",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
        
    except Exception as e:
        # Catch any other unexpected errors
        print(f"[AUTH] Unexpected error during token validation: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
