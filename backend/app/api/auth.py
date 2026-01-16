from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Header

from app.auth.dependencies import get_current_user
from app.auth.firebase import get_firebase_auth
from app.models.otp import OTPRequest, OTPVerifyRequest, OTPResponse, OTPVerifyResponse
from app.services.otp_service import generate_otp, store_otp, verify_otp, send_otp_email, normalize_email, get_existing_otp, mask_otp

router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
)


@router.get(
    "/me",
    summary="Get current authenticated user",
    description="Returns information about the currently authenticated user based on their identity token.",
    responses={
        200: {
            "description": "Successfully retrieved user information",
            "content": {
                "application/json": {
                    "example": {
                        "uid": "user123",
                        "email": "user@example.com",
                        "email_verified": True,
                        "name": "John Doe",
                    }
                }
            },
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get information about the currently authenticated user.
    
    This endpoint requires a valid Firebase identity token in the Authorization header.
    The token is validated and the user information is extracted and returned.
    
    Args:
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        Dict[str, Any]: User information including:
            - uid: User's unique identifier
            - email: User's email address (if available)
            - email_verified: Whether email is verified (if available)
            - name: User's display name (if available)
    """
    return current_user


@router.post(
    "/otp/generate",
    response_model=OTPResponse,
    summary="Generate OTP code",
    description="Generate a 4-digit OTP code and send it to the user's email. Works for unauthenticated users during signup.",
)
async def generate_otp_code(
    request: OTPRequest,
    authorization: Optional[str] = Header(None)
) -> OTPResponse:
    """Generate and send OTP code to user's email.
    
    This endpoint works for both authenticated and unauthenticated users.
    If a token is provided, it validates the uid matches the token.
    
    Args:
        request: OTP request containing email and uid
        authorization: Optional Bearer token for validation
        
    Returns:
        OTPResponse: Success status and message
    """
    try:
        # If token is provided, validate uid matches
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            try:
                auth_service = get_firebase_auth()
                decoded_token = auth_service.verify_id_token(token)
                if decoded_token["uid"] != request.uid:
                    return OTPResponse(
                        success=False,
                        message="Unable to send verification code. Please try again."
                    )
            except Exception:
                # Token validation failed, but we still allow the request
                # (user might be in signup flow)
                pass
        
        # Verify uid exists in Firebase (basic validation)
        try:
            auth_service = get_firebase_auth()
            auth_service.get_user(request.uid)
        except Exception:
            return OTPResponse(
                success=False,
                message="Unable to send verification code. Please try again."
            )
        
        # Normalize email
        normalized_email = normalize_email(request.email)
        
        # Check if we should reuse existing OTP (only if not explicit resend)
        is_new_otp = True
        code_to_send = None
        
        if not request.resend:
            # Check for existing unexpired OTP
            existing_otp = get_existing_otp(normalized_email, request.uid)
            if existing_otp:
                # Reuse existing OTP
                existing_code = existing_otp.get('code')
                if existing_code:
                    is_new_otp = False
                    code_to_send = str(existing_code).strip()
                    print(f"[OTP] Reusing existing OTP for {normalized_email} (code: {mask_otp(code_to_send)})")
                    # IMPORTANT: Don't generate or store new OTP - just return the existing one
                    # This prevents overwriting the OTP the user is trying to verify
                else:
                    # OTP exists but code missing - generate new one
                    print(f"[OTP] Existing OTP found but code missing - generating new OTP")
                    is_new_otp = True
        
        # Generate new OTP ONLY if we don't have an existing one to reuse
        if is_new_otp:
            code = generate_otp()
            # Store OTP in Firestore (this will overwrite any existing OTP)
            store_otp(request.email, request.uid, code, is_resend=request.resend)
            code_to_send = str(code).strip()  # Ensure it's a string
            print(f"[OTP] Generated new OTP for {normalized_email} (code: {mask_otp(code_to_send)})")
        
        # Send OTP via email
        if code_to_send:
            email_sent = send_otp_email(normalized_email, code_to_send)
            
            if email_sent:
                if is_new_otp:
                    return OTPResponse(
                        success=True,
                        message="Verification code has been sent to your email"
                    )
                else:
                    # Reusing existing OTP but resending email
                    return OTPResponse(
                        success=True,
                        message="Verification code has been resent to your email"
                    )
            else:
                # Still return success as code is stored
                return OTPResponse(
                    success=True,
                    message="Verification code generated. Please check your email."
                )
        else:
            # This shouldn't happen, but handle gracefully
            return OTPResponse(
                success=False,
                message="Unable to generate verification code. Please try again."
            )
    except Exception as e:
        # Generic error message to avoid email enumeration
        return OTPResponse(
            success=False,
            message="Unable to send verification code. Please try again."
        )


@router.post(
    "/otp/verify",
    response_model=OTPVerifyResponse,
    summary="Verify OTP code",
    description="Verify the 4-digit OTP code and mark email as verified. Works for unauthenticated users during signup.",
)
async def verify_otp_code(
    request: OTPVerifyRequest,
    authorization: Optional[str] = Header(None)
) -> OTPVerifyResponse:
    """Verify OTP code.
    
    This endpoint works for both authenticated and unauthenticated users.
    If a token is provided, it validates the uid matches the token.
    
    Args:
        request: OTP verification request containing email, code, and uid
        authorization: Optional Bearer token for validation
        
    Returns:
        OTPVerifyResponse: Verification status and message
    """
    try:
        # If token is provided, validate uid matches
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            try:
                auth_service = get_firebase_auth()
                decoded_token = auth_service.verify_id_token(token)
                if decoded_token["uid"] != request.uid:
                    return OTPVerifyResponse(
                        success=False,
                        message="Invalid or expired verification code"
                    )
            except Exception:
                # Token validation failed, but we still allow the request
                pass
        
        is_valid, message = verify_otp(request.email, request.uid, request.code)
        
        print(f"[OTP API] Verification result: is_valid={is_valid}, message={message}")
        
        if is_valid:
            return OTPVerifyResponse(
                success=True,
                message="Email verified successfully"
            )
        else:
            # Generic error message
            return OTPVerifyResponse(
                success=False,
                message="Invalid or expired verification code"
            )
    except Exception as e:
        # Log the actual error for debugging
        print(f"[OTP API] Exception during verification: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"[OTP API] Traceback: {traceback.format_exc()}")
        return OTPVerifyResponse(
            success=False,
            message="Unable to verify code. Please try again."
        )

