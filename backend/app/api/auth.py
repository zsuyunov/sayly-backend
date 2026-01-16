from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Header

from app.auth.dependencies import get_current_user
from app.auth.firebase import get_firebase_auth
from app.models.otp import OTPRequest, OTPVerifyRequest, OTPResponse, OTPVerifyResponse
from app.services.otp_service import generate_otp, store_otp, verify_otp, send_otp_email, normalize_email

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
        
        # Generate 4-digit OTP
        code = generate_otp()
        
        # Store OTP in Firestore with expiration (is_resend flag controls whether to reuse existing)
        is_new_otp, code_to_send = store_otp(request.email, request.uid, code, is_resend=request.resend)
        
        # Normalize email
        normalized_email = normalize_email(request.email)
        
        # Send OTP via email if we have a code to send
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
                # Still return success as code is stored (email sending might fail but code is available)
                return OTPResponse(
                    success=True,
                    message="Verification code generated. Please check your email."
                )
        else:
            # Reusing existing OTP but code not available for resend (shouldn't happen with current implementation)
            return OTPResponse(
                success=True,
                message="A verification code was already sent to your email. Please check your inbox."
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
        return OTPVerifyResponse(
            success=False,
            message="Unable to verify code. Please try again."
        )

