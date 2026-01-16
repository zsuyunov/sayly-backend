from typing import Optional
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime


class OTPRequest(BaseModel):
    """Request model for generating OTP."""
    email: EmailStr = Field(..., description="Email address to send OTP to")
    uid: str = Field(..., description="Firebase user UID")


class OTPVerifyRequest(BaseModel):
    """Request model for verifying OTP."""
    email: EmailStr = Field(..., description="Email address")
    code: str = Field(..., min_length=4, max_length=4, description="4-digit OTP code")
    uid: str = Field(..., description="Firebase user UID")


class OTPResponse(BaseModel):
    """Response model for OTP generation."""
    success: bool = Field(..., description="Whether OTP was generated successfully")
    message: str = Field(..., description="Response message")


class OTPVerifyResponse(BaseModel):
    """Response model for OTP verification."""
    success: bool = Field(..., description="Whether OTP verification was successful")
    message: str = Field(..., description="Response message")

