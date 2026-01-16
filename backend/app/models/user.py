from typing import Optional
from pydantic import BaseModel, Field, EmailStr


class User(BaseModel):
    """Model representing an authenticated user in the system.
    
    This model is used to represent users across the application,
    typically populated from Firebase Authentication tokens.
    
    Attributes:
        uid: Unique identifier for the user (required)
        email: User's email address (optional)
        email_verified: Whether the email address has been verified (optional)
        name: User's display name (optional)
    """
    
    uid: str = Field(
        ...,
        description="Unique identifier for the user",
        min_length=1,
    )
    email: Optional[EmailStr] = Field(
        None,
        description="User's email address",
    )
    email_verified: Optional[bool] = Field(
        None,
        description="Whether the email address has been verified",
        default=False,
    )
    name: Optional[str] = Field(
        None,
        description="User's display name",
        max_length=255,
    )
    
    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "uid": "user123abc",
                "email": "user@example.com",
                "email_verified": True,
                "name": "John Doe",
            }
        }

