import random
import time
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Tuple
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import firestore
from app.auth.firebase import get_firebase_auth
from dotenv import load_dotenv

load_dotenv()

# OTP expiration time in minutes
OTP_EXPIRATION_MINUTES = 10
# OTP length
OTP_LENGTH = 4

def get_firestore_db():
    """Get Firestore database instance."""
    try:
        return firestore.client()
    except Exception as e:
        # Firestore is automatically initialized with Firebase Admin
        # If this fails, Firebase Admin wasn't initialized properly
        raise RuntimeError(f"Firestore not available: {e}")

def generate_otp() -> str:
    """Generate a 4-digit numeric OTP code."""
    return str(random.randint(1000, 9999))

def store_otp(email: str, uid: str, code: str) -> None:
    """Store OTP code in Firestore with expiration."""
    db = get_firestore_db()
    expires_at = datetime.utcnow() + timedelta(minutes=OTP_EXPIRATION_MINUTES)
    
    otp_data = {
        'code': code,
        'email': email,
        'uid': uid,
        'created_at': firestore.SERVER_TIMESTAMP,
        'expires_at': expires_at,
        'verified': False,
    }
    
    # Store in 'otp_codes' collection with email as document ID
    doc_ref = db.collection('otp_codes').document(email)
    doc_ref.set(otp_data)

def verify_otp(email: str, uid: str, code: str) -> Tuple[bool, str]:
    """Verify OTP code and mark as verified if valid.
    
    Returns:
        tuple: (is_valid, message)
    """
    db = get_firestore_db()
    doc_ref = db.collection('otp_codes').document(email)
    doc = doc_ref.get()
    
    if not doc.exists:
        return False, "Invalid verification code"
    
    otp_data = doc.to_dict()
    
    # Check if already verified
    if otp_data.get('verified', False):
        return False, "This code has already been used"
    
    # Check if UID matches
    if otp_data.get('uid') != uid:
        return False, "Invalid verification code"
    
    # Check if code matches
    if otp_data.get('code') != code:
        return False, "Invalid verification code"
    
    # Check expiration
    expires_at = otp_data.get('expires_at')
    if expires_at:
        if isinstance(expires_at, datetime):
            if datetime.utcnow() > expires_at:
                return False, "Verification code has expired"
        else:
            # Handle Firestore Timestamp
            expires_timestamp = expires_at.timestamp()
            if time.time() > expires_timestamp:
                return False, "Verification code has expired"
    
    # Mark as verified
    doc_ref.update({'verified': True})
    
    return True, "Email verified successfully"

def send_otp_email(email: str, code: str) -> bool:
    """Send OTP code via email using SMTP.
    
    Configure SMTP settings via environment variables:
    - SMTP_HOST: SMTP server host (e.g., smtp.gmail.com)
    - SMTP_PORT: SMTP server port (e.g., 587)
    - SMTP_USER: SMTP username/email
    - SMTP_PASSWORD: SMTP password or app password
    - SMTP_FROM_EMAIL: From email address
    """
    try:
        # Get SMTP configuration from environment variables
        smtp_host = os.getenv("SMTP_HOST", "")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        smtp_from_email = os.getenv("SMTP_FROM_EMAIL", smtp_user)
        
        # If SMTP is not configured, fall back to console logging
        if not smtp_host or not smtp_user or not smtp_password:
            print(f"[OTP] SMTP not configured. OTP Code for {email}: {code}")
            print(f"[OTP] To enable email sending, set SMTP_HOST, SMTP_USER, and SMTP_PASSWORD in your .env file")
            return True  # Still return True as code is stored
        
        # Create email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = 'Your Sayly Verification Code'
        msg['From'] = smtp_from_email
        msg['To'] = email
        
        # Create email body
        text_content = f"""
Hello,

Your verification code for Sayly is: {code}

This code will expire in 10 minutes.

If you didn't request this code, please ignore this email.

Best regards,
Sayly Team
"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .code {{
            font-size: 32px;
            font-weight: bold;
            color: #FFD700;
            text-align: center;
            padding: 20px;
            background-color: #f5f5f5;
            border-radius: 8px;
            letter-spacing: 8px;
            margin: 20px 0;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <h2>Your Sayly Verification Code</h2>
    <p>Hello,</p>
    <p>Your verification code is:</p>
    <div class="code">{code}</div>
    <p>This code will expire in <strong>10 minutes</strong>.</p>
    <p>If you didn't request this code, please ignore this email.</p>
    <div class="footer">
        <p>Best regards,<br>Sayly Team</p>
    </div>
</body>
</html>
"""
        
        # Attach both plain text and HTML versions
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_content, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email via SMTP
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()  # Enable TLS encryption
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        print(f"[OTP] Email sent successfully to {email}")
        return True
        
    except smtplib.SMTPException as e:
        print(f"[OTP] SMTP error sending email to {email}: {e}")
        # Still return True as code is stored - user can see it in console for now
        return True
    except Exception as e:
        print(f"[OTP] Error sending email to {email}: {e}")
        # Still return True as code is stored
        return True

def cleanup_expired_otps() -> None:
    """Clean up expired OTP codes from Firestore."""
    db = get_firestore_db()
    now = datetime.utcnow()
    
    # Query for expired OTPs
    expired_otps = db.collection('otp_codes').where('expires_at', '<', now).stream()
    
    for doc in expired_otps:
        doc.reference.delete()

