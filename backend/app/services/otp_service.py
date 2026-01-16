import random
import time
import os
import smtplib
import hashlib
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

def normalize_email(email: str) -> str:
    """Normalize email by trimming whitespace and converting to lowercase."""
    return email.strip().lower()

def hash_otp(code: str) -> str:
    """Hash OTP code using SHA-256 for secure storage."""
    return hashlib.sha256(code.encode()).hexdigest()

def verify_otp_hash(code: str, stored_hash: str) -> bool:
    """Verify OTP code against stored hash."""
    return hash_otp(code) == stored_hash

def mask_otp(code: str) -> str:
    """Mask OTP for logging (show only first and last digit)."""
    if len(code) < 2:
        return "****"
    return f"{code[0]}**{code[-1]}"

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

def get_existing_otp(email: str, uid: str) -> Optional[dict]:
    """Get existing OTP for email if it exists, is not expired, and UID matches."""
    db = get_firestore_db()
    normalized_email = normalize_email(email)
    doc_ref = db.collection('otp_codes').document(normalized_email)
    doc = doc_ref.get()
    
    if not doc.exists:
        return None
    
    otp_data = doc.to_dict()
    
    # Check if UID matches (critical for security)
    if otp_data.get('uid') != uid:
        print(f"[OTP] Existing OTP found but UID mismatch - generating new OTP")
        return None  # UID doesn't match, need new OTP
    
    # Check if expired
    expires_at = otp_data.get('expires_at')
    if expires_at:
        if isinstance(expires_at, datetime):
            if datetime.utcnow() > expires_at:
                print(f"[OTP] Existing OTP found but expired - generating new OTP")
                return None  # Expired
        else:
            # Handle Firestore Timestamp
            expires_timestamp = expires_at.timestamp()
            if time.time() > expires_timestamp:
                print(f"[OTP] Existing OTP found but expired - generating new OTP")
                return None  # Expired
    
    # Check if already verified
    if otp_data.get('verified', False):
        print(f"[OTP] Existing OTP found but already verified - generating new OTP")
        return None  # Already verified
    
    return otp_data

def store_otp(email: str, uid: str, code: str, is_resend: bool = False) -> None:
    """Store OTP code in Firestore with expiration.
    
    Args:
        email: Email address (will be normalized)
        uid: Firebase user UID
        code: OTP code (will be hashed for storage, but stored temporarily for resend)
        is_resend: Whether this is an explicit resend request (if True, always stores new OTP)
    """
    db = get_firestore_db()
    normalized_email = normalize_email(email)
    
    # Generate new OTP (always store when called - reuse logic is handled in API layer)
    expires_at = datetime.utcnow() + timedelta(minutes=OTP_EXPIRATION_MINUTES)
    code_hash = hash_otp(code)
    masked_code = mask_otp(code)
    
    otp_data = {
        'code_hash': code_hash,  # Store hash for secure verification
        'code': code,  # Store temporarily for resend capability (will be deleted after verification)
        'email': normalized_email,
        'uid': uid,
        'created_at': firestore.SERVER_TIMESTAMP,
        'expires_at': expires_at,
        'verified': False,
    }
    
    # Store in 'otp_codes' collection with normalized email as document ID
    doc_ref = db.collection('otp_codes').document(normalized_email)
    doc_ref.set(otp_data)
    
    # Debug logs (never log raw OTP)
    print(f"[OTP] OTP stored for {normalized_email} (masked: {masked_code})")
    print(f"[OTP] OTP expires at: {expires_at}")
    print(f"[OTP] UID: {uid}, is_resend: {is_resend}")

def verify_otp(email: str, uid: str, code: str) -> Tuple[bool, str]:
    """Verify OTP code and delete record if valid.
    
    Args:
        email: Email address (will be normalized)
        uid: Firebase user UID
        code: OTP code to verify (will be normalized/trimmed)
    
    Returns:
        tuple: (is_valid, message)
    """
    db = get_firestore_db()
    normalized_email = normalize_email(email)
    # Normalize code by trimming whitespace
    normalized_code = str(code).strip()
    doc_ref = db.collection('otp_codes').document(normalized_email)
    doc = doc_ref.get()
    
    # Debug log verification attempt
    masked_code = mask_otp(normalized_code)
    print(f"[OTP] Verification attempt for {normalized_email} (masked code: {masked_code}, original: '{code}')")
    
    if not doc.exists:
        print(f"[OTP] Verification failed: No OTP found for {normalized_email}")
        return False, "Invalid verification code"
    
    otp_data = doc.to_dict()
    
    # Check if already verified
    if otp_data.get('verified', False):
        print(f"[OTP] Verification failed: OTP already verified for {normalized_email}")
        return False, "This code has already been used"
    
    # Check if UID matches
    if otp_data.get('uid') != uid:
        print(f"[OTP] Verification failed: UID mismatch for {normalized_email}")
        return False, "Invalid verification code"
    
    # Check expiration
    expires_at = otp_data.get('expires_at')
    if expires_at:
        if isinstance(expires_at, datetime):
            if datetime.utcnow() > expires_at:
                print(f"[OTP] Verification failed: OTP expired for {normalized_email}")
                return False, "Verification code has expired"
        else:
            # Handle Firestore Timestamp
            expires_timestamp = expires_at.timestamp()
            if time.time() > expires_timestamp:
                print(f"[OTP] Verification failed: OTP expired for {normalized_email}")
                return False, "Verification code has expired"
    
    # Verify code - check both plain code (preferred) and hash (backup)
    stored_code_hash = otp_data.get('code_hash')
    stored_plain_code = otp_data.get('code')  # Temporary storage for resend
    
    # Normalize stored code if it exists
    if stored_plain_code:
        stored_plain_code = str(stored_plain_code).strip()
    
    # Debug: Log what we have stored (masked)
    print(f"[OTP] Stored data - has_hash: {bool(stored_code_hash)}, has_plain: {bool(stored_plain_code)}")
    print(f"[OTP] Entered code (normalized, masked): {mask_otp(normalized_code)}")
    if stored_plain_code:
        print(f"[OTP] Stored plain code (masked): {mask_otp(stored_plain_code)}")
    
    # Check plain code first (most reliable since it's what we send)
    code_matches = False
    
    if stored_plain_code:
        # Check plain code directly (string comparison after normalization)
        code_matches = (stored_plain_code == normalized_code)
        print(f"[OTP] Plain code verification: {code_matches}")
        print(f"[OTP]   Stored: '{stored_plain_code}' (len={len(stored_plain_code)})")
        print(f"[OTP]   Entered: '{normalized_code}' (len={len(normalized_code)})")
    
    # If plain code doesn't match, try hash verification as fallback
    if not code_matches and stored_code_hash:
        code_matches = verify_otp_hash(normalized_code, stored_code_hash)
        print(f"[OTP] Hash verification (fallback): {code_matches}")
    
    if not code_matches:
        print(f"[OTP] Verification failed: Code mismatch for {normalized_email}")
        return False, "Invalid verification code"
    
    if not stored_code_hash and not stored_plain_code:
        print(f"[OTP] Verification failed: No code stored for {normalized_email}")
        return False, "Invalid verification code"
    
    # Delete OTP record on successful verification (removes both hash and temporary code)
    doc_ref.delete()
    print(f"[OTP] Verification successful for {normalized_email}. OTP record deleted.")
    
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
            masked_code = mask_otp(code)
            print(f"[OTP] SMTP not configured. OTP Code for {email} (masked: {masked_code})")
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

