"""
Model Versioning Service for Speaker Embedding Models

This service tracks model metadata (ID, revision, version) and ensures
compatibility between enrollment embeddings and verification operations.
"""
import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from firebase_admin import firestore


# Model configuration
HF_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "main")  # Can be commit hash or branch
INTERNAL_MODEL_VERSION = os.getenv("INTERNAL_MODEL_VERSION", "1.0.0")  # Our versioning


class ModelMetadata:
    """Model metadata for tracking versioning."""
    
    def __init__(
        self,
        model_id: str = HF_MODEL_ID,
        model_revision: str = HF_MODEL_REVISION,
        internal_version: str = INTERNAL_MODEL_VERSION,
        extracted_at: Optional[datetime] = None
    ):
        self.model_id = model_id
        self.model_revision = model_revision
        self.internal_version = internal_version
        self.extracted_at = extracted_at or datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "modelId": self.model_id,
            "modelRevision": self.model_revision,
            "modelVersion": self.internal_version,
            "extractedAt": self.extracted_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        extracted_at = data.get("extractedAt")
        if extracted_at and hasattr(extracted_at, 'timestamp'):
            extracted_at = datetime.fromtimestamp(extracted_at.timestamp(), tz=timezone.utc)
        
        return cls(
            model_id=data.get("modelId", HF_MODEL_ID),
            model_revision=data.get("modelRevision", HF_MODEL_REVISION),
            internal_version=data.get("modelVersion", INTERNAL_MODEL_VERSION),
            extracted_at=extracted_at
        )
    
    def is_compatible_with(self, other: "ModelMetadata") -> bool:
        """Check if this model metadata is compatible with another.
        
        Compatibility rules:
        - Same model ID required
        - Same internal version OR explicitly compatible versions
        """
        if self.model_id != other.model_id:
            return False
        
        # Same internal version is always compatible
        if self.internal_version == other.internal_version:
            return True
        
        # Check compatibility list (can be extended)
        compatible_versions = get_compatible_versions(self.internal_version)
        return other.internal_version in compatible_versions
    
    def __eq__(self, other):
        if not isinstance(other, ModelMetadata):
            return False
        return (
            self.model_id == other.model_id and
            self.model_revision == other.model_revision and
            self.internal_version == other.internal_version
        )


def get_current_model_metadata() -> ModelMetadata:
    """Get current model metadata.
    
    Returns:
        ModelMetadata: Current model metadata
    """
    return ModelMetadata()


def get_compatible_versions(version: str) -> list[str]:
    """Get list of compatible versions for a given version.
    
    This can be extended to support version compatibility matrices.
    
    Args:
        version: Model version string
        
    Returns:
        List of compatible version strings
    """
    # For now, only same version is compatible
    # This can be extended with a compatibility matrix
    return [version]


def check_embedding_compatibility(
    enrollment_metadata: Dict[str, Any],
    current_metadata: Optional[ModelMetadata] = None
) -> tuple[bool, Optional[str]]:
    """Check if enrollment embeddings are compatible with current model.
    
    Args:
        enrollment_metadata: Metadata from stored enrollment embeddings
        current_metadata: Current model metadata (if None, uses current)
        
    Returns:
        Tuple of (is_compatible, warning_message)
    """
    if current_metadata is None:
        current_metadata = get_current_model_metadata()
    
    # Extract model metadata from enrollment
    enrollment_model = ModelMetadata.from_dict(enrollment_metadata)
    
    # Check compatibility
    if not current_metadata.is_compatible_with(enrollment_model):
        warning = (
            f"Enrollment embeddings were created with model version {enrollment_model.internal_version} "
            f"(revision: {enrollment_model.model_revision}), but current model is version "
            f"{current_metadata.internal_version} (revision: {current_metadata.model_revision}). "
            f"Please re-enroll your voice for best accuracy."
        )
        return False, warning
    
    # Check if revision changed (warning but not blocking)
    if enrollment_model.model_revision != current_metadata.model_revision:
        warning = (
            f"Model revision changed from {enrollment_model.model_revision} to "
            f"{current_metadata.model_revision}. Consider re-enrolling for optimal accuracy."
        )
        return True, warning
    
    return True, None


def store_model_metadata_for_verification(
    session_id: str,
    uid: str,
    model_metadata: ModelMetadata
) -> None:
    """Store model metadata for a verification operation.
    
    This creates an audit trail of which model version was used for verification.
    
    Args:
        session_id: Session ID
        uid: User ID
        model_metadata: Model metadata used for verification
    """
    try:
        db = firestore.client()
        verification_ref = db.collection('listening_sessions').document(session_id)
        verification_ref.update({
            'verificationModelMetadata': model_metadata.to_dict(),
            'verificationModelVersion': model_metadata.internal_version,
        })
    except Exception as e:
        print(f"[MODEL_VERSIONING] Error storing verification metadata: {e}")

