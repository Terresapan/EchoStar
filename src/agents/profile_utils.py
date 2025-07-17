"""
Profile management utilities for handling user profile updates and deduplication.
Enhanced with comprehensive error handling and data validation.
"""
import logging
from typing import Optional, List, Dict, Any
from langgraph.store.memory import InMemoryStore
from .schemas import UserProfile
from .memory_error_handler import MemoryErrorHandler, ProfileSerializer

logger = logging.getLogger(__name__)


def search_existing_profile(store: InMemoryStore, user_id: str = "Lily") -> Optional[Dict[str, Any]]:
    """
    Search for existing user profile in the store.
    
    Args:
        store: The memory store instance
        user_id: The user identifier (defaults to "Lily")
        
    Returns:
        The existing profile data if found, None otherwise
    """
    try:
        namespace = ("echo_star", user_id, "profile")
        profiles = store.search(namespace)
        
        if profiles:
            # Return the most recent profile if multiple exist
            if len(profiles) > 1:
                logger.warning(f"Multiple profiles found for user {user_id}: {len(profiles)} profiles")
                # Sort by created_at timestamp and return the most recent
                sorted_profiles = sorted(profiles, key=lambda x: getattr(x, 'created_at', ''), reverse=True)
                return sorted_profiles[0].value
            else:
                return profiles[0].value
                
    except Exception as e:
        logger.error(f"Error searching for existing profile: {e}")
        
    return None


def merge_profile_data(existing_profile: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge new profile data with existing profile, prioritizing new information.
    
    This function intelligently merges profile data by:
    1. Preserving all existing information
    2. Adding new information when it's meaningful (not empty or generic)
    3. Updating existing fields only when new data is more specific
    4. Handling special cases for different field types
    
    Args:
        existing_profile: The current profile data
        new_data: New profile information to merge
        
    Returns:
        Merged profile data
    """
    # Start with existing profile data
    merged_data = existing_profile.copy()
    
    # Generic placeholders that should not overwrite existing data
    generic_placeholders = {
        "Background information is not provided yet.",
        "No specific information provided.",
        "Not specified",
        "Unknown",
        "",
        None
    }
    
    # Update with new data, but preserve existing data where new data is empty or generic
    for key, value in new_data.items():
        # Skip if new value is empty, None, or a generic placeholder
        if not value or value in generic_placeholders:
            logger.debug(f"Skipped empty/generic value for field '{key}': {value}")
            continue
            
        # Clean the value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        
        # Check if we should update this field
        existing_value = merged_data.get(key)
        
        if not existing_value or existing_value in generic_placeholders:
            # No existing value or existing value is generic, use new value
            merged_data[key] = value
            logger.debug(f"Added new value for field '{key}': {value[:50]}...")
        elif len(str(value)) > len(str(existing_value)) * 1.2:
            # New value is significantly longer (more detailed), use it
            merged_data[key] = value
            logger.debug(f"Updated field '{key}' with more detailed value")
        elif key in ['name'] and existing_value != value:
            # For critical fields like name, always update if different
            merged_data[key] = value
            logger.debug(f"Updated critical field '{key}' with new value")
        else:
            # Keep existing value as it's likely more established
            logger.debug(f"Preserved existing value for field '{key}'")
    
    return merged_data


def validate_profile_data(profile_data: Dict[str, Any]) -> bool:
    """
    Validate profile data before saving.
    
    Args:
        profile_data: Profile data to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Try to create a UserProfile instance to validate
        UserProfile(**profile_data)
        return True
    except Exception as e:
        logger.error(f"Profile validation failed: {e}")
        return False


def get_profile_summary(profile_data: Dict[str, Any]) -> str:
    """
    Get a summary of profile data for logging purposes.
    
    Args:
        profile_data: Profile data to summarize
        
    Returns:
        Summary string
    """
    summary_parts = []
    
    if profile_data.get('name'):
        summary_parts.append(f"name: {profile_data['name']}")
    
    if profile_data.get('background'):
        bg_preview = profile_data['background'][:50] + "..." if len(profile_data['background']) > 50 else profile_data['background']
        summary_parts.append(f"background: {bg_preview}")
    
    if profile_data.get('communication_style'):
        summary_parts.append(f"style: {profile_data['communication_style'][:30]}...")
    
    if profile_data.get('emotional_baseline'):
        summary_parts.append(f"emotion: {profile_data['emotional_baseline'][:30]}...")
    
    return " | ".join(summary_parts) if summary_parts else "empty profile"


def safe_update_or_create_profile(store: InMemoryStore, profile_data: Dict[str, Any], 
                                user_id: str = "Lily") -> bool:
    """
    Safe wrapper for update_or_create_profile that ensures conversation flow is never broken.
    This function guarantees that any profile storage errors are handled gracefully.
    
    Args:
        store: The memory store instance
        profile_data: New profile data to save
        user_id: The user identifier (defaults to "Lily")
        
    Returns:
        True if operation was successful, False otherwise (but never raises exceptions)
    """
    try:
        return update_or_create_profile(store, profile_data, user_id)
    except Exception as e:
        # Absolute safety net - ensure no exceptions escape this function
        logger.error(f"Critical error in profile storage, ensuring conversation continuity: {str(e)}")
        return False


def update_or_create_profile(store: InMemoryStore, profile_data: Dict[str, Any], 
                           user_id: str = "Lily") -> bool:
    """
    Update existing profile or create new one if none exists.
    This function ensures only one profile exists per user by using replace operations.
    Enhanced with comprehensive error handling and data validation.
    
    Args:
        store: The memory store instance
        profile_data: New profile data to save
        user_id: The user identifier (defaults to "Lily")
        
    Returns:
        True if operation was successful, False otherwise
    """
    try:
        namespace = ("echo_star", user_id, "profile")
        
        # Step 1: Pre-validation - Check if profile_data is valid before any processing
        if not isinstance(profile_data, dict):
            logger.error(f"Profile data must be a dictionary, got {type(profile_data).__name__}")
            return False
        
        if not profile_data:
            logger.error("Profile data cannot be empty")
            return False
        
        # Step 2: Enhanced validation using the error handler
        is_valid, validation_errors = MemoryErrorHandler.validate_hashable_structure(profile_data)
        if not is_valid:
            logger.warning(f"Profile data has hashable structure issues, attempting to sanitize: {validation_errors}")
            try:
                profile_data = MemoryErrorHandler.sanitize_for_storage(profile_data)
                logger.info("Profile data sanitized for storage compatibility")
            except Exception as sanitize_error:
                logger.error(f"Failed to sanitize profile data: {sanitize_error}")
                return False
        
        # Step 3: Additional profile-specific validation
        profile_valid, profile_errors = ProfileSerializer.validate_profile_structure(profile_data)
        if not profile_valid:
            logger.error(f"Profile structure validation failed: {profile_errors}")
            # Don't return False immediately - try to continue with available data
            logger.info("Attempting to continue with partial profile data despite validation errors")
        
        existing_profile = search_existing_profile(store, user_id)
        
        if existing_profile:
            # Update existing profile using replace operation
            logger.info(f"Updating existing profile for user {user_id}")
            merged_data = merge_profile_data(existing_profile, profile_data)
            
            # Serialize the merged data to ensure compatibility
            serialized_data = ProfileSerializer.serialize_profile(merged_data)
            
            # Validate merged data with UserProfile schema
            if not validate_profile_data(serialized_data):
                logger.warning("Merged profile data validation failed, attempting to create minimal valid profile")
                # Create a minimal valid profile to ensure conversation can continue
                minimal_profile = {
                    "name": serialized_data.get("name", "User"),
                    "background": "Profile data validation failed, using minimal profile",
                    "communication_style": "Standard",
                    "emotional_baseline": "Neutral"
                }
                if validate_profile_data(minimal_profile):
                    serialized_data = minimal_profile
                    logger.info("Using minimal valid profile to ensure conversation continuity")
                else:
                    logger.error("Even minimal profile validation failed, skipping profile storage")
                    return False
                
            validated_profile = UserProfile(**serialized_data)
            
            # Get all existing profiles to ensure we replace the right one and clean up duplicates
            existing_profiles = store.search(namespace)
            if existing_profiles:
                # Use the key from the first (most recent) profile for replacement
                primary_key = existing_profiles[0].key
                
                try:
                    # Replace the primary profile with updated data
                    profile_dict = validated_profile.dict()
                    store.put(namespace, primary_key, profile_dict)
                    logger.info(f"Successfully replaced profile for user {user_id}")
                    logger.debug(f"Profile summary: {get_profile_summary(profile_dict)}")
                    
                except Exception as storage_e:
                    # Attempt error recovery
                    if MemoryErrorHandler.handle_storage_error(storage_e, profile_dict, "profile_update"):
                        logger.info("Attempting profile storage with sanitized data")
                        sanitized_dict = MemoryErrorHandler.sanitize_for_storage(profile_dict)
                        store.put(namespace, primary_key, sanitized_dict)
                        logger.info("Profile storage successful after sanitization")
                    else:
                        logger.error("Profile storage failed even after error recovery attempt")
                        return False
                
                # Delete any duplicate profiles to ensure only one exists
                if len(existing_profiles) > 1:
                    logger.warning(f"Found {len(existing_profiles)} profiles, removing duplicates")
                    cleanup_errors = []
                    for duplicate in existing_profiles[1:]:
                        try:
                            store.delete(namespace, duplicate.key)
                            logger.debug(f"Deleted duplicate profile with key: {duplicate.key}")
                        except Exception as delete_error:
                            cleanup_errors.append(delete_error)
                            logger.error(f"Failed to delete duplicate profile: {delete_error}")
                    
                    if cleanup_errors:
                        error_summary = MemoryErrorHandler.create_error_summary(cleanup_errors, "profile_cleanup")
                        logger.warning(f"Some profile cleanup operations failed: {error_summary}")
                    else:
                        logger.info(f"Cleaned up {len(existing_profiles) - 1} duplicate profiles")
                
                return True
            else:
                logger.error("No existing profiles found despite search returning data")
                return False
                
        else:
            # Create new profile (first time user)
            logger.info(f"Creating new profile for user {user_id}")
            
            # Serialize the profile data to ensure compatibility
            serialized_data = ProfileSerializer.serialize_profile(profile_data)
            
            if not validate_profile_data(serialized_data):
                logger.warning("New profile data validation failed, attempting to create minimal valid profile")
                # Create a minimal valid profile to ensure conversation can continue
                minimal_profile = {
                    "name": serialized_data.get("name", "User"),
                    "background": "Profile data validation failed, using minimal profile",
                    "communication_style": "Standard",
                    "emotional_baseline": "Neutral"
                }
                if validate_profile_data(minimal_profile):
                    serialized_data = minimal_profile
                    logger.info("Using minimal valid profile for new user to ensure conversation continuity")
                else:
                    logger.error("Even minimal profile validation failed, skipping profile creation")
                    return False
                
            validated_profile = UserProfile(**serialized_data)
            
            # Generate a unique key for the new profile
            import uuid
            profile_key = str(uuid.uuid4())
            
            # Ensure no profiles exist before creating (safety check)
            existing_check = store.search(namespace)
            if existing_check:
                logger.warning(f"Found {len(existing_check)} existing profiles during creation, cleaning up first")
                for existing in existing_check:
                    try:
                        store.delete(namespace, existing.key)
                    except Exception as cleanup_e:
                        logger.warning(f"Failed to cleanup existing profile: {cleanup_e}")
            
            try:
                # Create the new profile
                profile_dict = validated_profile.dict()
                store.put(namespace, profile_key, profile_dict)
                logger.info(f"Successfully created new profile for user {user_id}")
                logger.debug(f"Profile summary: {get_profile_summary(profile_dict)}")
                return True
                
            except Exception as creation_e:
                # Attempt error recovery for profile creation
                if MemoryErrorHandler.handle_storage_error(creation_e, profile_dict, "profile_creation"):
                    logger.info("Attempting profile creation with sanitized data")
                    sanitized_dict = MemoryErrorHandler.sanitize_for_storage(profile_dict)
                    store.put(namespace, profile_key, sanitized_dict)
                    logger.info("Profile creation successful after sanitization")
                    return True
                else:
                    logger.error("Profile creation failed even after error recovery attempt")
                    return False
                
    except Exception as e:
        # Enhanced error logging with context
        MemoryErrorHandler.log_error_context(e, {
            "operation": "update_or_create_profile",
            "user_id": user_id,
            "profile_data_type": type(profile_data).__name__,
            "profile_keys": list(profile_data.keys()) if isinstance(profile_data, dict) else "not_dict"
        })
        return False


def replace_profile_directly(store: InMemoryStore, profile_data: Dict[str, Any], 
                           user_id: str = "Lily") -> bool:
    """
    Directly replace the user profile, ensuring only one profile exists.
    This is a more aggressive approach that completely replaces the profile.
    
    Args:
        store: The memory store instance
        profile_data: Complete profile data to save
        user_id: The user identifier (defaults to "Lily")
        
    Returns:
        True if operation was successful, False otherwise
    """
    try:
        namespace = ("echo_star", user_id, "profile")
        
        # Validate the profile data first
        if not validate_profile_data(profile_data):
            logger.error("Profile data validation failed for direct replacement")
            return False
            
        validated_profile = UserProfile(**profile_data)
        
        # Delete all existing profiles first
        existing_profiles = store.search(namespace)
        for existing in existing_profiles:
            try:
                store.delete(namespace, existing.key)
                logger.debug(f"Deleted existing profile with key: {existing.key}")
            except Exception as delete_error:
                logger.error(f"Failed to delete existing profile: {delete_error}")
        
        # Create the new profile with a fresh key
        import uuid
        profile_key = str(uuid.uuid4())
        store.put(namespace, profile_key, validated_profile.dict())
        
        logger.info(f"Successfully replaced profile for user {user_id}")
        logger.debug(f"Profile summary: {get_profile_summary(validated_profile.dict())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in replace_profile_directly: {e}")
        return False


def cleanup_duplicate_profiles(store: InMemoryStore, user_id: str = "Lily") -> int:
    """
    Clean up duplicate profiles, keeping only the most recent one.
    
    Args:
        store: The memory store instance
        user_id: The user identifier (defaults to "Lily")
        
    Returns:
        Number of duplicate profiles removed
    """
    try:
        namespace = ("echo_star", user_id, "profile")
        all_profiles = store.search(namespace)
        
        if len(all_profiles) <= 1:
            return 0
            
        logger.warning(f"Found {len(all_profiles)} profiles for user {user_id}, cleaning up duplicates")
        
        # Sort by created_at timestamp, keep the most recent
        sorted_profiles = sorted(all_profiles, key=lambda x: getattr(x, 'created_at', ''), reverse=True)
        
        # Delete all but the first (most recent) profile
        deleted_count = 0
        for duplicate in sorted_profiles[1:]:
            try:
                store.delete(namespace, duplicate.key)
                deleted_count += 1
            except Exception as delete_error:
                logger.error(f"Failed to delete duplicate profile: {delete_error}")
                
        logger.info(f"Successfully cleaned up {deleted_count} duplicate profiles for user {user_id}")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error in cleanup_duplicate_profiles: {e}")
        return 0