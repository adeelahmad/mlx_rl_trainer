"""
Example external data validation script.
"""
from typing import Dict, Any

def validate_sample(sample: Dict[str, Any]) -> bool:
    """
    Validates a single data sample.

    Args:
        sample (Dict[str, Any]): A dictionary representing a single data sample.

    Returns:
        bool: True if the sample is valid, False otherwise.
    """
    # Example validation: Ensure the prompt is not empty.
    if not sample.get("prompt", "").strip():
        return False
    
    # Example validation: Ensure the completion is not empty.
    if not sample.get("completion", "").strip():
        return False

    return True