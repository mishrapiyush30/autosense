from typing import Optional, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


class QueryError(Exception):
    """Base exception for query-related errors."""
    pass


class EmptyQueryError(QueryError):
    """Raised when query is empty or contains only whitespace."""
    pass


class LongInputError(QueryError):
    """Raised when input exceeds maximum length."""
    pass


class InvalidVINError(QueryError):
    """Raised when VIN format is invalid."""
    pass


class InvalidDTCError(QueryError):
    """Raised when DTC code format is invalid."""
    pass


class MalformedInputError(QueryError):
    """Raised when input contains suspicious or malformed content."""
    pass


def validate_query(query: str) -> None:
    """Validate query input and raise appropriate errors."""
    if not query or not query.strip():
        raise EmptyQueryError("Query cannot be empty")
    
    if len(query) > 500:
        raise LongInputError("Query exceeds maximum length of 500 characters")
    
    # Check for suspicious content
    suspicious_patterns = [
        r'<script',  # Potential XSS
        r'javascript:',  # JavaScript injection
        r'data:text/html',  # Data URI injection
        r'vbscript:',  # VBScript injection
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise MalformedInputError(f"Query contains suspicious content: {pattern}")


def validate_vin(vin: Optional[str]) -> Optional[str]:
    """Validate VIN format and return normalized VIN."""
    if not vin:
        return None
    
    vin = vin.strip().upper()
    
    # VIN validation: 17 characters, no I, O, Q
    vin_pattern = r'^[A-HJ-NPR-Z0-9]{17}$'
    
    if not re.match(vin_pattern, vin):
        raise InvalidVINError(f"Invalid VIN format: {vin}. VIN must be 17 characters and contain only valid characters.")
    
    return vin


def validate_dtc_code(code: Optional[str]) -> Optional[str]:
    """Validate DTC code format and return normalized code."""
    if not code:
        return None
    
    code = code.strip().upper()
    
    # DTC validation: P/B/C/U followed by 4 digits
    dtc_pattern = r'^[PBCU][0-9]{4}$'
    
    if not re.match(dtc_pattern, code):
        raise InvalidDTCError(f"Invalid DTC code format: {code}. DTC must be P/B/C/U followed by 4 digits.")
    
    return code


def sanitize_input(text: str) -> str:
    """Sanitize input text to prevent injection attacks."""
    # Remove or escape potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&']
    for char in dangerous_chars:
        text = text.replace(char, '')
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text


def extract_and_validate_components(query: str) -> Dict[str, Any]:
    """Extract and validate VIN and DTC from query."""
    components = {
        'vin': None,
        'dtc_code': None,
        'sanitized_query': sanitize_input(query)
    }
    
    # Extract VIN
    vin_pattern = r'\b[A-HJ-NPR-Z0-9]{17}\b'
    vin_match = re.search(vin_pattern, query.upper())
    if vin_match:
        try:
            components['vin'] = validate_vin(vin_match.group())
        except InvalidVINError as e:
            logger.warning(f"Invalid VIN in query: {e}")
    
    # Extract DTC code
    dtc_pattern = r'\b[PBCU][0-9]{4}\b'
    dtc_match = re.search(dtc_pattern, query.upper())
    if dtc_match:
        try:
            components['dtc_code'] = validate_dtc_code(dtc_match.group())
        except InvalidDTCError as e:
            logger.warning(f"Invalid DTC code in query: {e}")
    
    return components


def handle_edge_cases(query: str, vin: Optional[str] = None) -> Dict[str, Any]:
    """Handle various edge cases and return processed components."""
    try:
        # Validate query
        validate_query(query)
        
        # Extract and validate components
        components = extract_and_validate_components(query)
        
        # Override with provided VIN if it's more specific
        if vin:
            try:
                components['vin'] = validate_vin(vin)
            except InvalidVINError as e:
                logger.warning(f"Provided VIN is invalid: {e}")
        
        # Add validation status
        components['is_valid'] = True
        components['errors'] = []
        
        return components
        
    except QueryError as e:
        logger.error(f"Query validation failed: {e}")
        return {
            'is_valid': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'sanitized_query': sanitize_input(query),
            'vin': None,
            'dtc_code': None
        }
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        return {
            'is_valid': False,
            'error': "Unexpected validation error",
            'error_type': 'ValidationError',
            'sanitized_query': sanitize_input(query),
            'vin': None,
            'dtc_code': None
        }


def create_error_response(error: QueryError, query: str) -> Dict[str, Any]:
    """Create a standardized error response."""
    return {
        'query': query,
        'error': str(error),
        'error_type': type(error).__name__,
        'suggestion': get_error_suggestion(error),
        'timestamp': None,
        'processing_time': 0
    }


def get_error_suggestion(error: QueryError) -> str:
    """Get helpful suggestion based on error type."""
    if isinstance(error, EmptyQueryError):
        return "Please provide a description of the problem or a DTC code."
    elif isinstance(error, LongInputError):
        return "Please keep your query under 500 characters."
    elif isinstance(error, InvalidVINError):
        return "VIN should be 17 characters long and contain only letters (excluding I, O, Q) and numbers."
    elif isinstance(error, InvalidDTCError):
        return "DTC code should start with P, B, C, or U followed by 4 digits (e.g., P0420)."
    elif isinstance(error, MalformedInputError):
        return "Please provide a valid automotive diagnostic query."
    else:
        return "Please check your input and try again."


# Convenience function for the main agent
def ensure_valid_input(query: str, vin: Optional[str] = None) -> Dict[str, Any]:
    """Main validation function for the agent."""
    return handle_edge_cases(query, vin) 