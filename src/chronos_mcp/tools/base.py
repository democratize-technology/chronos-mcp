"""
Base utilities for MCP tools
"""

import uuid
from functools import wraps
from typing import Any

from ..exceptions import ChronosError, ErrorSanitizer
from ..logging_config import setup_logging


logger = setup_logging()


def handle_tool_errors(func):
    """Decorator to handle common error patterns in tools"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        request_id = str(uuid.uuid4())
        kwargs["request_id"] = request_id

        try:
            return await func(*args, **kwargs)
        except ChronosError as e:
            e.request_id = request_id
            logger.error(
                f"Request {request_id} failed: {ErrorSanitizer.sanitize_message(str(e))}"
            )
            return {
                "success": False,
                "error": ErrorSanitizer.sanitize_message(str(e)),
                "error_code": type(e).__name__,
                "request_id": request_id,
            }
        except Exception as e:
            sanitized_error = ErrorSanitizer.sanitize_message(str(e))
            logger.error(
                f"Unexpected error in request {request_id}: {type(e).__name__}: {sanitized_error}"
            )
            return {
                "success": False,
                "error": f"Error: {type(e).__name__}: {sanitized_error}",
                "error_code": type(e).__name__,
                "request_id": request_id,
            }

    return wrapper


def create_success_response(message: str, request_id: str, **kwargs) -> dict[str, Any]:
    """Create a standardized success response"""
    response = {
        "success": True,
        "message": message,
        "request_id": request_id,
    }
    response.update(kwargs)
    return response
