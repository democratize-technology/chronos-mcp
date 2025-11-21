"""
Calendar management tools for Chronos MCP
"""

import uuid
from typing import Any, Dict, Optional

from pydantic import Field

from ..exceptions import (
    CalendarNotFoundError,
    ChronosError,
    ErrorSanitizer,
    ValidationError,
)
from ..logging_config import setup_logging
from ..rate_limiter import rate_limit
from ..validation import InputValidator
from .base import create_success_response, handle_tool_errors

logger = setup_logging()

# Module-level managers dictionary for dependency injection
_managers = {}


# Calendar tool functions - defined as standalone functions for importability
@rate_limit("calendars")
async def list_calendars(
    account: Optional[str] = Field(
        None, description="Account alias (uses default if not specified)"
    ),
) -> Dict[str, Any]:
    """List all calendars for an account"""
    calendars = _managers["calendar_manager"].list_calendars(account)

    return {
        "calendars": [
            {
                "uid": cal.uid,
                "name": cal.name,
                "description": cal.description,
                "color": cal.color,
                "url": cal.url,
                "account": cal.account_alias,
            }
            for cal in calendars
        ],
        "total": len(calendars),
        "account": account or _managers["config_manager"].config.default_account,
    }


@handle_tool_errors
@rate_limit("calendars")
async def create_calendar(
    name: str = Field(..., description="Calendar name"),
    description: Optional[str] = Field(None, description="Calendar description"),
    color: Optional[str] = Field(None, description="Calendar color (hex format)"),
    account: Optional[str] = Field(
        None, description="Account alias (uses default if not specified)"
    ),
) -> Dict[str, Any]:
    """Create a new calendar"""
    try:
        # Validate inputs
        name = InputValidator.validate_text_field(name, "calendar_name", required=True)
        if description:
            description = InputValidator.validate_text_field(description, "description")
        if color and not InputValidator.PATTERNS["color"].match(color):
            raise ValidationError(
                "Invalid color format. Must be hex color like #FF0000"
            )
        if account:
            account = InputValidator.validate_text_field(
                account, "alias", required=False
            )
    except ValidationError as e:
        return {"success": False, "error": str(e)}

    calendar = _managers["calendar_manager"].create_calendar(
        name, description, color, account
    )

    if calendar:
        return {
            "success": True,
            "calendar": {
                "uid": calendar.uid,
                "name": calendar.name,
                "url": calendar.url,
            },
        }
    else:
        return {"success": False, "error": "Failed to create calendar"}


@handle_tool_errors
@rate_limit("calendars")
async def delete_calendar(
    calendar_uid: str = Field(..., description="Calendar UID to delete"),
    account: Optional[str] = Field(
        None, description="Account alias (uses default if not specified)"
    ),
    request_id: str = None,
) -> Dict[str, Any]:
    """Delete a calendar"""
    # Validate inputs
    calendar_uid = InputValidator.validate_uid(calendar_uid)
    if account:
        account = InputValidator.validate_text_field(account, "alias", required=False)

    _managers["calendar_manager"].delete_calendar(
        calendar_uid, account, request_id=request_id
    )

    return create_success_response(
        message=f"Calendar '{calendar_uid}' deleted successfully",
        request_id=request_id,
    )


def register_calendar_tools(mcp, managers):
    """Register calendar management tools with the MCP server"""

    # Update module-level managers for dependency injection
    _managers.update(managers)

    # Register all calendar tools with the MCP server
    mcp.tool(list_calendars)
    mcp.tool(create_calendar)
    mcp.tool(delete_calendar)


# Add .fn attribute to each function for backwards compatibility with tests
list_calendars.fn = list_calendars
create_calendar.fn = create_calendar
delete_calendar.fn = delete_calendar


# Export all tools for backwards compatibility
__all__ = [
    "list_calendars",
    "create_calendar",
    "delete_calendar",
    "register_calendar_tools",
]
