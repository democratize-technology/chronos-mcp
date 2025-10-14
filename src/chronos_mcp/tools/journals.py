"""
Journal management tools for Chronos MCP
"""

import uuid
from typing import Any

from pydantic import Field

from ..exceptions import (
    CalendarNotFoundError,
    ChronosError,
    ErrorSanitizer,
    ValidationError,
)
from ..logging_config import setup_logging
from ..utils import parse_datetime
from ..validation import InputValidator
from .base import create_success_response, handle_tool_errors


logger = setup_logging()

# Module-level managers dictionary for dependency injection
_managers: dict[str, Any] = {}


# Journal tool functions - defined as standalone functions for importability
async def create_journal(
    calendar_uid: str = Field(..., description="Calendar UID"),
    summary: str = Field(..., description="Journal entry title/summary"),
    description: str | None = Field(None, description="Journal entry content"),
    entry_date: str | None = Field(None, description="Journal entry date (ISO format)"),
    related_to: list[str] | None = Field(
        None, description="List of related component UIDs"
    ),
    account: str | None = Field(None, description="Account alias"),
) -> dict[str, Any]:
    """Create a new journal entry"""
    request_id = str(uuid.uuid4())

    try:
        # Validate and sanitize text inputs
        try:
            summary = InputValidator.validate_text_field(
                summary, "summary", required=True
            )
            if description:
                description = InputValidator.validate_text_field(
                    description, "description"
                )
        except ValidationError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": "VALIDATION_ERROR",
                "request_id": request_id,
            }

        # Parse entry date if provided
        entry_dt = None
        if entry_date:
            entry_dt = parse_datetime(entry_date)

        journal = _managers["journal_manager"].create_journal(
            calendar_uid=calendar_uid,
            summary=summary,
            description=description,
            dtstart=entry_dt,
            related_to=related_to,
            account_alias=account,
            request_id=request_id,
        )

        return {
            "success": True,
            "journal": {
                "uid": journal.uid,
                "summary": journal.summary,
                "description": journal.description,
                "entry_date": (
                    journal.dtstart.isoformat() if journal.dtstart else None
                ),
                "related_to": journal.related_to,
            },
            "request_id": request_id,
        }

    except ChronosError as e:
        e.request_id = request_id
        logger.error(f"Create journal failed: {e}")
        return {
            "success": False,
            "error": ErrorSanitizer.get_user_friendly_message(e),
            "error_code": e.error_code,
            "request_id": request_id,
        }

    except Exception as e:
        chronos_error = ChronosError(
            message=f"Failed to create journal: {e!s}",
            details={
                "tool": "create_journal",
                "summary": summary,
                "calendar_uid": calendar_uid,
                "original_error": str(e),
                "original_type": type(e).__name__,
            },
            request_id=request_id,
        )
        logger.error(f"Unexpected error in create_journal: {chronos_error}")
        return {
            "success": False,
            "error": ErrorSanitizer.get_user_friendly_message(chronos_error),
            "error_code": chronos_error.error_code,
            "request_id": request_id,
        }


async def list_journals(
    calendar_uid: str = Field(..., description="Calendar UID"),
    account: str | None = Field(None, description="Account alias"),
    limit: int | str | None = Field(
        50, description="Maximum number of journals to return"
    ),
) -> dict[str, Any]:
    """List journal entries in a calendar"""
    request_id = str(uuid.uuid4())

    # Handle type conversion for limit parameter
    if limit is not None:
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            return {
                "journals": [],
                "total": 0,
                "error": f"Invalid limit value: {limit}. Must be an integer",
                "error_code": "VALIDATION_ERROR",
                "request_id": request_id,
            }

    try:
        journals = _managers["journal_manager"].list_journals(
            calendar_uid=calendar_uid,
            limit=limit,
            account_alias=account,
        )

        return {
            "journals": [
                {
                    "uid": journal.uid,
                    "summary": journal.summary,
                    "description": journal.description,
                    "entry_date": (
                        journal.dtstart.isoformat() if journal.dtstart else None
                    ),
                    "related_to": journal.related_to,
                }
                for journal in journals
            ],
            "total": len(journals),
            "calendar_uid": calendar_uid,
            "request_id": request_id,
        }

    except CalendarNotFoundError as e:
        e.request_id = request_id
        logger.error(f"Calendar not found for journal listing: {e}")
        return {
            "journals": [],
            "total": 0,
            "error": ErrorSanitizer.get_user_friendly_message(e),
            "error_code": e.error_code,
            "request_id": request_id,
        }

    except ChronosError as e:
        e.request_id = request_id
        logger.error(f"List journals failed: {e}")
        return {
            "journals": [],
            "total": 0,
            "error": ErrorSanitizer.get_user_friendly_message(e),
            "error_code": e.error_code,
            "request_id": request_id,
        }

    except Exception as e:
        chronos_error = ChronosError(
            message=f"Failed to list journals: {e!s}",
            details={
                "tool": "list_journals",
                "calendar_uid": calendar_uid,
                "original_error": str(e),
                "original_type": type(e).__name__,
            },
            request_id=request_id,
        )
        logger.error(f"Unexpected error in list_journals: {chronos_error}")
        return {
            "journals": [],
            "total": 0,
            "error": ErrorSanitizer.get_user_friendly_message(chronos_error),
            "error_code": chronos_error.error_code,
            "request_id": request_id,
        }


@handle_tool_errors
async def update_journal(
    calendar_uid: str = Field(..., description="Calendar UID"),
    journal_uid: str = Field(..., description="Journal UID to update"),
    summary: str | None = Field(None, description="Journal entry title/summary"),
    description: str | None = Field(None, description="Journal entry content"),
    entry_date: str | None = Field(None, description="Journal entry date (ISO format)"),
    account: str | None = Field(None, description="Account alias"),
    request_id: str | None = None,
) -> dict[str, Any]:
    """Update an existing journal entry. Only provided fields will be updated."""
    # Validate inputs
    if summary is not None:
        summary = InputValidator.validate_text_field(summary, "summary", required=True)
    if description is not None:
        description = InputValidator.validate_text_field(description, "description")

    # Parse entry date if provided
    entry_dt = None
    if entry_date is not None:
        entry_dt = parse_datetime(entry_date)

    updated_journal = _managers["journal_manager"].update_journal(
        calendar_uid=calendar_uid,
        journal_uid=journal_uid,
        summary=summary,
        description=description,
        dtstart=entry_dt,
        account_alias=account,
        request_id=request_id,
    )

    return create_success_response(
        message=f"Journal '{journal_uid}' updated successfully",
        request_id=request_id,
        journal={
            "uid": updated_journal.uid,
            "summary": updated_journal.summary,
            "description": updated_journal.description,
            "entry_date": (
                updated_journal.dtstart.isoformat() if updated_journal.dtstart else None
            ),
            "related_to": updated_journal.related_to,
        },
    )


@handle_tool_errors
async def delete_journal(
    calendar_uid: str = Field(..., description="Calendar UID"),
    journal_uid: str = Field(..., description="Journal UID to delete"),
    account: str | None = Field(None, description="Account alias"),
    request_id: str | None = None,
) -> dict[str, Any]:
    """Delete a journal entry"""
    _managers["journal_manager"].delete_journal(
        calendar_uid=calendar_uid,
        journal_uid=journal_uid,
        account_alias=account,
        request_id=request_id,
    )

    return create_success_response(
        message=f"Journal '{journal_uid}' deleted successfully",
        request_id=request_id,
    )


def register_journal_tools(mcp, managers):
    """Register journal management tools with the MCP server"""

    # Update module-level managers for dependency injection
    _managers.update(managers)

    # Register all journal tools with the MCP server
    mcp.tool(create_journal)
    mcp.tool(list_journals)
    mcp.tool(update_journal)
    mcp.tool(delete_journal)


# Add .fn attribute to each function for backwards compatibility with tests
create_journal.fn = create_journal
list_journals.fn = list_journals
update_journal.fn = update_journal
delete_journal.fn = delete_journal


# Export all tools for backwards compatibility
__all__ = [
    "create_journal",
    "delete_journal",
    "list_journals",
    "register_journal_tools",
    "update_journal",
]
