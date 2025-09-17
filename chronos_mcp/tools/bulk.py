"""
Bulk operation tools for Chronos MCP
"""

import uuid
from typing import Any, Dict, List, Optional

from pydantic import Field

from ..bulk import BulkOperationMode, BulkOptions
from ..exceptions import ChronosError, ErrorSanitizer, ValidationError
from ..logging_config import setup_logging
from ..rate_limiter import rate_limit
from ..validation import InputValidator
from .base import create_success_response, handle_tool_errors

logger = setup_logging()

# Module-level managers dictionary for dependency injection
_managers = {}


def _format_bulk_response(result, request_id: str, **extra_fields) -> Dict[str, Any]:
    """Format bulk operation response with consistent success indicators"""
    response = {
        "success": result.failed == 0,  # Only true if ALL succeed
        "partial_success": 0
        < result.successful
        < result.total,  # True for mixed results
        "total": result.total,
        "succeeded": result.successful,
        "failed": result.failed,
        "request_id": request_id,
    }

    # Add any extra fields
    response.update(extra_fields)

    return response


def _ensure_managers_initialized():
    """Ensure managers are initialized, with fallback to server-level managers"""
    if not _managers:
        try:
            # Try to import and use server-level managers for backwards compatibility
            from .. import server

            # Use the real bulk manager from the server
            bulk_manager = getattr(server, "bulk_manager", None)
            event_manager = getattr(server, "event_manager", None)
            task_manager = getattr(server, "task_manager", None)
            journal_manager = getattr(server, "journal_manager", None)

            if not bulk_manager:
                raise AttributeError("bulk_manager not found in server module")

            _managers.update(
                {
                    "bulk_manager": bulk_manager,
                    "event_manager": event_manager,
                    "task_manager": task_manager,
                    "journal_manager": journal_manager,
                }
            )
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to initialize managers: {str(e)}")
            raise RuntimeError(f"Manager initialization failed: {str(e)}")


# Bulk tool functions - defined as standalone functions for importability
@handle_tool_errors
@rate_limit("bulk")
async def bulk_create_events(
    calendar_uid: str = Field(..., description="Calendar UID"),
    events: List[Dict[str, Any]] = Field(
        ..., description="List of event data dictionaries"
    ),
    mode: str = Field("continue", description="Operation mode: continue, fail_fast"),
    validate_before_execute: bool = Field(
        True, description="Validate events before creation"
    ),
    account: Optional[str] = Field(None, description="Account alias"),
    request_id: str = None,
) -> Dict[str, Any]:
    """Create multiple events in bulk"""
    # Ensure managers are available for backwards compatibility with tests
    _ensure_managers_initialized()

    try:
        # Validate input
        if not isinstance(events, list):
            return {
                "success": False,
                "error": "Events must be a list",
                "error_code": "VALIDATION_ERROR",
                "request_id": request_id,
            }

        # Validate mode
        if mode not in ["continue", "fail_fast"]:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}. Must be 'continue' or 'fail_fast'",
                "error_code": "VALIDATION_ERROR",
                "request_id": request_id,
            }

        # Handle empty list
        if not events:
            return {
                "success": True,
                "total": 0,
                "succeeded": 0,
                "failed": 0,
                "details": [],
                "request_id": request_id,
            }

        # Convert mode to BulkOperationMode for compatibility
        if mode == "continue":
            bulk_mode = BulkOperationMode.CONTINUE_ON_ERROR
        else:  # fail_fast
            bulk_mode = BulkOperationMode.FAIL_FAST

        # Create bulk options
        options = BulkOptions(mode=bulk_mode)

        # Parse datetime fields and attendees JSON in events
        import json

        from ..utils import parse_datetime

        parsed_events = []
        for event in events:
            parsed_event = event.copy()

            # Normalize field names: convert "start"/"end" to "dtstart"/"dtend"
            if "start" in parsed_event:
                parsed_event["dtstart"] = parsed_event.pop("start")
            if "end" in parsed_event:
                parsed_event["dtend"] = parsed_event.pop("end")

            # Parse datetime fields
            if "dtstart" in parsed_event and isinstance(parsed_event["dtstart"], str):
                parsed_event["dtstart"] = parse_datetime(parsed_event["dtstart"])
            if "dtend" in parsed_event and isinstance(parsed_event["dtend"], str):
                parsed_event["dtend"] = parse_datetime(parsed_event["dtend"])

            # Parse alarm_minutes if it's a string
            if "alarm_minutes" in parsed_event and isinstance(
                parsed_event["alarm_minutes"], str
            ):
                try:
                    parsed_event["alarm_minutes"] = int(parsed_event["alarm_minutes"])
                except ValueError:
                    pass  # Keep as string if not a valid int

            # Parse attendees JSON if provided
            if "attendees_json" in parsed_event:
                try:
                    parsed_event["attendees"] = json.loads(
                        parsed_event["attendees_json"]
                    )
                    del parsed_event["attendees_json"]
                except json.JSONDecodeError:
                    pass  # Will be caught by validation

            parsed_events.append(parsed_event)

        # Execute bulk operation
        result = _managers["bulk_manager"].bulk_create_events(
            calendar_uid=calendar_uid,
            events=parsed_events,
            options=options,
            account_alias=account,
        )

        # Format response to match test expectations
        details = []
        for res in result.results:
            detail = {
                "index": res.index,
                "success": res.success,
            }
            if res.success:
                detail["uid"] = res.uid
                # Try to get summary from original event data
                if res.index < len(events):
                    detail["summary"] = events[res.index].get("summary")
            else:
                detail["error"] = res.error
            details.append(detail)

        return _format_bulk_response(
            result,
            request_id,
            details=details,
        )

    except ChronosError as e:
        e.request_id = request_id
        logger.error(f"Bulk create events failed: {e}")
        return {
            "success": False,
            "error": ErrorSanitizer.get_user_friendly_message(e),
            "error_code": e.error_code,
            "request_id": request_id,
        }

    except Exception as e:
        chronos_error = ChronosError(
            message=f"Failed to bulk create events: {str(e)}",
            details={
                "tool": "bulk_create_events",
                "calendar_uid": calendar_uid,
                "original_error": str(e),
                "original_type": type(e).__name__,
            },
            request_id=request_id,
        )
        logger.error(f"Unexpected error in bulk_create_events: {chronos_error}")
        return {
            "success": False,
            "error": ErrorSanitizer.get_user_friendly_message(chronos_error),
            "error_code": chronos_error.error_code,
            "request_id": request_id,
        }


@handle_tool_errors
@handle_tool_errors
@rate_limit("bulk")
async def bulk_delete_events(
    calendar_uid: str = Field(..., description="Calendar UID"),
    event_uids: List[str] = Field(..., description="List of event UIDs to delete"),
    mode: str = Field("continue", description="Operation mode"),
    parallel: bool = Field(True, description="Execute operations in parallel"),
    account: Optional[str] = Field(None, description="Account alias"),
    request_id: str = None,
) -> Dict[str, Any]:
    """Delete multiple events in bulk"""
    # Ensure managers are available for backwards compatibility with tests
    _ensure_managers_initialized()

    # Convert mode string to BulkOperationMode
    mode_mapping = {
        "continue": BulkOperationMode.CONTINUE_ON_ERROR,
        "fail_fast": BulkOperationMode.FAIL_FAST,
        "atomic": BulkOperationMode.ATOMIC,
    }

    if mode not in mode_mapping:
        raise ValidationError(
            f"Invalid mode: {mode}. Must be one of: continue, fail_fast, atomic"
        )

    bulk_mode = mode_mapping[mode]

    # Create bulk options
    options = BulkOptions(mode=bulk_mode)

    # Execute bulk operation
    result = _managers["bulk_manager"].bulk_delete_events(
        calendar_uid=calendar_uid,
        event_uids=event_uids,
        options=options,
        account_alias=account,
        request_id=request_id,
    )

    # Format response to match test expectations
    details = []
    for res in result.results:
        detail = {
            "index": res.index,
            "success": res.success,
            "uid": res.uid if res.uid else event_uids[res.index],
        }
        if not res.success:
            detail["error"] = res.error
        details.append(detail)

    return _format_bulk_response(
        result,
        request_id,
        details=details,
    )


@handle_tool_errors
@handle_tool_errors
@rate_limit("bulk")
async def bulk_create_tasks(
    calendar_uid: str = Field(..., description="Calendar UID"),
    tasks_json: str = Field(..., description="JSON array of task data"),
    mode: str = Field("continue", description="Operation mode"),
    parallel: bool = Field(True, description="Execute operations in parallel"),
    account: Optional[str] = Field(None, description="Account alias"),
    request_id: str = None,
) -> Dict[str, Any]:
    """Create multiple tasks in bulk"""
    import json

    # Ensure managers are available for backwards compatibility with tests
    _ensure_managers_initialized()

    # Parse tasks JSON
    try:
        tasks_data = json.loads(tasks_json)
        if not isinstance(tasks_data, list):
            raise ValueError("Tasks data must be a JSON array")
    except (json.JSONDecodeError, ValueError) as e:
        raise ValidationError(f"Invalid tasks JSON: {str(e)}")

    # Convert mode string to BulkOperationMode
    mode_mapping = {
        "continue": BulkOperationMode.CONTINUE_ON_ERROR,
        "fail_fast": BulkOperationMode.FAIL_FAST,
        "atomic": BulkOperationMode.ATOMIC,
    }

    if mode not in mode_mapping:
        raise ValidationError(
            f"Invalid mode: {mode}. Must be one of: continue, fail_fast, atomic"
        )

    bulk_mode = mode_mapping[mode]

    # Create bulk options
    options = BulkOptions(mode=bulk_mode)

    # Execute bulk operation
    if "bulk_manager" not in _managers or _managers["bulk_manager"] is None:
        raise RuntimeError("BulkOperationManager not available")

    try:
        result = _managers["bulk_manager"].bulk_create_tasks(
            calendar_uid=calendar_uid,
            tasks=tasks_data,
            options=options,
            account_alias=account,
        )
    except AttributeError as e:
        raise RuntimeError(f"BulkOperationManager missing method: {str(e)}")
    except Exception as e:
        logger.error(f"Bulk task creation failed: {type(e).__name__}: {str(e)}")
        raise

    return _format_bulk_response(
        result,
        request_id,
        message=f"Bulk task creation completed: {result.successful} created, {result.failed} failed",
        created_count=result.successful,
        failed_count=result.failed,
        results=result.results,
        errors=[r.error for r in result.results if r.error],
    )


@handle_tool_errors
@handle_tool_errors
@rate_limit("bulk")
async def bulk_delete_tasks(
    calendar_uid: str = Field(..., description="Calendar UID"),
    task_uids: List[str] = Field(..., description="List of task UIDs to delete"),
    mode: str = Field("continue", description="Operation mode"),
    parallel: bool = Field(True, description="Execute operations in parallel"),
    account: Optional[str] = Field(None, description="Account alias"),
    request_id: str = None,
) -> Dict[str, Any]:
    """Delete multiple tasks in bulk"""
    # Ensure managers are available for backwards compatibility with tests
    _ensure_managers_initialized()

    # Convert mode string to BulkOperationMode
    mode_mapping = {
        "continue": BulkOperationMode.CONTINUE_ON_ERROR,
        "fail_fast": BulkOperationMode.FAIL_FAST,
        "atomic": BulkOperationMode.ATOMIC,
    }

    if mode not in mode_mapping:
        raise ValidationError(
            f"Invalid mode: {mode}. Must be one of: continue, fail_fast, atomic"
        )

    bulk_mode = mode_mapping[mode]

    # Create bulk options
    options = BulkOptions(mode=bulk_mode)

    # Execute bulk operation
    result = _managers["bulk_manager"].bulk_delete_tasks(
        calendar_uid=calendar_uid,
        task_uids=task_uids,
        options=options,
        account_alias=account,
        request_id=request_id,
    )

    return _format_bulk_response(
        result,
        request_id,
        message=f"Bulk task deletion completed: {result.successful} deleted, {result.failed} failed",
        deleted_count=result.successful,
        failed_count=result.failed,
        results=result.results,
        errors=[r.error for r in result.results if r.error],
    )


@handle_tool_errors
@handle_tool_errors
@rate_limit("bulk")
async def bulk_create_journals(
    calendar_uid: str = Field(..., description="Calendar UID"),
    journals_json: str = Field(..., description="JSON array of journal data"),
    mode: str = Field("continue", description="Operation mode"),
    parallel: bool = Field(True, description="Execute operations in parallel"),
    account: Optional[str] = Field(None, description="Account alias"),
    request_id: str = None,
) -> Dict[str, Any]:
    """Create multiple journal entries in bulk"""
    import json

    # Ensure managers are available for backwards compatibility with tests
    _ensure_managers_initialized()

    # Parse journals JSON
    try:
        journals_data = json.loads(journals_json)
        if not isinstance(journals_data, list):
            raise ValueError("Journals data must be a JSON array")
    except (json.JSONDecodeError, ValueError) as e:
        raise ValidationError(f"Invalid journals JSON: {str(e)}")

    # Convert mode string to BulkOperationMode
    mode_mapping = {
        "continue": BulkOperationMode.CONTINUE_ON_ERROR,
        "fail_fast": BulkOperationMode.FAIL_FAST,
        "atomic": BulkOperationMode.ATOMIC,
    }

    if mode not in mode_mapping:
        raise ValidationError(
            f"Invalid mode: {mode}. Must be one of: continue, fail_fast, atomic"
        )

    bulk_mode = mode_mapping[mode]

    # Create bulk options
    options = BulkOptions(mode=bulk_mode)

    # Execute bulk operation
    if "bulk_manager" not in _managers or _managers["bulk_manager"] is None:
        raise RuntimeError("BulkOperationManager not available")

    try:
        result = _managers["bulk_manager"].bulk_create_journals(
            calendar_uid=calendar_uid,
            journals=journals_data,
            options=options,
            account_alias=account,
        )
    except AttributeError as e:
        raise RuntimeError(f"BulkOperationManager missing method: {str(e)}")
    except Exception as e:
        logger.error(f"Bulk journal creation failed: {type(e).__name__}: {str(e)}")
        raise

    return _format_bulk_response(
        result,
        request_id,
        message=f"Bulk journal creation completed: {result.successful} created, {result.failed} failed",
        created_count=result.successful,
        failed_count=result.failed,
        results=result.results,
        errors=[r.error for r in result.results if r.error],
    )


@handle_tool_errors
@handle_tool_errors
@rate_limit("bulk")
async def bulk_delete_journals(
    calendar_uid: str = Field(..., description="Calendar UID"),
    journal_uids: List[str] = Field(..., description="List of journal UIDs to delete"),
    mode: str = Field("continue", description="Operation mode"),
    parallel: bool = Field(True, description="Execute operations in parallel"),
    account: Optional[str] = Field(None, description="Account alias"),
    request_id: str = None,
) -> Dict[str, Any]:
    """Delete multiple journal entries in bulk"""
    # Ensure managers are available for backwards compatibility with tests
    _ensure_managers_initialized()

    # Convert mode string to BulkOperationMode
    mode_mapping = {
        "continue": BulkOperationMode.CONTINUE_ON_ERROR,
        "fail_fast": BulkOperationMode.FAIL_FAST,
        "atomic": BulkOperationMode.ATOMIC,
    }

    if mode not in mode_mapping:
        raise ValidationError(
            f"Invalid mode: {mode}. Must be one of: continue, fail_fast, atomic"
        )

    bulk_mode = mode_mapping[mode]

    # Create bulk options
    options = BulkOptions(mode=bulk_mode)

    # Execute bulk operation
    result = _managers["bulk_manager"].bulk_delete_journals(
        calendar_uid=calendar_uid,
        journal_uids=journal_uids,
        options=options,
        account_alias=account,
        request_id=request_id,
    )

    return _format_bulk_response(
        result,
        request_id,
        message=f"Bulk journal deletion completed: {result.successful} deleted, {result.failed} failed",
        deleted_count=result.successful,
        failed_count=result.failed,
        results=result.results,
        errors=[r.error for r in result.results if r.error],
    )


def register_bulk_tools(mcp, managers):
    """Register bulk operation tools with the MCP server"""

    # Update module-level managers for dependency injection
    _managers.update(managers)

    # Register all bulk tools with the MCP server
    mcp.tool(bulk_create_events)
    mcp.tool(bulk_delete_events)
    mcp.tool(bulk_create_tasks)
    mcp.tool(bulk_delete_tasks)
    mcp.tool(bulk_create_journals)
    mcp.tool(bulk_delete_journals)


# Add .fn attribute to each function for backwards compatibility with tests
# This mimics the behavior of FastMCP decorated functions
bulk_create_events.fn = bulk_create_events
bulk_delete_events.fn = bulk_delete_events
bulk_create_tasks.fn = bulk_create_tasks
bulk_delete_tasks.fn = bulk_delete_tasks
bulk_create_journals.fn = bulk_create_journals
bulk_delete_journals.fn = bulk_delete_journals


# Export all tools for backwards compatibility
__all__ = [
    "bulk_create_events",
    "bulk_delete_events",
    "bulk_create_tasks",
    "bulk_delete_tasks",
    "bulk_create_journals",
    "bulk_delete_journals",
    "register_bulk_tools",
]
