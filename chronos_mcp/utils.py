"""
Utility functions for Chronos MCP
"""

from datetime import datetime, timezone
from typing import Optional, Tuple, Union

from dateutil import parser  # type: ignore[import-untyped]
from icalendar import Event as iEvent  # type: ignore[import-untyped]

from .logging_config import setup_logging

logger = setup_logging()


def parse_datetime(dt_str: Union[str, datetime]) -> datetime:
    """Parse datetime string or return datetime object"""
    if isinstance(dt_str, datetime):
        return dt_str

    # Try parsing with dateutil
    try:
        dt = parser.parse(dt_str.strip("\"'"))
        # Ensure timezone awareness
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception as e:
        logger.error(f"Error parsing datetime '{dt_str}': {e}")
        raise ValueError(f"Invalid datetime format: {dt_str}")


def datetime_to_ical(dt: datetime, all_day: bool = False) -> str:
    """Convert datetime to iCalendar format"""
    if all_day:
        return dt.strftime("%Y%m%d")
    else:
        # Ensure UTC timezone
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y%m%dT%H%M%SZ")


def ical_to_datetime(ical_dt) -> datetime:
    """Convert iCalendar datetime to Python datetime"""
    if hasattr(ical_dt, "dt"):
        dt = ical_dt.dt
    else:
        dt = ical_dt

    # Handle date-only (all-day events)
    if not isinstance(dt, datetime):
        dt = datetime.combine(dt, datetime.min.time())
        dt = dt.replace(tzinfo=timezone.utc)

    # Ensure timezone awareness
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


def create_ical_event(event_data: dict) -> iEvent:
    """Create iCalendar event from data"""
    event = iEvent()

    # Required fields
    event.add("uid", event_data.get("uid"))
    event.add("summary", event_data.get("summary"))
    event.add("dtstart", event_data.get("start"))
    event.add("dtend", event_data.get("end"))

    # Optional fields
    if "description" in event_data:
        event.add("description", event_data["description"])
    if "location" in event_data:
        event.add("location", event_data["location"])
    if "status" in event_data:
        event.add("status", event_data["status"])

    return event


def validate_rrule(rrule: str) -> Tuple[bool, Optional[str]]:
    """
    Validate RRULE syntax according to RFC 5545.

    Delegates to RRuleValidator for canonical validation.
    Preserves empty-input behavior for backward compatibility.

    Args:
        rrule: The RRULE string to validate

    Returns:
        tuple: (is_valid, error_message)
    """
    if not rrule:
        return True, None

    from .rrule import RRuleValidator

    return RRuleValidator.validate_rrule(rrule)
