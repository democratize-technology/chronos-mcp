"""
Utility functions for Chronos MCP
"""

from datetime import datetime, timezone
from typing import Optional, Tuple, Union

from dateutil import parser
from icalendar import Event as iEvent

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

    Args:
        rrule: The RRULE string to validate

    Returns:
        tuple: (is_valid, error_message)
    """
    if not rrule:
        return True, None

    try:
        # Basic validation - must have FREQ
        if not rrule.startswith("FREQ="):
            return False, "RRULE must start with FREQ="

        # Parse components
        parts = rrule.split(";")
        rules = {}

        for part in parts:
            if "=" not in part:
                return False, f"Invalid RRULE component: {part}"

            key, value = part.split("=", 1)
            rules[key] = value

        # Validate FREQ is present and valid
        if "FREQ" not in rules:
            return False, "FREQ is required in RRULE"

        valid_freqs = ["DAILY", "WEEKLY", "MONTHLY", "YEARLY"]
        if rules["FREQ"] not in valid_freqs:
            return (
                False,
                f"Invalid FREQ value: {rules['FREQ']}. Must be one of {valid_freqs}",
            )

        # Validate other common components
        if "INTERVAL" in rules:
            try:
                interval = int(rules["INTERVAL"])
                if interval < 1:
                    return False, "INTERVAL must be a positive integer"
            except ValueError:
                return False, "INTERVAL must be an integer"

        if "COUNT" in rules:
            try:
                count = int(rules["COUNT"])
                if count < 1:
                    return False, "COUNT must be a positive integer"
            except ValueError:
                return False, "COUNT must be an integer"

        if "UNTIL" in rules:
            # Basic format check for UNTIL (should be datetime)
            until = rules["UNTIL"]
            if not (len(until) >= 8 and until[0:8].isdigit()):
                return False, "UNTIL must be in YYYYMMDD or YYYYMMDDTHHMMSSZ format"

        if "BYDAY" in rules:
            # Validate day abbreviations
            valid_days = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
            days = rules["BYDAY"].split(",")
            for day in days:
                # Remove position prefix if present (e.g., 2MO for 2nd Monday)
                day_abbr = day.lstrip("-+0123456789")
                if day_abbr not in valid_days:
                    return False, f"Invalid day abbreviation: {day}"

        # If we get here, basic validation passed
        return True, None

    except Exception as e:
        return False, f"Error parsing RRULE: {str(e)}"
