"""
Utility functions for Chronos MCP
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil import parser
from icalendar import Event as iEvent
from icalendar import Timezone as iTimezone

from .logging_config import setup_logging

logger = setup_logging()

_OFFSET_RE = re.compile(r"^(?:UTC)?([+-])(\d{1,2}):?(\d{2})$", re.IGNORECASE)


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


def resolve_timezone(tz_name: Optional[str]) -> Optional[timezone]:
    """Resolve a timezone name to a tzinfo.

    Accepts either an IANA name ('Europe/Prague') or a fixed UTC offset
    ('+02:00', '-0530', 'UTC+05:30'). Returns None if tz_name is None.
    Raises ValueError for anything else unrecognized.
    """
    if tz_name is None:
        return None
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        pass

    match = _OFFSET_RE.match(tz_name.strip())
    if not match:
        raise ValueError(
            f"Unknown timezone: '{tz_name}' (expected an IANA name like "
            "'Europe/Prague', or a UTC offset like '+02:00')"
        )
    sign, hours, minutes = match.groups()
    hours, minutes = int(hours), int(minutes)
    if hours > 23 or minutes > 59:
        raise ValueError(f"Unknown timezone: '{tz_name}' (offset out of range)")
    total_minutes = hours * 60 + minutes
    if sign == "-":
        total_minutes = -total_minutes
    try:
        return timezone(timedelta(minutes=total_minutes))
    except ValueError:
        raise ValueError(f"Unknown timezone: '{tz_name}' (offset out of range)")


def apply_timezone(dt: datetime, tz: Optional[timezone]) -> datetime:
    """Attach/convert dt to an explicit zone (named or fixed-offset).

    A naive dt is localized to tz; an aware dt is converted to tz. If tz is
    None, dt is returned unchanged (see ensure_ical_safe for the storage
    fallback applied when no explicit zone was requested).
    """
    if tz is None:
        return dt
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def normalize_datetime_for_storage(
    dt: datetime, tz: Optional[timezone]
) -> datetime:
    """Prepare dt for icalendar serialization.

    With an explicit tz (from resolve_timezone), the caller asked for that
    exact zone/offset -- honor it as-is. Without one, fall back to
    ensure_ical_safe so incidental fixed-offset input (e.g. free-text
    "...+02:00" parsed with no explicit timezone= given) doesn't get
    silently corrupted into icalendar's ambiguous floating-time format.
    """
    if tz is not None:
        return apply_timezone(dt, tz)
    return ensure_ical_safe(dt)


def fixed_offset_vtimezone(tz: timezone) -> iTimezone:
    """Build a VTIMEZONE component for a bare fixed-offset tzinfo.

    icalendar's Calendar.add_missing_timezones() only knows how to build
    VTIMEZONE blocks for real IANA zones (via zoneinfo lookup) -- it can't
    synthesize one for a plain `datetime.timezone(offset)`, yet that's
    exactly what icalendar itself uses as the TZID for such a tzinfo
    (its .tzname(None), e.g. "UTC+05:30"). Without a matching VTIMEZONE,
    the emitted DTSTART;TZID=... would dangle. This builds the matching
    single-phase (no DST) VTIMEZONE so the reference resolves.
    """
    name = tz.tzname(None)
    offset = tz.utcoffset(None)
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    offset_str = f"{sign}{total_seconds // 3600:02d}{(total_seconds % 3600) // 60:02d}"
    ics = (
        "BEGIN:VTIMEZONE\r\n"
        f"TZID:{name}\r\n"
        "BEGIN:STANDARD\r\n"
        "DTSTART:19700101T000000\r\n"
        f"TZOFFSETFROM:{offset_str}\r\n"
        f"TZOFFSETTO:{offset_str}\r\n"
        f"TZNAME:{name}\r\n"
        "END:STANDARD\r\n"
        "END:VTIMEZONE\r\n"
    )
    return iTimezone.from_ical(ics)


def ensure_fixed_offset_vtimezone(cal, tz: Optional[timezone]) -> None:
    """Insert a fixed-offset VTIMEZONE into cal if tz needs one and it's missing.

    No-op for named zones (ZoneInfo) -- those are handled by
    Calendar.add_missing_timezones(). Also a no-op for a zero offset, since
    icalendar always renders that as plain UTC ("Z" suffix, no TZID) rather
    than a named/offset zone. Call before cal.add_missing_timezones().
    """
    if tz is None or isinstance(tz, ZoneInfo):
        return
    if tz.utcoffset(None) == timedelta(0):
        return
    tzid = tz.tzname(None)
    existing = {getattr(vt, "tz_name", None) for vt in cal.timezones}
    if tzid in existing:
        return
    cal.subcomponents.insert(0, fixed_offset_vtimezone(tz))


def ensure_ical_safe(dt: datetime) -> datetime:
    """Normalize dt so icalendar serializes it unambiguously.

    icalendar only emits a TZID parameter (and, combined with
    Calendar.add_missing_timezones(), a VTIMEZONE block) when dt.tzinfo is a
    *named* zone (e.g. ZoneInfo, pytz) that exposes a key/zone identifier.
    A bare fixed-offset tzinfo (dateutil tzoffset, datetime.timezone) has no
    such identifier, so icalendar silently drops the offset and writes a
    "floating" local time with no Z suffix and no TZID -- ambiguous, and
    wrong for whatever offset the caller actually meant. Converting those to
    UTC keeps the value unambiguous. A naive dt is treated as UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    if hasattr(dt.tzinfo, "key"):  # zoneinfo.ZoneInfo, or pytz zones via .zone
        return dt
    if getattr(dt.tzinfo, "zone", None):
        return dt
    return dt.astimezone(timezone.utc)


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
