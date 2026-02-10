"""
Event operations for Chronos MCP
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import caldav
from caldav import Event as CalDAVEvent
from icalendar import Calendar as iCalendar
from icalendar import Event as iEvent

from .caldav_utils import get_item_with_fallback
from .calendars import CalendarManager
from .exceptions import (
    CalendarNotFoundError,
    EventCreationError,
    EventDeletionError,
    EventNotFoundError,
)
from .logging_config import setup_logging
from .models import Alarm, Attendee, Event
from .utils import ical_to_datetime, validate_rrule

logger = setup_logging()


class EventManager:
    """Manage calendar events"""

    def __init__(self, calendar_manager: CalendarManager):
        self.calendars = calendar_manager

    def _get_default_account(self) -> Optional[str]:
        try:
            return self.calendars.accounts.config.config.default_account
        except Exception:
            return None

    def create_event(
        self,
        calendar_uid: str,
        summary: str,
        start: datetime,
        end: datetime,
        description: Optional[str] = None,
        location: Optional[str] = None,
        all_day: bool = False,
        attendees: Optional[List[Dict[str, Any]]] = None,
        alarm_minutes: Optional[int] = None,
        recurrence_rule: Optional[str] = None,
        related_to: Optional[List[str]] = None,
        account_alias: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[Event]:
        """Create a new event - raises exceptions on failure"""
        request_id = request_id or str(uuid.uuid4())

        calendar = self.calendars.get_calendar(
            calendar_uid, account_alias, request_id=request_id
        )
        if not calendar:
            raise CalendarNotFoundError(
                calendar_uid, account_alias, request_id=request_id
            )

        try:
            # Validate RRULE if provided
            if recurrence_rule:
                is_valid, error_msg = validate_rrule(recurrence_rule)
                if not is_valid:
                    raise EventCreationError(
                        summary, f"Invalid RRULE: {error_msg}", request_id=request_id
                    )

            cal = iCalendar()
            event = iEvent()

            # Generate UID if not provided
            event_uid = str(uuid.uuid4())

            event.add("uid", event_uid)
            event.add("summary", summary)
            if all_day:
                event.add("dtstart", start.date())
                event.add("dtend", end.date())
            else:
                event.add("dtstart", start)
                event.add("dtend", end)
            event.add("dtstamp", datetime.now(timezone.utc))

            if description:
                event.add("description", description)
            if location:
                event.add("location", location)
            if recurrence_rule:
                event.add("rrule", recurrence_rule)

            if attendees:
                for att in attendees:
                    attendee_str = f"mailto:{att['email']}"
                    event.add(
                        "attendee",
                        attendee_str,
                        parameters={
                            "CN": att.get("name", att["email"]),
                            "ROLE": att.get("role", "REQ-PARTICIPANT"),
                            "PARTSTAT": att.get("status", "NEEDS-ACTION"),
                            "RSVP": "TRUE" if att.get("rsvp", True) else "FALSE",
                        },
                    )

            if related_to:
                for related_uid in related_to:
                    event.add("related-to", related_uid)

            if alarm_minutes:
                from icalendar import Alarm as iAlarm

                alarm = iAlarm()
                alarm.add("action", "DISPLAY")
                alarm.add("trigger", timedelta(minutes=-alarm_minutes))
                alarm.add("description", summary)
                event.add_component(alarm)

            cal.add_component(event)

            # Save to CalDAV server
            caldav_event = calendar.save_event(cal.to_ical().decode("utf-8"))

            event_model = Event(
                uid=event_uid,
                summary=summary,
                description=description,
                start=start,
                end=end,
                all_day=all_day,
                location=location,
                calendar_uid=calendar_uid,
                account_alias=account_alias or self._get_default_account() or "default",
                recurrence_rule=recurrence_rule,
                related_to=related_to or [],
            )

            # Add attendees to model
            if attendees:
                event_model.attendees = [
                    Attendee(
                        email=att["email"],
                        name=att.get("name", att["email"]),
                        role=att.get("role", "REQ-PARTICIPANT"),
                        status=att.get("status", "NEEDS-ACTION"),
                        rsvp=att.get("rsvp", True),
                    )
                    for att in attendees
                ]

            # Add alarm to model
            if alarm_minutes:
                event_model.alarms = [
                    Alarm(
                        action="DISPLAY",
                        trigger=f"-PT{alarm_minutes}M",
                        description=summary,
                    )
                ]

            return event_model

        except caldav.lib.error.AuthorizationError as e:
            logger.error(
                f"Authorization error creating event '{summary}': {e}",
                extra={"request_id": request_id},
            )
            raise EventCreationError(
                summary, "Authorization failed", request_id=request_id
            )
        except Exception as e:
            logger.error(
                f"Error creating event '{summary}': {e}",
                extra={"request_id": request_id},
            )
            raise EventCreationError(summary, str(e), request_id=request_id)

    def get_events_range(
        self,
        calendar_uid: str,
        start_date: datetime,
        end_date: datetime,
        account_alias: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> List[Event]:
        """Get events within a date range - raises exceptions on failure"""
        request_id = request_id or str(uuid.uuid4())

        calendar = self.calendars.get_calendar(
            calendar_uid, account_alias, request_id=request_id
        )
        if not calendar:
            raise CalendarNotFoundError(
                calendar_uid, account_alias, request_id=request_id
            )

        events = []
        try:
            # Search for events in date range
            results = calendar.date_search(start=start_date, end=end_date, expand=True)

            for caldav_event in results:
                event_data = self._parse_caldav_event(
                    caldav_event, calendar_uid, account_alias
                )
                if event_data:
                    events.append(event_data)

        except Exception as e:
            logger.error(f"Error getting events: {e}")

        return events

    def _parse_caldav_event(
        self, caldav_event: CalDAVEvent, calendar_uid: str, account_alias: Optional[str]
    ) -> Optional[Event]:
        """Parse CalDAV event to Event model"""
        try:
            # Parse iCalendar data
            ical = iCalendar.from_ical(caldav_event.data)

            for component in ical.walk():
                if component.name == "VEVENT":
                    # Parse date/time values
                    dtstart = component.get("dtstart")
                    dtend = component.get("dtend")

                    start_dt = ical_to_datetime(dtstart)
                    end_dt = ical_to_datetime(dtend)

                    # Detect all-day events
                    # Check if the original values were DATE (not DATE-TIME) or if it's midnight to midnight
                    is_all_day = False
                    if dtstart and dtend:
                        # Check if values are DATE type (no time component)
                        if hasattr(dtstart, "dt") and not hasattr(dtstart.dt, "hour"):
                            is_all_day = True
                        # Also check for midnight-to-midnight pattern
                        elif (
                            start_dt.hour == 0
                            and start_dt.minute == 0
                            and start_dt.second == 0
                            and end_dt.hour == 0
                            and end_dt.minute == 0
                            and end_dt.second == 0
                            and (end_dt - start_dt).days >= 1
                        ):
                            is_all_day = True

                    # Parse basic event data
                    event = Event(
                        uid=str(component.get("uid", "")),
                        summary=str(component.get("summary", "No Title")),
                        description=(
                            str(component.get("description", ""))
                            if component.get("description")
                            else None
                        ),
                        start=start_dt,
                        end=end_dt,
                        all_day=is_all_day,
                        location=(
                            str(component.get("location", ""))
                            if component.get("location")
                            else None
                        ),
                        calendar_uid=calendar_uid,
                        account_alias=account_alias
                        or self._get_default_account()
                        or "default",
                        recurrence_rule=(
                            str(component.get("rrule", ""))
                            if component.get("rrule")
                            else None
                        ),
                    )

                    # Parse attendees
                    attendees = component.get("attendee", [])
                    if attendees:
                        if not isinstance(attendees, list):
                            attendees = [attendees]

                        for attendee in attendees:
                            params = (
                                attendee.params if hasattr(attendee, "params") else {}
                            )
                            email = str(attendee).replace("mailto:", "")
                            event.attendees.append(
                                Attendee(
                                    email=email,
                                    name=params.get("CN", email),
                                    role=params.get("ROLE", "REQ-PARTICIPANT"),
                                    status=params.get("PARTSTAT", "NEEDS-ACTION"),
                                    rsvp=params.get("RSVP", "TRUE").upper() == "TRUE",
                                )
                            )

                    return event

        except Exception as e:
            logger.error(f"Error parsing event: {e}")

        return None

    def delete_event(
        self,
        calendar_uid: str,
        event_uid: str,
        account_alias: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> bool:
        """Delete an event by UID - raises exceptions on failure"""
        request_id = request_id or str(uuid.uuid4())

        calendar = self.calendars.get_calendar(
            calendar_uid, account_alias, request_id=request_id
        )
        if not calendar:
            raise CalendarNotFoundError(
                calendar_uid, account_alias, request_id=request_id
            )

        try:
            # Use utility function to find event with automatic fallback
            event = get_item_with_fallback(
                calendar, event_uid, "event", request_id=request_id
            )
            event.delete()
            logger.info(
                f"Deleted event '{event_uid}'",
                extra={"request_id": request_id},
            )
            return True
        except ValueError:
            # get_item_with_fallback raises ValueError when not found
            raise EventNotFoundError(event_uid, calendar_uid, request_id=request_id)

        except EventNotFoundError:
            raise  # Re-raise our own exception
        except caldav.lib.error.AuthorizationError as e:
            logger.error(
                f"Authorization error deleting event '{event_uid}': {e}",
                extra={"request_id": request_id},
            )
            raise EventDeletionError(
                event_uid, "Authorization failed", request_id=request_id
            )
        except Exception as e:
            logger.error(
                f"Error deleting event '{event_uid}': {e}",
                extra={"request_id": request_id},
            )
            raise EventDeletionError(event_uid, str(e), request_id=request_id)

    def update_event(
        self,
        calendar_uid: str,
        event_uid: str,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        location: Optional[str] = None,
        all_day: Optional[bool] = None,
        attendees: Optional[List[Dict[str, Any]]] = None,
        alarm_minutes: Optional[int] = None,
        recurrence_rule: Optional[str] = None,
        account_alias: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[Event]:
        """Update an existing event - raises exceptions on failure

        Only provided fields will be updated. Other fields remain unchanged.
        """
        request_id = request_id or str(uuid.uuid4())

        calendar = self.calendars.get_calendar(
            calendar_uid, account_alias, request_id=request_id
        )
        if not calendar:
            raise CalendarNotFoundError(
                calendar_uid, account_alias, request_id=request_id
            )

        try:
            # Get existing event using utility function with automatic fallback
            try:
                caldav_event = get_item_with_fallback(
                    calendar, event_uid, "event", request_id=request_id
                )
            except ValueError:
                raise EventNotFoundError(event_uid, calendar_uid, request_id=request_id)
            # Parse existing event data
            ical = iCalendar.from_ical(caldav_event.data)
            existing_event = None

            for component in ical.walk():
                if component.name == "VEVENT":
                    existing_event = component
                    break

            if not existing_event:
                raise EventCreationError(
                    f"Event {event_uid}",
                    "Could not parse existing event data",
                    request_id=request_id,
                )

            # Validate RRULE if provided
            if recurrence_rule is not None:
                is_valid, error_msg = validate_rrule(recurrence_rule)
                if not is_valid:
                    raise EventCreationError(
                        summary or str(existing_event.get("summary", "")),
                        f"Invalid RRULE: {error_msg}",
                        request_id=request_id,
                    )

            # Update only provided fields
            if summary is not None:
                existing_event["summary"] = summary

            if description is not None:
                if description:
                    existing_event["description"] = description
                elif "description" in existing_event:
                    del existing_event["description"]

            if start is not None:
                if all_day:
                    existing_event["dtstart"].dt = start.date()
                else:
                    existing_event["dtstart"].dt = start

            if end is not None:
                if all_day:
                    existing_event["dtend"].dt = end.date()
                else:
                    existing_event["dtend"].dt = end

            if location is not None:
                if location:
                    existing_event["location"] = location
                elif "location" in existing_event:
                    del existing_event["location"]
            if recurrence_rule is not None:
                if recurrence_rule:
                    existing_event["rrule"] = recurrence_rule
                elif "rrule" in existing_event:
                    del existing_event["rrule"]

            # Update attendees if provided
            if attendees is not None:
                # Remove existing attendees
                if "attendee" in existing_event:
                    del existing_event["attendee"]

                # Add new attendees
                for att in attendees:
                    attendee_str = f"mailto:{att['email']}"
                    existing_event.add(
                        "attendee",
                        attendee_str,
                        parameters={
                            "CN": att.get("name", att["email"]),
                            "ROLE": att.get("role", "REQ-PARTICIPANT"),
                            "PARTSTAT": att.get("status", "NEEDS-ACTION"),
                            "RSVP": "TRUE" if att.get("rsvp", True) else "FALSE",
                        },
                    )

            # Update alarm if provided
            if alarm_minutes is not None:
                # Remove existing alarms
                for component in list(existing_event.subcomponents):
                    if component.name == "VALARM":
                        existing_event.subcomponents.remove(component)

                # Add new alarm if specified
                if alarm_minutes > 0:
                    from icalendar import Alarm as iAlarm

                    alarm = iAlarm()
                    alarm.add("action", "DISPLAY")
                    alarm.add("trigger", timedelta(minutes=-alarm_minutes))
                    alarm.add("description", existing_event.get("summary", ""))
                    existing_event.add_component(alarm)

            # Update last-modified timestamp
            existing_event["last-modified"] = datetime.now(timezone.utc)

            # Save the updated event
            caldav_event.data = ical.to_ical().decode("utf-8")
            caldav_event.save()
            # Parse and return the updated event
            return self._parse_caldav_event(caldav_event, calendar_uid, account_alias)

        except EventNotFoundError:
            raise
        except EventCreationError:
            raise
        except Exception as e:
            logger.error(
                f"Error updating event '{event_uid}': {e}",
                extra={"request_id": request_id},
            )
            raise EventCreationError(
                event_uid, f"Failed to update event: {str(e)}", request_id=request_id
            )
