"""
Unit tests for event management
"""

import re
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest
import pytz
from icalendar import Calendar as iCalendar
from icalendar import Event as iEvent

from chronos_mcp.calendars import CalendarManager
from chronos_mcp.events import EventManager


class TestEventManager:
    """Test event management functionality"""

    @pytest.fixture
    def mock_calendar_manager(self):
        """Mock CalendarManager"""
        return Mock(spec=CalendarManager)

    @pytest.fixture
    def mock_calendar(self):
        """Mock calendar object"""
        calendar = Mock()
        calendar.save_event = Mock()
        calendar.events = Mock()
        return calendar

    @pytest.fixture
    def sample_event_data(self):
        """Sample event data for testing"""
        return {
            "calendar_uid": "cal-123",
            "summary": "Test Meeting",
            "start": datetime(2025, 7, 10, 14, 0, tzinfo=pytz.UTC),
            "end": datetime(2025, 7, 10, 15, 0, tzinfo=pytz.UTC),
            "description": "Test Description",
            "location": "Conference Room A",
            "account_alias": "test_account",
        }

    def test_init(self, mock_calendar_manager):
        """Test EventManager initialization"""
        mgr = EventManager(mock_calendar_manager)
        assert mgr.calendars == mock_calendar_manager

    def test_create_event_calendar_not_found(self, mock_calendar_manager, sample_event_data):
        """Test creating event when calendar not found"""
        mock_calendar_manager.get_calendar.return_value = None
        mgr = EventManager(mock_calendar_manager)

        # Should raise CalendarNotFoundError
        from chronos_mcp.exceptions import CalendarNotFoundError

        with pytest.raises(CalendarNotFoundError) as exc_info:
            mgr.create_event(**sample_event_data)

        assert "cal-123" in str(exc_info.value)
        mock_calendar_manager.get_calendar.assert_called_once()

    @patch("chronos_mcp.events.uuid.uuid4")
    def test_create_event_success(
        self, mock_uuid, mock_calendar_manager, mock_calendar, sample_event_data
    ):
        """Test successful event creation"""
        mock_uuid.return_value = "evt-test-123"
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_caldav_event = Mock()
        mock_calendar.save_event.return_value = mock_caldav_event

        mgr = EventManager(mock_calendar_manager)

        result = mgr.create_event(**sample_event_data)

        assert result is not None
        assert result.uid == "evt-test-123"
        assert result.summary == "Test Meeting"
        assert result.description == "Test Description"
        assert result.location == "Conference Room A"
        assert result.calendar_uid == "cal-123"
        assert result.account_alias == "test_account"

        # Verify calendar.save_event was called with proper ical data
        mock_calendar.save_event.assert_called_once()
        ical_data = mock_calendar.save_event.call_args[0][0]
        assert "BEGIN:VCALENDAR" in ical_data

    @patch("chronos_mcp.events.uuid.uuid4")
    def test_create_event_preserves_timezone_as_utc(
        self, mock_uuid, mock_calendar_manager, mock_calendar
    ):
        """Test that non-UTC timezone is converted to UTC in iCal output (issue #17)"""
        mock_uuid.return_value = "evt-tz-test"
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.save_event.return_value = Mock()

        mgr = EventManager(mock_calendar_manager)

        eastern = pytz.timezone("US/Eastern")
        start_local = eastern.localize(datetime(2025, 7, 10, 14, 0))
        end_local = eastern.localize(datetime(2025, 7, 10, 15, 0))

        result = mgr.create_event(
            calendar_uid="cal-123",
            summary="Timezone Test",
            start=start_local,
            end=end_local,
        )

        assert result is not None
        ical_data = mock_calendar.save_event.call_args[0][0]
        assert "BEGIN:VCALENDAR" in ical_data
        assert "SUMMARY:Timezone Test" in ical_data
        # DTSTART and DTEND should be in UTC (14:00 EDT = 18:00 UTC)
        assert "DTSTART:20250710T180000Z" in ical_data
        assert "DTEND:20250710T190000Z" in ical_data

    def test_create_event_with_attendees(
        self, mock_calendar_manager, mock_calendar, sample_event_data
    ):
        """Test creating event with attendees"""
        mock_calendar_manager.get_calendar.return_value = mock_calendar

        attendees = [
            {
                "email": "user1@example.com",
                "name": "User One",
                "role": "REQ-PARTICIPANT",
            },
            {
                "email": "user2@example.com",
                "name": "User Two",
                "role": "OPT-PARTICIPANT",
                "rsvp": False,
            },
        ]
        sample_event_data["attendees"] = attendees

        mgr = EventManager(mock_calendar_manager)
        result = mgr.create_event(**sample_event_data)

        assert result is not None
        assert len(result.attendees) == 2
        assert result.attendees[0].email == "user1@example.com"
        assert result.attendees[1].role == "OPT-PARTICIPANT"

        # Check ical contains attendees
        ical_data = mock_calendar.save_event.call_args[0][0]
        assert "ATTENDEE" in ical_data
        assert "mailto:user1@example.com" in ical_data

    def test_create_event_with_alarm(self, mock_calendar_manager, mock_calendar, sample_event_data):
        """Test creating event with alarm"""
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        sample_event_data["alarm_minutes"] = 15

        mgr = EventManager(mock_calendar_manager)
        result = mgr.create_event(**sample_event_data)

        assert result is not None
        assert result.alarms is not None
        assert len(result.alarms) == 1
        assert result.alarms[0].trigger == "-PT15M"
        assert result.alarms[0].action == "DISPLAY"

        # Check ical contains alarm
        ical_data = mock_calendar.save_event.call_args[0][0]
        assert "BEGIN:VALARM" in ical_data
        assert "TRIGGER:-PT15M" in ical_data

    def test_create_event_with_recurrence(
        self, mock_calendar_manager, mock_calendar, sample_event_data
    ):
        """Test creating recurring event"""
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        sample_event_data["recurrence_rule"] = "FREQ=WEEKLY;BYDAY=MO,WE,FR;COUNT=10"

        mgr = EventManager(mock_calendar_manager)
        result = mgr.create_event(**sample_event_data)

        assert result is not None
        assert result.recurrence_rule == "FREQ=WEEKLY;BYDAY=MO,WE,FR;COUNT=10"

        # Check ical contains rrule (iCalendar reorders: COUNT before BYDAY)
        ical_data = mock_calendar.save_event.call_args[0][0]
        assert "RRULE:FREQ=WEEKLY;COUNT=10;BYDAY=MO,WE,FR" in ical_data

    def test_create_event_all_day(self, mock_calendar_manager, mock_calendar):
        """Test creating all-day event"""
        mock_calendar_manager.get_calendar.return_value = mock_calendar

        mgr = EventManager(mock_calendar_manager)
        result = mgr.create_event(
            calendar_uid="cal-123",
            summary="All Day Event",
            start=datetime(2025, 7, 10, 0, 0, tzinfo=pytz.UTC),
            end=datetime(2025, 7, 11, 0, 0, tzinfo=pytz.UTC),
            all_day=True,
        )

        assert result is not None
        assert result.all_day is True

    def test_create_event_exception(self, mock_calendar_manager, mock_calendar, sample_event_data):
        """Test event creation with exception"""
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.save_event.side_effect = Exception("CalDAV error")

        mgr = EventManager(mock_calendar_manager)

        # Should raise EventCreationError
        from chronos_mcp.exceptions import EventCreationError

        with pytest.raises(EventCreationError) as exc_info:
            mgr.create_event(**sample_event_data)

        assert "CalDAV error" in str(exc_info.value)

    def test_get_events_range_calendar_not_found(self, mock_calendar_manager):
        """Test getting events when calendar not found"""
        mock_calendar_manager.get_calendar.return_value = None

        mgr = EventManager(mock_calendar_manager)

        # Should raise CalendarNotFoundError
        from chronos_mcp.exceptions import CalendarNotFoundError

        with pytest.raises(CalendarNotFoundError) as exc_info:
            mgr.get_events_range(
                calendar_uid="cal-123",
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=1),
            )

        assert "cal-123" in str(exc_info.value)

    def test_get_events_range_success(self, mock_calendar_manager, mock_calendar):
        """Test successful event range retrieval"""
        mock_calendar_manager.get_calendar.return_value = mock_calendar

        # Create mock CalDAV events
        mock_event1 = Mock()
        mock_event1.data = """BEGIN:VEVENT
UID:evt-1
SUMMARY:Event 1
DTSTART:20250710T140000Z
DTEND:20250710T150000Z
END:VEVENT"""

        mock_event2 = Mock()
        mock_event2.data = """BEGIN:VEVENT
UID:evt-2
SUMMARY:Event 2
DTSTART:20250710T160000Z
DTEND:20250710T170000Z
DESCRIPTION:Test description
LOCATION:Room B
END:VEVENT"""

        mock_calendar.date_search.return_value = [mock_event1, mock_event2]

        mgr = EventManager(mock_calendar_manager)
        result = mgr.get_events_range(
            calendar_uid="cal-123",
            start_date=datetime(2025, 7, 10, 0, 0, tzinfo=pytz.UTC),
            end_date=datetime(2025, 7, 11, 0, 0, tzinfo=pytz.UTC),
        )

        assert len(result) == 2
        assert result[0].uid == "evt-1"
        assert result[0].summary == "Event 1"
        assert result[1].uid == "evt-2"
        assert result[1].summary == "Event 2"
        assert result[1].description == "Test description"
        assert result[1].location == "Room B"

    def test_get_events_range_with_attendees(self, mock_calendar_manager, mock_calendar):
        """Test getting events with attendees"""
        mock_calendar_manager.get_calendar.return_value = mock_calendar

        mock_event = Mock()
        mock_event.data = """BEGIN:VEVENT
UID:evt-3
SUMMARY:Meeting
DTSTART:20250710T140000Z
DTEND:20250710T150000Z
ATTENDEE;CN=User One;ROLE=REQ-PARTICIPANT:mailto:user1@example.com
ATTENDEE;CN=User Two;ROLE=OPT-PARTICIPANT;RSVP=FALSE:mailto:user2@example.com
END:VEVENT"""

        mock_calendar.date_search.return_value = [mock_event]

        mgr = EventManager(mock_calendar_manager)
        result = mgr.get_events_range(
            calendar_uid="cal-123",
            start_date=datetime(2025, 7, 10, tzinfo=pytz.UTC),
            end_date=datetime(2025, 7, 11, tzinfo=pytz.UTC),
        )

        assert len(result) == 1
        assert len(result[0].attendees) == 2
        assert result[0].attendees[0].email == "user1@example.com"
        assert result[0].attendees[0].name == "User One"
        assert result[0].attendees[1].role == "OPT-PARTICIPANT"

    def test_get_events_range_exception(self, mock_calendar_manager, mock_calendar):
        """Test event retrieval with exception"""
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.events.side_effect = Exception("CalDAV error")

        mgr = EventManager(mock_calendar_manager)
        result = mgr.get_events_range(
            calendar_uid="cal-123",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=1),
        )

        assert result == []

    def test_delete_event_calendar_not_found(self, mock_calendar_manager):
        """Test deleting event when calendar not found"""
        from unittest.mock import ANY

        from chronos_mcp.exceptions import CalendarNotFoundError

        mock_calendar_manager.get_calendar.return_value = None

        mgr = EventManager(mock_calendar_manager)

        # Should raise CalendarNotFoundError
        with pytest.raises(CalendarNotFoundError) as exc_info:
            mgr.delete_event("cal-123", "evt-123")

        assert "cal-123" in str(exc_info.value)
        mock_calendar_manager.get_calendar.assert_called_once_with("cal-123", None, request_id=ANY)

    def test_create_event_with_valid_rrule(self, mock_calendar_manager, mock_calendar):
        """Test creating event with valid RRULE"""
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.save_event.return_value = Mock()

        mgr = EventManager(mock_calendar_manager)

        # Test with daily recurrence
        event = mgr.create_event(
            calendar_uid="cal-123",
            summary="Daily Standup",
            start=datetime.now(),
            end=datetime.now() + timedelta(hours=1),
            recurrence_rule="FREQ=DAILY;BYDAY=MO,TU,WE,TH,FR;COUNT=20",
        )

        assert event is not None
        assert event.summary == "Daily Standup"
        assert event.recurrence_rule == "FREQ=DAILY;BYDAY=MO,TU,WE,TH,FR;COUNT=20"

        # Verify the iCalendar was created with RRULE (iCalendar reorders: COUNT before BYDAY)
        mock_calendar.save_event.assert_called_once()
        ical_data = mock_calendar.save_event.call_args[0][0]
        assert "RRULE:FREQ=DAILY;COUNT=20;BYDAY=MO,TU,WE,TH,FR" in ical_data

    def test_create_event_with_invalid_rrule(self, mock_calendar_manager, mock_calendar):
        """Test creating event with invalid RRULE raises error"""
        from chronos_mcp.exceptions import EventCreationError

        mock_calendar_manager.get_calendar.return_value = mock_calendar

        mgr = EventManager(mock_calendar_manager)

        # Test with invalid RRULE
        with pytest.raises(EventCreationError) as exc_info:
            mgr.create_event(
                calendar_uid="cal-123",
                summary="Bad Recurring Event",
                start=datetime.now(),
                end=datetime.now() + timedelta(hours=1),
                recurrence_rule="INVALID=RRULE",
            )

        assert "Invalid RRULE" in str(exc_info.value)
        # Should not have called save_event due to validation failure
        mock_calendar.save_event.assert_not_called()

    def test_update_event_success(self, mock_calendar_manager, mock_calendar):
        """Test successful event update"""

        # Setup
        mock_calendar_manager.get_calendar.return_value = mock_calendar

        # Create mock CalDAV event
        mock_caldav_event = MagicMock()

        # Create test iCalendar data
        cal = iCalendar()
        event = iEvent()
        event.add("uid", "evt-123")
        event.add("summary", "Original Title")
        event.add("description", "Original Description")
        event.add("dtstart", datetime.now())
        event.add("dtend", datetime.now() + timedelta(hours=1))
        event.add("location", "Original Location")
        cal.add_component(event)

        mock_caldav_event.data = cal.to_ical().decode("utf-8")
        mock_calendar.event_by_uid.return_value = mock_caldav_event

        mgr = EventManager(mock_calendar_manager)

        # Update event
        updated_event = mgr.update_event(  # noqa: F841
            calendar_uid="cal-123",
            event_uid="evt-123",
            summary="Updated Title",
            description="Updated Description",
        )

        # Verify update was called
        mock_caldav_event.save.assert_called_once()

        # Verify the event data was updated
        saved_data = mock_caldav_event.data
        assert "Updated Title" in saved_data
        assert "Updated Description" in saved_data
        assert "Original Location" in saved_data  # Unchanged field

    def test_update_event_last_modified_is_ical_encoded(self, mock_calendar_manager, mock_calendar):
        """LAST-MODIFIED must serialize as an iCalendar UTC stamp, not a Python repr.

        Regression: assigning a datetime via __setitem__ bypasses icalendar's
        property encoding and emits "2026-08-18 02:28:05.609518+00:00", which
        Fastmail rejects with HTTP 403 on every update. Tasks and journals
        already used the delete-then-add idiom; events did not.
        """

        mock_calendar_manager.get_calendar.return_value = mock_calendar

        mock_caldav_event = MagicMock()

        cal = iCalendar()
        event = iEvent()
        event.add("uid", "evt-lastmod")
        event.add("summary", "Original Title")
        event.add("dtstart", datetime.now())
        event.add("dtend", datetime.now() + timedelta(hours=1))
        cal.add_component(event)

        mock_caldav_event.data = cal.to_ical().decode("utf-8")
        mock_calendar.event_by_uid.return_value = mock_caldav_event

        mgr = EventManager(mock_calendar_manager)
        mgr.update_event(
            calendar_uid="cal-123",
            event_uid="evt-lastmod",
            summary="Updated Title",
        )

        saved_data = mock_caldav_event.data
        lastmod = [line for line in saved_data.splitlines() if line.startswith("LAST-MODIFIED")]
        assert len(lastmod) == 1, f"expected exactly one LAST-MODIFIED, got {lastmod}"

        value = lastmod[0].split(":", 1)[1].strip()
        # Valid form: 20260818T022805Z — no spaces, no microseconds, trailing Z.
        assert re.fullmatch(r"\d{8}T\d{6}Z", value), f"malformed LAST-MODIFIED: {value!r}"

    def test_update_event_partial_update(self, mock_calendar_manager, mock_calendar):
        """Test updating only specific fields"""

        mock_calendar_manager.get_calendar.return_value = mock_calendar

        # Create mock CalDAV event with full data
        mock_caldav_event = MagicMock()
        cal = iCalendar()
        event = iEvent()
        event.add("uid", "evt-123")
        event.add("summary", "Original Title")
        event.add("description", "Original Description")
        event.add("dtstart", datetime.now())
        event.add("dtend", datetime.now() + timedelta(hours=1))
        event.add("location", "Conference Room A")
        event.add("rrule", "FREQ=WEEKLY;BYDAY=MO")
        cal.add_component(event)

        mock_caldav_event.data = cal.to_ical().decode("utf-8")
        mock_calendar.event_by_uid.return_value = mock_caldav_event

        mgr = EventManager(mock_calendar_manager)

        # Update only location
        mgr.update_event(calendar_uid="cal-123", event_uid="evt-123", location="Conference Room B")

        # Verify save was called
        mock_caldav_event.save.assert_called_once()

        # Verify only location changed
        saved_data = mock_caldav_event.data
        assert "Original Title" in saved_data
        assert "Original Description" in saved_data
        assert "Conference Room B" in saved_data
        assert "FREQ=WEEKLY;BYDAY=MO" in saved_data

    def test_update_event_remove_optional_fields(self, mock_calendar_manager, mock_calendar):
        """Test removing optional fields by setting them to empty string"""

        mock_calendar_manager.get_calendar.return_value = mock_calendar

        # Create event with optional fields
        mock_caldav_event = MagicMock()
        cal = iCalendar()
        event = iEvent()
        event.add("uid", "evt-123")
        event.add("summary", "Meeting")
        event.add("description", "Team sync")
        event.add("location", "Room 101")
        event.add("dtstart", datetime.now())
        event.add("dtend", datetime.now() + timedelta(hours=1))
        cal.add_component(event)

        mock_caldav_event.data = cal.to_ical().decode("utf-8")
        mock_calendar.event_by_uid.return_value = mock_caldav_event

        mgr = EventManager(mock_calendar_manager)

        # Remove description and location
        mgr.update_event(
            calendar_uid="cal-123",
            event_uid="evt-123",
            description="",  # Empty string removes field
            location="",  # Empty string removes field
        )

        saved_data = mock_caldav_event.data
        assert "Meeting" in saved_data  # Summary unchanged
        assert "Team sync" not in saved_data  # Description removed
        assert "Room 101" not in saved_data  # Location removed

    def test_update_event_not_found(self, mock_calendar_manager, mock_calendar):
        """Test updating non-existent event"""
        from chronos_mcp.exceptions import EventNotFoundError

        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.side_effect = Exception("Not found")
        mock_calendar.events.return_value = []  # No events

        mgr = EventManager(mock_calendar_manager)

        with pytest.raises(EventNotFoundError) as exc_info:
            mgr.update_event(calendar_uid="cal-123", event_uid="non-existent", summary="New Title")

        assert "non-existent" in str(exc_info.value)

    def test_update_event_invalid_rrule(self, mock_calendar_manager, mock_calendar):
        """Test updating event with invalid RRULE"""
        from chronos_mcp.exceptions import EventCreationError

        mock_calendar_manager.get_calendar.return_value = mock_calendar

        # Create simple event
        mock_caldav_event = MagicMock()
        cal = iCalendar()
        event = iEvent()
        event.add("uid", "evt-123")
        event.add("summary", "Meeting")
        event.add("dtstart", datetime.now())
        event.add("dtend", datetime.now() + timedelta(hours=1))
        cal.add_component(event)

        mock_caldav_event.data = cal.to_ical().decode("utf-8")
        mock_calendar.event_by_uid.return_value = mock_caldav_event

        mgr = EventManager(mock_calendar_manager)

        # Try to update with invalid RRULE
        with pytest.raises(EventCreationError) as exc_info:
            mgr.update_event(
                calendar_uid="cal-123",
                event_uid="evt-123",
                recurrence_rule="INVALID=RRULE",
            )

        assert "Invalid RRULE" in str(exc_info.value)
        # Verify save was NOT called due to validation failure
        mock_caldav_event.save.assert_not_called()
