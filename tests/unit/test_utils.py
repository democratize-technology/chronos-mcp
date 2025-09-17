"""
Unit tests for utility functions
"""

from datetime import date, datetime, timezone

import pytest
import pytz

from chronos_mcp.utils import (
    create_ical_event,
    datetime_to_ical,
    ical_to_datetime,
    parse_datetime,
)


class TestParseDatetime:
    """Test parse_datetime function"""

    def test_parse_datetime_object(self):
        """Test parsing when input is already a datetime"""
        dt = datetime(2025, 7, 10, 14, 0, tzinfo=timezone.utc)
        result = parse_datetime(dt)
        assert result == dt

    def test_parse_iso_string(self):
        """Test parsing ISO format string"""
        result = parse_datetime("2025-07-10T14:00:00Z")
        expected = datetime(2025, 7, 10, 14, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_parse_naive_datetime(self):
        """Test parsing datetime without timezone"""
        result = parse_datetime("2025-07-10 14:00:00")
        assert result.tzinfo == timezone.utc
        assert result.year == 2025
        assert result.month == 7
        assert result.day == 10
        assert result.hour == 14

    def test_parse_various_formats(self):
        """Test parsing various datetime formats"""
        formats = [
            "2025-07-10",
            "07/10/2025",
            "July 10, 2025",
            "2025-07-10T14:00:00+00:00",
            "2025-07-10T14:00:00-05:00",
        ]

        for fmt in formats:
            result = parse_datetime(fmt)
            assert isinstance(result, datetime)
            assert result.tzinfo is not None

    def test_parse_invalid_format(self):
        """Test parsing invalid datetime format"""
        with pytest.raises(ValueError, match="Invalid datetime format"):
            parse_datetime("not a date")


class TestDatetimeToIcal:
    """Test datetime_to_ical function"""

    def test_datetime_to_ical_regular(self):
        """Test converting regular datetime to iCal format"""
        dt = datetime(2025, 7, 10, 14, 30, 45, tzinfo=timezone.utc)
        result = datetime_to_ical(dt)
        assert result == "20250710T143045Z"

    def test_datetime_to_ical_all_day(self):
        """Test converting all-day event to iCal format"""
        dt = datetime(2025, 7, 10, 0, 0, 0, tzinfo=timezone.utc)
        result = datetime_to_ical(dt, all_day=True)
        assert result == "20250710"

    def test_datetime_to_ical_naive(self):
        """Test converting naive datetime (assumes UTC)"""
        dt = datetime(2025, 7, 10, 14, 30, 45)
        result = datetime_to_ical(dt)
        assert result == "20250710T143045Z"

    def test_datetime_to_ical_other_timezone(self):
        """Test converting datetime in non-UTC timezone"""
        eastern = pytz.timezone("US/Eastern")
        dt = eastern.localize(datetime(2025, 7, 10, 14, 30, 45))
        result = datetime_to_ical(dt)
        # Should be converted to UTC
        assert result.endswith("Z")
        assert "1830" in result or "1930" in result  # Accounts for DST


class TestIcalToDatetime:
    """Test ical_to_datetime function"""

    def test_ical_to_datetime_with_dt_attribute(self):
        """Test converting iCal object with dt attribute"""
        from icalendar import vDatetime

        ical_dt = vDatetime.from_ical("20250710T143045Z")
        result = ical_to_datetime(ical_dt)
        expected = datetime(2025, 7, 10, 14, 30, 45, tzinfo=timezone.utc)
        assert result == expected

    def test_ical_to_datetime_direct_datetime(self):
        """Test converting direct datetime object"""
        dt = datetime(2025, 7, 10, 14, 30, 45, tzinfo=timezone.utc)
        result = ical_to_datetime(dt)
        assert result == dt

    def test_ical_to_datetime_date_only(self):
        """Test converting date-only (all-day event)"""
        dt = date(2025, 7, 10)
        result = ical_to_datetime(dt)
        expected = datetime(2025, 7, 10, 0, 0, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_ical_to_datetime_naive(self):
        """Test converting naive datetime"""
        dt = datetime(2025, 7, 10, 14, 30, 45)
        result = ical_to_datetime(dt)
        assert result.tzinfo == timezone.utc
        assert result.replace(tzinfo=None) == dt


class TestCreateIcalEvent:
    """Test create_ical_event function"""

    def test_create_ical_event_minimal(self):
        """Test creating event with minimal data"""
        event_data = {
            "uid": "test-123",
            "summary": "Test Event",
            "start": datetime(2025, 7, 10, 14, 0, tzinfo=timezone.utc),
            "end": datetime(2025, 7, 10, 15, 0, tzinfo=timezone.utc),
        }

        event = create_ical_event(event_data)

        assert event["uid"] == "test-123"
        assert event["summary"] == "Test Event"
        assert event["dtstart"].dt == event_data["start"]
        assert event["dtend"].dt == event_data["end"]

    def test_create_ical_event_full(self):
        """Test creating event with all optional fields"""
        event_data = {
            "uid": "test-456",
            "summary": "Full Event",
            "start": datetime(2025, 7, 10, 14, 0, tzinfo=timezone.utc),
            "end": datetime(2025, 7, 10, 15, 0, tzinfo=timezone.utc),
            "description": "This is a test event",
            "location": "Conference Room A",
            "status": "CONFIRMED",
        }

        event = create_ical_event(event_data)

        assert event["uid"] == "test-456"
        assert event["summary"] == "Full Event"
        assert event["description"] == "This is a test event"
        assert event["location"] == "Conference Room A"
        assert event["status"] == "CONFIRMED"

    def test_create_ical_event_missing_optional(self):
        """Test creating event without optional fields"""
        event_data = {
            "uid": "test-789",
            "summary": "Basic Event",
            "start": datetime(2025, 7, 10, 14, 0, tzinfo=timezone.utc),
            "end": datetime(2025, 7, 10, 15, 0, tzinfo=timezone.utc),
        }

        event = create_ical_event(event_data)

        # Optional fields should not be present
        assert "description" not in event
        assert "location" not in event
        assert "status" not in event
