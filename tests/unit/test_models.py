"""
Unit tests for Chronos MCP models
"""

from datetime import datetime

import pytest
import pytz

from chronos_mcp.models import (
    Account,
    AccountStatus,
    Attendee,
    AttendeeRole,
    AttendeeStatus,
    Calendar,
    Event,
)


class TestAccount:
    def test_account_creation(self, sample_account):
        """Test creating an account model"""
        assert sample_account.alias == "test_account"
        assert sample_account.username == "testuser"
        assert sample_account.password == "testpass"
        assert sample_account.status == AccountStatus.UNKNOWN

    def test_account_without_password(self):
        """Test creating account without password"""
        account = Account(
            alias="no_pass", url="https://caldav.example.com", username="user"
        )
        assert account.password is None

    def test_account_url_validation(self):
        """Test URL validation"""
        with pytest.raises(ValueError):
            Account(alias="bad_url", url="not-a-url", username="user")


class TestCalendar:
    def test_calendar_creation(self, sample_calendar):
        """Test creating a calendar model"""
        assert sample_calendar.uid == "cal-123"
        assert sample_calendar.name == "Test Calendar"
        assert sample_calendar.color == "#FF0000"
        assert not sample_calendar.read_only

    def test_calendar_minimal(self):
        """Test calendar with minimal fields"""
        cal = Calendar(uid="minimal", name="Minimal Calendar", account_alias="test")
        assert cal.description is None
        assert cal.color is None


class TestEvent:
    def test_event_creation(self, sample_event):
        """Test creating event"""
        assert sample_event.summary == "Test Event"
        assert sample_event.all_day is False
        assert sample_event.attendees == []

    def test_all_day_event(self):
        """Test all-day event creation"""
        event = Event(
            uid="all-day-123",
            summary="All Day Event",
            start=datetime(2025, 7, 5, tzinfo=pytz.UTC),
            end=datetime(2025, 7, 6, tzinfo=pytz.UTC),
            all_day=True,
            calendar_uid="cal-123",
            account_alias="test",
        )
        assert event.all_day is True

    def test_event_with_attendees(self):
        """Test event with attendees"""
        attendee = Attendee(
            email="attendee@example.com",
            name="Test Attendee",
            role=AttendeeRole.REQ_PARTICIPANT,
            status=AttendeeStatus.ACCEPTED,
        )
        event = Event(
            uid="meeting-123",
            summary="Meeting",
            start=datetime.now(pytz.UTC),
            end=datetime.now(pytz.UTC),
            attendees=[attendee],
            calendar_uid="cal-123",
            account_alias="test",
        )
        assert len(event.attendees) == 1
        assert event.attendees[0].email == "attendee@example.com"
