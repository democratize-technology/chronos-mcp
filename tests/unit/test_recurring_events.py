"""Integration tests for recurring event functionality."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from chronos_mcp.models import Event

# Import the actual function directly
from chronos_mcp.server import create_recurring_event


class TestRecurringEventIntegration:
    """Test recurring event MCP tools integration."""

    @pytest.mark.asyncio
    async def test_create_recurring_event_success(self):
        """Test successful creation of recurring event."""
        # Mock the event manager
        mock_event = Event(
            uid="event-123",
            summary="Weekly Team Meeting",
            start=datetime.now(timezone.utc),
            end=datetime.now(timezone.utc) + timedelta(hours=1),
            all_day=False,
            calendar_uid="cal-456",
            recurrence_rule="FREQ=WEEKLY;BYDAY=MO;COUNT=10",
            account_alias="default",
        )

        with patch(
            "chronos_mcp.server.event_manager.create_event", return_value=mock_event
        ):
            # Direct function call
            result = await create_recurring_event.fn(
                calendar_uid="cal-456",
                summary="Weekly Team Meeting",
                start=datetime.now(timezone.utc).isoformat(),
                duration_minutes=60,
                recurrence_rule="FREQ=WEEKLY;BYDAY=MO;COUNT=10",
                description="Weekly sync meeting",
                location=None,
                alarm_minutes=None,
                attendees_json=None,
                account=None,
            )
        assert result["success"] is True
        assert result["event"]["uid"] == "event-123"
        assert result["event"]["summary"] == "Weekly Team Meeting"
        assert result["event"]["recurrence_rule"] == "FREQ=WEEKLY;BYDAY=MO;COUNT=10"

    @pytest.mark.asyncio
    async def test_create_recurring_event_invalid_rrule(self):
        """Test creation fails with invalid RRULE."""
        # Direct function call
        result = await create_recurring_event.fn(
            calendar_uid="cal-456",
            summary="Invalid Event",
            start=datetime.now(timezone.utc).isoformat(),
            duration_minutes=60,
            recurrence_rule="FREQ=DAILY",  # Missing COUNT or UNTIL
            description=None,
            location=None,
            alarm_minutes=None,
            attendees_json=None,
            account=None,
        )

        assert result["success"] is False
        assert "must have COUNT or UNTIL" in result["error"]

    @pytest.mark.asyncio
    async def test_create_recurring_event_count_too_high(self):
        """Test creation fails when COUNT exceeds limit."""
        # Direct function call
        result = await create_recurring_event.fn(
            calendar_uid="cal-456",
            summary="Too Many Events",
            start=datetime.now(timezone.utc).isoformat(),
            duration_minutes=30,
            recurrence_rule="FREQ=DAILY;COUNT=500",  # Exceeds MAX_COUNT
            description=None,
            location=None,
            alarm_minutes=None,
            attendees_json=None,
            account=None,
        )
        assert result["success"] is False
        assert "cannot exceed 365" in result["error"]

    # NOTE: get_recurring_instances function does not exist in server.py
    # These tests are commented out until the function is implemented
    # @pytest.mark.asyncio
    # async def test_get_recurring_instances_success(self):
    #     """Test successful retrieval of recurring instances."""
    #     # Mock event with recurrence
    #     mock_event = Event(
    #         uid="event-123",
    #         summary="Daily Standup",
    #         start=datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0),
    #         end=datetime.now(timezone.utc).replace(hour=9, minute=15, second=0, microsecond=0),
    #         all_day=False,
    #         calendar_uid="cal-456",
    #         recurrence_rule="FREQ=DAILY;COUNT=5"
    #     )
    #
    #     with patch('chronos_mcp.server.event_manager.get_event', return_value=mock_event):
    #         start_date = datetime.now(timezone.utc).isoformat()
    #         end_date = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    #
    #         result = await get_recurring_instances(
    #             calendar_uid="cal-456",
    #             event_uid="event-123",
    #             start_date=start_date,
    #             end_date=end_date
    #         )
    #
    #     assert result["success"] is True
    #     assert "instances" in result
    #     assert len(result["instances"]) <= 5  # Limited by COUNT=5
    #     assert result["recurrence_rule"] == "FREQ=DAILY;COUNT=5"
    #     # Verify instance structure
    #     if result["instances"]:
    #         instance = result["instances"][0]
    #         assert "start" in instance
    #         assert "end" in instance
    #         assert "summary" in instance
    #         assert instance["summary"] == "Daily Standup"
    #         assert instance["original_event_uid"] == "event-123"

    # @pytest.mark.asyncio
    # async def test_get_recurring_instances_non_recurring(self):
    #     """Test error when event is not recurring."""
    #     # Mock non-recurring event
    #     mock_event = Event(
    #         uid="event-123",
    #         summary="Single Event",
    #         start=datetime.now(timezone.utc),
    #         end=datetime.now(timezone.utc) + timedelta(hours=1),
    #         all_day=False,
    #         calendar_uid="cal-456",
    #         recurrence_rule=None  # No recurrence
    #     )
    #
    #     with patch('chronos_mcp.server.event_manager.get_event', return_value=mock_event):
    #         result = await get_recurring_instances(
    #             calendar_uid="cal-456",
    #             event_uid="event-123",
    #             start_date=datetime.now(timezone.utc).isoformat(),
    #             end_date=(datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    #         )
    #
    #     assert result["success"] is False
    #     assert "not a recurring event" in result["error"]
    #     assert result["error_code"] == "VALIDATION_ERROR"
