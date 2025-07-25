"""
Unit tests for advanced search functionality
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import the actual function directly
from chronos_mcp.server import search_events
from chronos_mcp.models import Event


class TestSearchEvents:
    """Test the search_events function"""

    @pytest.fixture
    def mock_managers(self):
        """Setup mock managers"""
        with (
            patch("chronos_mcp.server.calendar_manager") as mock_cal,
            patch("chronos_mcp.server.event_manager") as mock_event,
            patch("chronos_mcp.server.logger") as mock_logger,
        ):
            yield {"calendar": mock_cal, "event": mock_event, "logger": mock_logger}

    @pytest.fixture
    def sample_events(self):
        """Create sample events for testing"""
        base_date = datetime.now()
        return [
            Event(
                uid="evt-1",
                summary="Team Meeting - Project Review",
                description="Quarterly review meeting with the team",
                start=base_date + timedelta(days=1),
                end=base_date + timedelta(days=1, hours=1),
                location="Conference Room A",
                all_day=False,
                calendar_uid="test-calendar",
                account_alias="default",
            ),
            Event(
                uid="evt-2",
                summary="Client Call - ABC Corp",
                description="Discuss contract renewal",
                start=base_date + timedelta(days=2),
                end=base_date + timedelta(days=2, hours=1),
                location="Zoom Meeting",
                all_day=False,
                calendar_uid="test-calendar",
                account_alias="default",
            ),
            Event(
                uid="evt-3",
                summary="Workshop: Python Best Practices",
                description="Internal training workshop",
                start=base_date + timedelta(days=3),
                end=base_date + timedelta(days=3, hours=3),
                location="Training Room",
                all_day=False,
                calendar_uid="test-calendar",
                account_alias="default",
            ),
        ]

    @pytest.mark.asyncio
    async def test_search_events_basic(self, mock_managers, sample_events):
        """Test basic search functionality"""
        # Setup mocks
        mock_cal = Mock()
        mock_cal.uid = "test-calendar"
        mock_managers["calendar"].list_calendars.return_value = [mock_cal]
        mock_managers["event"].get_events_range.return_value = sample_events

        # Execute search
        result = await search_events.fn(
            query="meeting",
            fields=["summary", "description", "location"],
            case_sensitive=False,
            date_start=None,
            date_end=None,
            calendar_uid=None,
            max_results=50,
            account=None,
        )

        # Debug print
        if not result["success"]:
            print(f"Search failed: {result}")

        # Verify results
        assert result["success"] is True
        assert result["query"] == "meeting"
        assert len(result["matches"]) == 2  # Should find 2 events with "meeting"
        assert result["total"] == 2
        assert result["truncated"] is False

        # Check matched events
        matches = result["matches"]
        assert matches[0]["uid"] == "evt-1"
        assert matches[0]["matched_field"] == "summary"
        assert matches[1]["uid"] == "evt-2"
        assert matches[1]["matched_field"] == "location"

    @pytest.mark.asyncio
    async def test_search_events_case_sensitive(self, mock_managers, sample_events):
        """Test case-sensitive search"""
        mock_cal = Mock()
        mock_cal.uid = "test-calendar"
        mock_managers["calendar"].list_calendars.return_value = [mock_cal]
        mock_managers["event"].get_events_range.return_value = sample_events

        # Case-sensitive search
        # Direct function call
        result = await search_events.fn(
            query="Meeting",
            fields=["summary", "description", "location"],
            case_sensitive=True,
            date_start=None,
            date_end=None,
            calendar_uid=None,
            max_results=50,
            account=None,
        )

        assert result["success"] is True
        assert len(result["matches"]) == 1  # Only "Zoom Meeting" matches
        assert result["matches"][0]["uid"] == "evt-2"

    @pytest.mark.asyncio
    async def test_search_events_specific_calendar(self, mock_managers, sample_events):
        """Test searching specific calendar"""
        mock_managers["event"].get_events_range.return_value = sample_events

        # Direct function call
        result = await search_events.fn(
            query="workshop",
            fields=["summary", "description", "location"],
            case_sensitive=False,
            date_start=None,
            date_end=None,
            calendar_uid="specific-cal",
            max_results=50,
            account=None,
        )

        # Should not call list_calendars when calendar_uid is specified
        mock_managers["calendar"].list_calendars.assert_not_called()

        # Should search only the specified calendar
        mock_managers["event"].get_events_range.assert_called_once()
        call_args = mock_managers["event"].get_events_range.call_args[1]
        assert call_args["calendar_uid"] == "specific-cal"

        assert result["success"] is True
        assert len(result["matches"]) == 1
        assert result["matches"][0]["uid"] == "evt-3"

    @pytest.mark.asyncio
    async def test_search_events_validation_errors(self, mock_managers):
        """Test input validation"""
        # Query too short
        # Direct function call
        result = await search_events.fn(
            query="a",
            fields=["summary", "description", "location"],
            case_sensitive=False,
            date_start=None,
            date_end=None,
            calendar_uid=None,
            max_results=50,
            account=None,
        )
        assert result["success"] is False
        assert "too short" in result["error"]

        # Query too long
        # Direct function call
        result = await search_events.fn(
            query="x" * 1001,
            fields=["summary", "description", "location"],
            case_sensitive=False,
            date_start=None,
            date_end=None,
            calendar_uid=None,
            max_results=50,
            account=None,
        )
        assert result["success"] is False
        assert "too long" in result["error"]

    @pytest.mark.asyncio
    async def test_search_events_field_validation(self, mock_managers):
        """Test field validation"""
        # Direct function call
        result = await search_events.fn(
            query="test",
            fields=["summary", "__proto__", "description"],
            case_sensitive=False,
            date_start=None,
            date_end=None,
            calendar_uid=None,
            max_results=50,
            account=None,
        )
        assert result["success"] is False
        assert "Invalid field" in result["error"]

    @pytest.mark.asyncio
    async def test_search_events_date_range(self, mock_managers, sample_events):
        """Test date range filtering"""
        mock_cal = Mock()
        mock_cal.uid = "test-calendar"
        mock_managers["calendar"].list_calendars.return_value = [mock_cal]
        mock_managers["event"].get_events_range.return_value = sample_events

        start_date = datetime.now().isoformat()
        end_date = (datetime.now() + timedelta(days=30)).isoformat()

        # Direct function call
        result = await search_events.fn(
            query="meeting",
            fields=["summary", "description", "location"],
            case_sensitive=False,
            date_start=start_date,
            date_end=end_date,
            calendar_uid=None,
            max_results=50,
            account=None,
        )

        # Verify date parsing was called
        call_args = mock_managers["event"].get_events_range.call_args[1]
        assert isinstance(call_args["start_date"], datetime)
        assert isinstance(call_args["end_date"], datetime)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_events_max_results(self, mock_managers):
        """Test max_results limiting"""
        # Create many events
        many_events = []
        for i in range(20):
            event = Event(
                uid=f"evt-{i}",
                summary=f"Meeting {i}",
                description="Test meeting",
                start=datetime.now() + timedelta(days=i),
                end=datetime.now() + timedelta(days=i, hours=1),
                all_day=False,
                calendar_uid="test-calendar",
                account_alias="default",
            )
            many_events.append(event)

        mock_cal = Mock()
        mock_cal.uid = "test-calendar"
        mock_managers["calendar"].list_calendars.return_value = [mock_cal]
        mock_managers["event"].get_events_range.return_value = many_events

        # Direct function call
        result = await search_events.fn(
            query="meeting",
            fields=["summary", "description", "location"],
            case_sensitive=False,
            date_start=None,
            date_end=None,
            calendar_uid=None,
            max_results=5,
            account=None,
        )

        assert result["success"] is True
        assert len(result["matches"]) == 5
        assert result["total"] == 5
        assert result["truncated"] is False  # We stop searching at max_results

    @pytest.mark.asyncio
    async def test_search_events_error_handling(self, mock_managers):
        """Test error handling in search"""
        mock_cal = Mock()
        mock_cal.uid = "test-calendar"
        mock_managers["calendar"].list_calendars.return_value = [mock_cal]
        mock_managers["event"].get_events_range.side_effect = Exception(
            "Calendar error"
        )

        # Direct function call
        result = await search_events.fn(
            query="meeting",
            fields=["summary", "description", "location"],
            case_sensitive=False,
            date_start=None,
            date_end=None,
            calendar_uid=None,
            max_results=50,
            account=None,
        )

        # Should continue searching other calendars
        assert result["success"] is True
        assert result["total"] == 0
        assert len(result["matches"]) == 0

        # Should log the error
        mock_managers["logger"].warning.assert_called()
