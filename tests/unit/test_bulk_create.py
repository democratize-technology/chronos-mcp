"""
Unit tests for bulk event creation functionality
"""

import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from chronos_mcp.exceptions import ChronosError

# Import the actual function directly
from chronos_mcp.server import bulk_create_events


class TestBulkCreateEvents:
    """Test the bulk_create_events function"""

    @pytest.fixture
    def mock_managers(self):
        """Setup mock managers"""
        from chronos_mcp.tools.bulk import _managers

        # Save original state
        original_managers = _managers.copy()

        # Create mock managers
        mock_bulk = Mock()
        mock_event = Mock()
        mock_logger = Mock()

        # Set up the global _managers dict
        _managers.clear()
        _managers.update(
            {
                "bulk_manager": mock_bulk,
                "event_manager": mock_event,
                "logger": mock_logger,
            }
        )

        try:
            yield {"event": mock_event, "bulk": mock_bulk, "logger": mock_logger}
        finally:
            # Restore original state
            _managers.clear()
            _managers.update(original_managers)

    @pytest.fixture
    def valid_events(self):
        """Valid event data for testing"""
        return [
            {
                "summary": "Event 1",
                "dtstart": "2025-01-20T10:00:00",
                "dtend": "2025-01-20T11:00:00",
                "description": "Test event 1",
            },
            {
                "summary": "Event 2",
                "dtstart": "2025-01-21T14:00:00",
                "dtend": "2025-01-21T15:00:00",
                "location": "Room B",
            },
            {
                "summary": "Event 3",
                "dtstart": "2025-01-22T09:00:00",
                "dtend": "2025-01-22T10:00:00",
                "all_day": False,
                "alarm_minutes": "15",
            },
        ]

    @pytest.mark.asyncio
    async def test_bulk_create_success(self, mock_managers, valid_events):
        """Test successful bulk creation"""
        # Mock successful bulk creation result
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=3, successful=3, failed=0)
        for i in range(3):
            mock_result.results.append(
                OperationResult(
                    index=i, success=True, uid=f"created-{i}", duration_ms=0.1
                )
            )

        mock_managers["bulk"].bulk_create_events.return_value = mock_result

        result = await bulk_create_events.fn(
            calendar_uid="test-cal",
            events=valid_events,
            mode="continue",
            validate_before_execute=True,
            account=None,
        )

        assert result["success"] is True
        assert result["total"] == 3
        assert result["succeeded"] == 3
        assert result["failed"] == 0
        assert len(result["details"]) == 3

        # Check each detail
        for i, detail in enumerate(result["details"]):
            assert detail["success"] is True
            assert detail["uid"] == f"created-{i}"
            assert detail["summary"] == valid_events[i]["summary"]

    @pytest.mark.asyncio
    async def test_bulk_create_validation_error(self, mock_managers):
        """Test validation errors"""
        invalid_events = [
            {
                "summary": "Valid Event",
                "dtstart": "2025-01-20T10:00:00",
                "dtend": "2025-01-20T11:00:00",
            },
            {
                # Missing summary
                "dtstart": "2025-01-21T14:00:00",
                "dtend": "2025-01-21T15:00:00",
            },
        ]

        # Mock validation failure result
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=2, successful=1, failed=1)
        mock_result.results.append(
            OperationResult(
                index=0, success=True, uid="valid-event-uid", duration_ms=0.1
            )
        )
        mock_result.results.append(
            OperationResult(
                index=1,
                success=False,
                error="Validation failed: Missing required field: summary",
                duration_ms=0.0,
            )
        )

        mock_managers["bulk"].bulk_create_events.return_value = mock_result

        # Direct function call
        result = await bulk_create_events.fn(
            calendar_uid="test-cal",
            events=invalid_events,
            mode="continue",
            validate_before_execute=True,
            account=None,
        )

        assert result["success"] is False
        assert "missing required" in result["details"][1]["error"].lower()

    @pytest.mark.asyncio
    async def test_bulk_create_continue_mode(self, mock_managers, valid_events):
        """Test continue mode with partial failures"""

        # Mock mixed success/failure result
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=3, successful=2, failed=1)
        mock_result.results.append(
            OperationResult(index=0, success=True, uid="uid-Event 1", duration_ms=0.1)
        )
        mock_result.results.append(
            OperationResult(
                index=1, success=False, error="Creation failed", duration_ms=0.1
            )
        )
        mock_result.results.append(
            OperationResult(index=2, success=True, uid="uid-Event 3", duration_ms=0.1)
        )

        mock_managers["bulk"].bulk_create_events.return_value = mock_result

        # Direct function call
        result = await bulk_create_events.fn(
            calendar_uid="test-cal",
            events=valid_events,
            mode="continue",
            validate_before_execute=True,
            account=None,
        )

        assert result["success"] is False
        assert result["total"] == 3
        assert result["succeeded"] == 2
        assert result["failed"] == 1

        # Check failed event
        failed = [d for d in result["details"] if not d["success"]][0]
        assert failed["index"] == 1
        assert "Creation failed" in failed["error"]

    @pytest.mark.asyncio
    async def test_bulk_create_fail_fast_mode(self, mock_managers, valid_events):
        """Test fail_fast mode stops on first error"""

        # Mock fail_fast result - only first event succeeds, then stops
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=3, successful=1, failed=1)
        mock_result.results.append(
            OperationResult(index=0, success=True, uid="uid-Event 1", duration_ms=0.1)
        )
        mock_result.results.append(
            OperationResult(
                index=1, success=False, error="Creation failed", duration_ms=0.1
            )
        )
        # In fail_fast mode, processing stops after first failure

        mock_managers["bulk"].bulk_create_events.return_value = mock_result

        # Direct function call
        result = await bulk_create_events.fn(
            calendar_uid="test-cal",
            events=valid_events,
            mode="fail_fast",
            validate_before_execute=True,
            account=None,
        )

        assert result["success"] is False
        assert result["total"] == 3
        assert result["succeeded"] == 1
        assert result["failed"] == 1
        assert len(result["details"]) == 2  # Stopped after failure

    @pytest.mark.asyncio
    async def test_bulk_create_invalid_mode(self, mock_managers, valid_events):
        """Test invalid mode validation"""
        # Direct function call
        result = await bulk_create_events.fn(
            calendar_uid="test-cal",
            events=valid_events,
            mode="atomic",  # Invalid mode
            validate_before_execute=True,
            account=None,
        )

        assert result["success"] is False
        assert "Invalid mode" in result["error"]

    @pytest.mark.asyncio
    async def test_bulk_create_datetime_parsing(self, mock_managers):
        """Test datetime parsing"""
        events = [
            {
                "summary": "Test Event",
                "dtstart": "2025-01-20T10:00:00Z",
                "dtend": "2025-01-20T11:00:00Z",
            }
        ]

        # Mock successful parsing result
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=1, successful=1, failed=0)
        mock_result.results.append(
            OperationResult(index=0, success=True, uid="test-uid", duration_ms=0.1)
        )

        mock_managers["bulk"].bulk_create_events.return_value = mock_result

        # Direct function call
        result = await bulk_create_events.fn(
            calendar_uid="test-cal",
            events=events,
            mode="continue",
            validate_before_execute=True,
            account=None,
        )

        # Check that parsing was successful (no errors)
        assert result["success"] is True
        assert result["succeeded"] == 1
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_bulk_create_attendees_parsing(self, mock_managers):
        """Test attendees JSON parsing"""
        attendees = [{"email": "test@example.com", "name": "Test User"}]
        events = [
            {
                "summary": "Meeting",
                "dtstart": "2025-01-20T10:00:00",
                "dtend": "2025-01-20T11:00:00",
                "attendees_json": json.dumps(attendees),
            }
        ]

        # Mock successful parsing result
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=1, successful=1, failed=0)
        mock_result.results.append(
            OperationResult(index=0, success=True, uid="test-uid", duration_ms=0.1)
        )

        mock_managers["bulk"].bulk_create_events.return_value = mock_result

        # Direct function call
        result = await bulk_create_events.fn(
            calendar_uid="test-cal",
            events=events,
            mode="continue",
            validate_before_execute=True,
            account=None,
        )

        # Check that parsing was successful (no errors)
        assert result["success"] is True
        assert result["succeeded"] == 1
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_bulk_create_invalid_attendees_json(self, mock_managers):
        """Test invalid attendees JSON handling"""
        events = [
            {
                "summary": "Meeting",
                "dtstart": "2025-01-20T10:00:00",
                "dtend": "2025-01-20T11:00:00",
                "attendees_json": "not-valid-json",
            }
        ]

        # Mock result for invalid JSON parsing (should still succeed)
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=1, successful=1, failed=0)
        mock_result.results.append(
            OperationResult(index=0, success=True, uid="test-uid", duration_ms=0.1)
        )

        mock_managers["bulk"].bulk_create_events.return_value = mock_result

        # Direct function call
        result = await bulk_create_events.fn(
            calendar_uid="test-cal",
            events=events,
            mode="continue",
            validate_before_execute=True,
            account=None,
        )

        # Invalid JSON is ignored, event still created successfully
        assert result["succeeded"] == 1
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_bulk_create_alarm_parsing(self, mock_managers):
        """Test alarm minutes parsing"""
        events = [
            {
                "summary": "Meeting",
                "dtstart": "2025-01-20T10:00:00",
                "dtend": "2025-01-20T11:00:00",
                "alarm_minutes": "30",
            }
        ]

        # Mock successful parsing result
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=1, successful=1, failed=0)
        mock_result.results.append(
            OperationResult(index=0, success=True, uid="test-uid", duration_ms=0.1)
        )

        mock_managers["bulk"].bulk_create_events.return_value = mock_result

        # Direct function call
        result = await bulk_create_events.fn(
            calendar_uid="test-cal",
            events=events,
            mode="continue",
            validate_before_execute=True,
            account=None,
        )

        # Check that parsing was successful (no errors)
        assert result["success"] is True
        assert result["succeeded"] == 1
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_bulk_create_empty_list(self, mock_managers):
        """Test empty event list"""
        # Direct function call
        result = await bulk_create_events.fn(
            calendar_uid="test-cal",
            events=[],
            mode="continue",
            validate_before_execute=True,
            account=None,
        )

        assert result["success"] is True
        assert result["total"] == 0
        assert result["succeeded"] == 0
        assert result["failed"] == 0
