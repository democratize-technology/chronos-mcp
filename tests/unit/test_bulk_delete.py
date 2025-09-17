"""
Unit tests for bulk event deletion functionality
"""

from unittest.mock import Mock, patch

import pytest

from chronos_mcp.exceptions import EventNotFoundError

# Import the actual function directly
from chronos_mcp.server import bulk_delete_events


class TestBulkDeleteEvents:
    """Test the bulk_delete_events function"""

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
    def event_uids(self):
        """Sample event UIDs for testing"""
        return ["uid-1", "uid-2", "uid-3", "uid-4", "uid-5"]

    @pytest.mark.asyncio
    async def test_bulk_delete_success(self, mock_managers, event_uids):
        """Test successful bulk deletion"""
        # Mock successful bulk deletion result
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=5, successful=5, failed=0)
        for i in range(5):
            mock_result.results.append(
                OperationResult(
                    index=i, success=True, uid=f"uid-{i+1}", duration_ms=0.1
                )
            )

        mock_managers["bulk"].bulk_delete_events.return_value = mock_result

        # Direct function call
        result = await bulk_delete_events.fn(
            calendar_uid="test-cal",
            event_uids=event_uids,
            mode="continue",
            account=None,
        )

        assert result["success"] is True
        assert result["total"] == 5
        assert result["succeeded"] == 5
        assert result["failed"] == 0
        assert len(result["details"]) == 5

        # Check all were successful
        for detail in result["details"]:
            assert detail["success"] is True
            assert "error" not in detail

        # Verify bulk delete was called
        mock_managers["bulk"].bulk_delete_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_bulk_delete_continue_mode(self, mock_managers, event_uids):
        """Test continue mode with partial failures"""
        # Mock mixed success/failure result
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=5, successful=3, failed=2)
        mock_result.results.append(
            OperationResult(index=0, success=True, uid="uid-1", duration_ms=0.1)
        )
        mock_result.results.append(
            OperationResult(
                index=1,
                success=False,
                error="EventNotFoundError: Event not found",
                duration_ms=0.1,
            )
        )
        mock_result.results.append(
            OperationResult(index=2, success=True, uid="uid-3", duration_ms=0.1)
        )
        mock_result.results.append(
            OperationResult(
                index=3,
                success=False,
                error="EventNotFoundError: Event not found",
                duration_ms=0.1,
            )
        )
        mock_result.results.append(
            OperationResult(index=4, success=True, uid="uid-5", duration_ms=0.1)
        )

        mock_managers["bulk"].bulk_delete_events.return_value = mock_result

        # Direct function call
        result = await bulk_delete_events.fn(
            calendar_uid="test-cal",
            event_uids=event_uids,
            mode="continue",
            account=None,
        )

        assert result["success"] is False
        assert result["total"] == 5
        assert result["succeeded"] == 3
        assert result["failed"] == 2

        # Check failed events
        failed_details = [d for d in result["details"] if not d["success"]]
        assert len(failed_details) == 2
        assert "EventNotFoundError" in failed_details[0]["error"]

    @pytest.mark.asyncio
    async def test_bulk_delete_fail_fast_mode(self, mock_managers, event_uids):
        """Test fail_fast mode stops on first error"""
        # Mock fail_fast result - stops after first failure
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=5, successful=2, failed=1)
        mock_result.results.append(
            OperationResult(index=0, success=True, uid="uid-1", duration_ms=0.1)
        )
        mock_result.results.append(
            OperationResult(index=1, success=True, uid="uid-2", duration_ms=0.1)
        )
        mock_result.results.append(
            OperationResult(
                index=2,
                success=False,
                error="EventNotFoundError: Event not found",
                duration_ms=0.1,
            )
        )
        # In fail_fast mode, processing stops after first failure

        mock_managers["bulk"].bulk_delete_events.return_value = mock_result

        # Direct function call
        result = await bulk_delete_events.fn(
            calendar_uid="test-cal",
            event_uids=event_uids,
            mode="fail_fast",
            account=None,
        )

        assert result["success"] is False
        assert result["total"] == 5
        assert result["succeeded"] == 2
        assert result["failed"] == 1
        assert len(result["details"]) == 3  # Stopped after failure

    @pytest.mark.asyncio
    async def test_bulk_delete_invalid_mode(self, mock_managers, event_uids):
        """Test invalid mode validation"""
        # Direct function call
        result = await bulk_delete_events.fn(
            calendar_uid="test-cal",
            event_uids=event_uids,
            mode="invalid",  # Invalid mode
            account=None,
        )

        assert result["success"] is False
        assert "Invalid mode" in result["error"]

    @pytest.mark.asyncio
    async def test_bulk_delete_empty_list(self, mock_managers):
        """Test empty UID list"""
        # Mock empty result
        from chronos_mcp.bulk import BulkResult

        mock_result = BulkResult(total=0, successful=0, failed=0)
        mock_managers["bulk"].bulk_delete_events.return_value = mock_result

        # Direct function call
        result = await bulk_delete_events.fn(
            calendar_uid="test-cal", event_uids=[], mode="continue", account=None
        )

        assert result["success"] is True
        assert result["total"] == 0
        assert result["succeeded"] == 0
        assert result["failed"] == 0
        assert len(result["details"]) == 0

    @pytest.mark.asyncio
    async def test_bulk_delete_with_account(self, mock_managers):
        """Test deletion with account parameter"""
        # Mock successful deletion result
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=1, successful=1, failed=0)
        mock_result.results.append(
            OperationResult(index=0, success=True, uid="uid-1", duration_ms=0.1)
        )
        mock_managers["bulk"].bulk_delete_events.return_value = mock_result

        # Direct function call
        result = await bulk_delete_events.fn(
            calendar_uid="test-cal",
            event_uids=["uid-1"],
            mode="continue",
            account="test-account",
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_bulk_delete_request_id_propagation(self, mock_managers):
        """Test request_id is properly propagated"""
        mock_managers["event"].delete_event.return_value = None

        # Direct function call
        result = await bulk_delete_events.fn(
            calendar_uid="test-cal",
            event_uids=["uid-1", "uid-2"],
            mode="continue",
            account=None,
        )

        assert "request_id" in result

        # Check request_id was passed to all delete calls
        for call in mock_managers["event"].delete_event.call_args_list:
            assert "request_id" in call[1]
            assert call[1]["request_id"] == result["request_id"]

    @pytest.mark.asyncio
    async def test_bulk_delete_generic_error_handling(self, mock_managers):
        """Test handling of non-ChronosError exceptions"""
        # Mock generic error result
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=1, successful=0, failed=1)
        mock_result.results.append(
            OperationResult(
                index=0, success=False, error="Network error", duration_ms=0.1
            )
        )
        mock_managers["bulk"].bulk_delete_events.return_value = mock_result

        # Direct function call
        result = await bulk_delete_events.fn(
            calendar_uid="test-cal", event_uids=["uid-1"], mode="continue", account=None
        )

        assert result["success"] is False
        assert result["failed"] == 1
        assert "Network error" in result["details"][0]["error"]

    @pytest.mark.asyncio
    async def test_bulk_delete_all_fail(self, mock_managers):
        """Test when all deletions fail"""
        # Mock all failing result
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=3, successful=0, failed=3)
        for i in range(3):
            mock_result.results.append(
                OperationResult(
                    index=i,
                    success=False,
                    error="EventNotFoundError: Event not found",
                    duration_ms=0.1,
                )
            )
        mock_managers["bulk"].bulk_delete_events.return_value = mock_result

        # Direct function call
        result = await bulk_delete_events.fn(
            calendar_uid="test-cal",
            event_uids=["uid-1", "uid-2", "uid-3"],
            mode="continue",
            account=None,
        )

        assert result["success"] is False
        assert result["succeeded"] == 0
        assert result["failed"] == 3
        assert all(not d["success"] for d in result["details"])

    @pytest.mark.asyncio
    async def test_bulk_delete_duplicate_uids(self, mock_managers):
        """Test handling of duplicate UIDs"""
        # Mock successful deletion result for duplicates
        from chronos_mcp.bulk import BulkResult, OperationResult

        mock_result = BulkResult(total=5, successful=5, failed=0)
        for i in range(5):
            mock_result.results.append(
                OperationResult(
                    index=i, success=True, uid=f"uid-{i+1}", duration_ms=0.1
                )
            )
        mock_managers["bulk"].bulk_delete_events.return_value = mock_result

        # Include duplicate UIDs
        uids_with_dupes = ["uid-1", "uid-2", "uid-1", "uid-3", "uid-2"]

        # Direct function call
        result = await bulk_delete_events.fn(
            calendar_uid="test-cal",
            event_uids=uids_with_dupes,
            mode="continue",
            account=None,
        )

        # Should process all UIDs including duplicates
        assert result["total"] == 5
        assert result["succeeded"] == 5
