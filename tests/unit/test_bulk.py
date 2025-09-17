"""
Unit tests for bulk operations
"""

from unittest.mock import Mock

from chronos_mcp.bulk import (
    BulkOperationManager,
    BulkOperationMode,
    BulkOptions,
    BulkResult,
    OperationResult,
)


class TestBulkOptions:
    def test_bulk_options_defaults(self):
        """Test BulkOptions default values"""
        opts = BulkOptions()

        assert opts.mode == BulkOperationMode.CONTINUE_ON_ERROR
        assert opts.max_parallel == 5
        assert opts.timeout_per_operation == 30
        assert opts.validate_before_execute is True
        assert opts.dry_run is False
        assert opts.adaptive_scaling is True
        assert opts.backpressure_threshold_ms == 1000.0
        assert opts.min_parallel == 1
        assert opts.max_parallel_limit == 20

    def test_bulk_operation_modes(self):
        """Test different bulk operation modes"""
        assert BulkOperationMode.ATOMIC.value == "atomic"
        assert BulkOperationMode.CONTINUE_ON_ERROR.value == "continue"
        assert BulkOperationMode.FAIL_FAST.value == "fail_fast"


class TestBulkResult:
    def test_bulk_result_properties(self):
        """Test BulkResult calculated properties"""
        result = BulkResult(total=10, successful=7, failed=3, duration_ms=1500.5)

        assert result.success_rate == 70.0

        # Add some results
        result.results = [
            OperationResult(index=0, success=True, uid="uid1"),
            OperationResult(index=1, success=False, error="Failed"),
            OperationResult(index=2, success=True, uid="uid2"),
        ]

        failures = result.get_failures()
        assert len(failures) == 1
        assert failures[0].index == 1

        successes = result.get_successes()
        assert len(successes) == 2
        assert successes[0].uid == "uid1"


class TestBulkOperationManager:
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_event_manager = Mock()
        self.bulk_manager = BulkOperationManager(self.mock_event_manager)

        # Sample events for testing
        self.test_events = [
            {
                "summary": "Event 1",
                "dtstart": "2025-07-10T10:00:00",
                "dtend": "2025-07-10T11:00:00",
            },
            {
                "summary": "Event 2",
                "dtstart": "2025-07-10T14:00:00",
                "dtend": "2025-07-10T15:00:00",
            },
        ]

    def test_validate_events_success(self):
        """Test event validation with valid events"""
        errors = self.bulk_manager._validate_events(self.test_events)
        assert len(errors) == 0

    def test_validate_events_missing_fields(self):
        """Test event validation with missing required fields"""
        invalid_events = [
            {"summary": "No dates"},  # Missing dtstart and dtend
            {
                "dtstart": "2025-07-10T10:00:00",
                "dtend": "2025-07-10T11:00:00",
            },  # Missing summary
            {
                "summary": "Invalid dates",
                "dtstart": "2025-07-10T11:00:00",
                "dtend": "2025-07-10T10:00:00",  # End before start
            },
        ]

        errors = self.bulk_manager._validate_events(invalid_events)
        assert (
            len(errors) == 5
        )  # 3 errors for first (missing dtstart, dtend, invalid date), 1 for second, 1 for third

        # Check specific errors
        error_messages = [err[1] for err in errors]
        assert any("Missing required field: dtstart" in msg for msg in error_messages)
        assert any("Missing required field: summary" in msg for msg in error_messages)
        assert any("End time before start time" in msg for msg in error_messages)

    def test_bulk_create_dry_run(self):
        """Test bulk create in dry run mode"""
        options = BulkOptions(dry_run=True)

        result = self.bulk_manager.bulk_create_events(
            calendar_uid="cal123", events=self.test_events, options=options
        )

        assert result.total == 2
        assert result.successful == 2
        assert result.failed == 0
        assert len(result.results) == 2

        # Event manager should not be called in dry run
        self.mock_event_manager.create_event.assert_not_called()

    def test_bulk_create_continue_on_error(self):
        """Test bulk create with continue on error mode"""
        # Mock event manager to fail on second event
        mock_event1 = Mock()
        mock_event1.uid = "created-1"
        self.mock_event_manager.create_event.side_effect = [
            mock_event1,  # Success
            Exception("Network error"),  # Failure
        ]

        options = BulkOptions(mode=BulkOperationMode.CONTINUE_ON_ERROR)

        result = self.bulk_manager.bulk_create_events(
            calendar_uid="cal123", events=self.test_events, options=options
        )

        assert result.total == 2
        assert result.successful == 1
        assert result.failed == 1
        assert self.mock_event_manager.create_event.call_count == 2

    def test_bulk_create_fail_fast(self):
        """Test bulk create with fail fast mode"""
        # Use smaller batch to test fail fast properly
        test_events = [
            {
                "summary": f"Event {i}",
                "dtstart": "2025-07-10T16:00:00",
                "dtend": "2025-07-10T17:00:00",
            }
            for i in range(1, 4)
        ]

        # Mock to fail on second event
        mock_event1 = Mock()
        mock_event1.uid = "created-1"
        mock_event3 = Mock()
        mock_event3.uid = "created-3"
        self.mock_event_manager.create_event.side_effect = [
            mock_event1,
            Exception("API limit reached"),
            mock_event3,
        ]

        options = BulkOptions(mode=BulkOperationMode.FAIL_FAST, max_parallel=2)

        result = self.bulk_manager.bulk_create_events(
            calendar_uid="cal123", events=test_events, options=options
        )

        # In fail_fast mode with batch processing
        assert result.failed >= 1  # At least one failure
        assert result.total == 3
        # Due to parallel batch processing, it may process 1-2 before stopping
        assert result.successful <= 2

    def test_bulk_create_parallel_execution(self):
        """Test that bulk operations execute in batches"""
        # Mock successful event creation
        mock_event1 = Mock()
        mock_event1.uid = "uid1"
        mock_event2 = Mock()
        mock_event2.uid = "uid2"

        self.mock_event_manager.create_event.side_effect = [mock_event1, mock_event2]

        options = BulkOptions(max_parallel=2)

        # Call the batch execution method directly
        results = self.bulk_manager._execute_batch_create(
            calendar_uid="cal123",
            batch=self.test_events,
            start_idx=0,
            options=options,
            account_alias=None,
        )

        # Should have created 2 events
        assert self.mock_event_manager.create_event.call_count == 2
        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].uid == "uid1"
        assert results[1].uid == "uid2"


class TestBulkDelete:
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_event_manager = Mock()
        self.bulk_manager = BulkOperationManager(self.mock_event_manager)
        self.test_uids = ["uid1", "uid2", "uid3"]

    def test_bulk_delete_success(self):
        """Test successful bulk delete"""
        # Mock successful deletions
        self.mock_event_manager.delete_event.return_value = True

        options = BulkOptions()
        result = self.bulk_manager.bulk_delete_events(
            calendar_uid="cal123", event_uids=self.test_uids, options=options
        )

        assert result.total == 3
        assert result.successful == 3
        assert result.failed == 0
        assert self.mock_event_manager.delete_event.call_count == 3

    def test_bulk_delete_with_failures(self):
        """Test bulk delete with some failures"""
        # Mock mixed results
        self.mock_event_manager.delete_event.side_effect = [
            True,  # Success
            Exception("Event not found"),  # Failure
            True,  # Success
        ]

        options = BulkOptions(mode=BulkOperationMode.CONTINUE_ON_ERROR)
        result = self.bulk_manager.bulk_delete_events(
            calendar_uid="cal123", event_uids=self.test_uids, options=options
        )

        assert result.total == 3
        assert result.successful == 2
        assert result.failed == 1

        # Check that the failed operation has error info
        failures = result.get_failures()
        assert len(failures) == 1
        assert "Event not found" in failures[0].error

    def test_adaptive_scaling_performance_tracking(self):
        """Test that performance metrics are tracked for adaptive scaling"""
        bulk_manager = BulkOperationManager()

        # Track some performance data
        bulk_manager._track_operation_performance("create_event", 500.0)
        bulk_manager._track_operation_performance("create_event", 1500.0)
        bulk_manager._track_operation_performance("create_event", 750.0)

        recent_perf = bulk_manager._get_recent_performance("create_event")
        assert len(recent_perf) == 3
        assert 500.0 in recent_perf
        assert 1500.0 in recent_perf
        assert 750.0 in recent_perf

    def test_adaptive_scaling_parallelism_calculation(self):
        """Test adaptive parallelism calculation based on performance"""
        bulk_manager = BulkOperationManager()
        options = BulkOptions(max_parallel=10, backpressure_threshold_ms=1000.0)

        # Test fast operations - should increase parallelism
        fast_performance = [200.0, 300.0, 250.0]  # All under threshold/2
        new_parallel = bulk_manager._calculate_adaptive_parallelism(
            options, "create_event", fast_performance
        )
        assert new_parallel > options.max_parallel  # Should increase

        # Test slow operations - should decrease parallelism
        slow_performance = [1500.0, 2000.0, 1800.0]  # All over threshold
        new_parallel = bulk_manager._calculate_adaptive_parallelism(
            options, "create_event", slow_performance
        )
        assert new_parallel == options.max_parallel // 2  # Should decrease

        # Test mixed performance - should stay same
        mixed_performance = [800.0, 900.0, 700.0]  # Within acceptable range
        new_parallel = bulk_manager._calculate_adaptive_parallelism(
            options, "create_event", mixed_performance
        )
        assert new_parallel == options.max_parallel  # Should stay same

    def test_adaptive_scaling_disabled(self):
        """Test that adaptive scaling can be disabled"""
        bulk_manager = BulkOperationManager()
        options = BulkOptions(adaptive_scaling=False, max_parallel=5)

        # Even with slow performance, should return original max_parallel
        slow_performance = [2000.0, 3000.0, 2500.0]
        new_parallel = bulk_manager._calculate_adaptive_parallelism(
            options, "create_event", slow_performance
        )
        assert new_parallel == options.max_parallel

    def test_performance_tracker_sliding_window(self):
        """Test that performance tracker maintains sliding window"""
        bulk_manager = BulkOperationManager()

        # Add more than 50 measurements
        for i in range(60):
            bulk_manager._track_operation_performance("create_event", float(i * 10))

        recent_perf = bulk_manager._get_recent_performance("create_event")
        # Should keep only last 50 measurements
        assert len(recent_perf) == 50
        # Should contain the most recent values (590, 580, ... 100)
        assert 590.0 in recent_perf
        assert 100.0 in recent_perf
        assert 90.0 not in recent_perf  # Should have been removed

    def test_bulk_create_with_adaptive_scaling(self):
        """Test bulk create operations adapt parallelism based on performance"""
        # Create a larger set of events to test adaptive scaling
        test_events = [
            {
                "summary": f"Event {i}",
                "dtstart": "2025-07-10T10:00:00",
                "dtend": "2025-07-10T11:00:00",
            }
            for i in range(15)  # More events to trigger multiple batches
        ]

        mock_event_manager = Mock()
        bulk_manager = BulkOperationManager(mock_event_manager)

        # Mock successful event creation with varying response times
        def create_event_mock(*args, **kwargs):
            # Extract the event number from the summary
            summary = kwargs.get("summary", args[1] if len(args) > 1 else "")
            if summary and "Event " in summary:
                event_num = summary.split("Event ")[1]
                mock_event = Mock()
                mock_event.uid = f"uid{event_num}"
                return mock_event
            else:
                # Fallback for any unexpected calls
                mock_event = Mock()
                mock_event.uid = (
                    f"uid_unknown_{len(mock_event_manager.create_event.call_args_list)}"
                )
                return mock_event

        mock_event_manager.create_event.side_effect = create_event_mock

        options = BulkOptions(
            adaptive_scaling=True,
            max_parallel=5,
            backpressure_threshold_ms=1000.0,
        )

        # Simulate some performance data that would trigger scaling
        for _ in range(10):
            bulk_manager._track_operation_performance("create_event", 1500.0)  # Slow

        result = bulk_manager.bulk_create_events(
            calendar_uid="cal123", events=test_events, options=options
        )

        assert result.total == 15
        assert result.successful == 15
        assert result.failed == 0

        # Verify all events were created
        assert mock_event_manager.create_event.call_count == 15
