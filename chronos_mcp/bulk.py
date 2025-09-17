"""Bulk operations for Chronos MCP."""

import concurrent.futures
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .logging_config import setup_logging
from .models import TaskStatus
from .utils import parse_datetime

logger = setup_logging()


class BulkOperationMode(Enum):
    """Modes for bulk operation execution."""

    ATOMIC = "atomic"
    CONTINUE_ON_ERROR = "continue"
    FAIL_FAST = "fail_fast"


@dataclass
class BulkOptions:
    """Configuration for bulk operations."""

    mode: BulkOperationMode = BulkOperationMode.CONTINUE_ON_ERROR
    max_parallel: int = 5
    timeout_per_operation: int = 30
    validate_before_execute: bool = True
    dry_run: bool = False
    adaptive_scaling: bool = True  # Enable adaptive parallel scaling
    backpressure_threshold_ms: float = 1000.0  # Reduce parallelism if ops take longer
    min_parallel: int = 1
    max_parallel_limit: int = 20


@dataclass
class OperationResult:
    """Result of a single operation within bulk."""

    index: int
    success: bool
    uid: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class BulkResult:
    """Aggregate result of bulk operation."""

    total: int
    successful: int
    failed: int
    results: List[OperationResult] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        return (self.successful / self.total * 100) if self.total > 0 else 0.0

    def get_failures(self) -> List[OperationResult]:
        return [r for r in self.results if not r.success]

    def get_successes(self) -> List[OperationResult]:
        return [r for r in self.results if r.success]


class BulkOperationManager:
    """Manages bulk CalDAV operations with adaptive concurrency control."""

    def __init__(self, event_manager=None, task_manager=None, journal_manager=None):
        self.event_manager = event_manager
        self.task_manager = task_manager
        self.journal_manager = journal_manager
        self.executor = ThreadPoolExecutor(max_workers=20)

        # Adaptive concurrency control
        self._performance_tracker = {}
        self._backpressure_lock = threading.Lock()

    def _calculate_adaptive_parallelism(
        self, options: BulkOptions, operation_type: str, recent_performance: List[float]
    ) -> int:
        """Calculate optimal parallelism based on recent performance"""
        if not options.adaptive_scaling or not recent_performance:
            return options.max_parallel

        # Calculate average operation time
        avg_time_ms = sum(recent_performance) / len(recent_performance)

        if avg_time_ms > options.backpressure_threshold_ms:
            # Operations are slow, reduce parallelism
            new_parallel = max(options.min_parallel, options.max_parallel // 2)
        elif avg_time_ms < options.backpressure_threshold_ms / 2:
            # Operations are fast, increase parallelism
            new_parallel = min(options.max_parallel_limit, options.max_parallel + 2)
        else:
            # Operations are within acceptable range
            new_parallel = options.max_parallel

        return new_parallel

    def _track_operation_performance(self, operation_type: str, duration_ms: float):
        """Track operation performance for adaptive scaling"""
        with self._backpressure_lock:
            if operation_type not in self._performance_tracker:
                self._performance_tracker[operation_type] = []

            perf_list = self._performance_tracker[operation_type]
            perf_list.append(duration_ms)

            # Keep only last 50 measurements for sliding window
            if len(perf_list) > 50:
                perf_list.pop(0)

    def _get_recent_performance(self, operation_type: str) -> List[float]:
        """Get recent performance measurements"""
        with self._backpressure_lock:
            return self._performance_tracker.get(operation_type, []).copy()

    def bulk_create_events(
        self,
        calendar_uid: str,
        events: List[Dict[str, Any]],
        options: BulkOptions = None,
        account_alias: Optional[str] = None,
    ) -> BulkResult:
        """Create multiple events with configurable error handling."""
        if options is None:
            options = BulkOptions()

        start_time = time.time()
        result = BulkResult(total=len(events), successful=0, failed=0)

        if options.validate_before_execute:
            validation_errors = self._validate_events(events)
            if validation_errors and options.mode == BulkOperationMode.ATOMIC:
                for idx, error in validation_errors:
                    result.results.append(
                        OperationResult(
                            index=idx,
                            success=False,
                            error=f"Validation failed: {error}",
                        )
                    )
                result.failed = len(validation_errors)
                result.duration_ms = (time.time() - start_time) * 1000
                return result

        if options.dry_run:
            for idx in range(len(events)):
                result.results.append(
                    OperationResult(
                        index=idx,
                        success=True,
                        uid=f"dry-run-uid-{idx}",
                        duration_ms=0.1,
                    )
                )
            result.successful = len(events)
        else:
            created_uids = []
            current_parallel = options.max_parallel
            batch_start = 0

            while batch_start < len(events):
                # Adaptive scaling - adjust parallelism based on performance
                if options.adaptive_scaling and batch_start > 0:
                    recent_perf = self._get_recent_performance("create_event")
                    current_parallel = self._calculate_adaptive_parallelism(
                        options, "create_event", recent_perf
                    )

                batch_end = min(batch_start + current_parallel, len(events))
                batch = events[batch_start:batch_end]

                batch_results = self._execute_batch_create(
                    calendar_uid, batch, batch_start, options, account_alias
                )

                # Track batch performance
                for op_result in batch_results:
                    self._track_operation_performance(
                        "create_event", op_result.duration_ms
                    )

                for op_result in batch_results:
                    result.results.append(op_result)
                    if op_result.success:
                        result.successful += 1
                        created_uids.append(op_result.uid)
                    else:
                        result.failed += 1

                        if options.mode == BulkOperationMode.FAIL_FAST:
                            break
                        elif options.mode == BulkOperationMode.ATOMIC:
                            self._rollback_created_events(calendar_uid, created_uids)
                            result.successful = 0
                            result.failed = len(events)
                            break

                if (
                    options.mode == BulkOperationMode.FAIL_FAST and result.failed > 0
                ) or (options.mode == BulkOperationMode.ATOMIC and result.failed > 0):
                    break

                # Move to next batch
                batch_start = batch_end

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def bulk_create_tasks(
        self,
        calendar_uid: str,
        tasks: List[Dict[str, Any]],
        options: BulkOptions = None,
        account_alias: Optional[str] = None,
    ) -> BulkResult:
        """Create multiple tasks with configurable error handling."""
        if options is None:
            options = BulkOptions()

        start_time = time.time()
        result = BulkResult(total=len(tasks), successful=0, failed=0)

        if options.validate_before_execute:
            validation_errors = self._validate_tasks(tasks)
            if validation_errors and options.mode == BulkOperationMode.ATOMIC:
                for idx, error in validation_errors:
                    result.results.append(
                        OperationResult(
                            index=idx,
                            success=False,
                            error=f"Validation failed: {error}",
                        )
                    )
                result.failed = len(validation_errors)
                result.duration_ms = (time.time() - start_time) * 1000
                return result

        if options.dry_run:
            for idx in range(len(tasks)):
                result.results.append(
                    OperationResult(
                        index=idx,
                        success=True,
                        uid=f"dry-run-task-uid-{idx}",
                        duration_ms=0.1,
                    )
                )
            result.successful = len(tasks)
        else:
            created_uids = []

            for batch_start in range(0, len(tasks), options.max_parallel):
                batch_end = min(batch_start + options.max_parallel, len(tasks))
                batch = tasks[batch_start:batch_end]

                batch_results = self._execute_batch_create_tasks(
                    calendar_uid, batch, batch_start, options, account_alias
                )

                for op_result in batch_results:
                    result.results.append(op_result)
                    if op_result.success:
                        result.successful += 1
                        created_uids.append(op_result.uid)
                    else:
                        result.failed += 1

                        if options.mode == BulkOperationMode.FAIL_FAST:
                            break
                        elif options.mode == BulkOperationMode.ATOMIC:
                            self._rollback_created_tasks(calendar_uid, created_uids)
                            result.successful = 0
                            result.failed = len(tasks)
                            break

                if (
                    options.mode == BulkOperationMode.FAIL_FAST and result.failed > 0
                ) or (options.mode == BulkOperationMode.ATOMIC and result.failed > 0):
                    break

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def bulk_create_journals(
        self,
        calendar_uid: str,
        journals: List[Dict[str, Any]],
        options: BulkOptions = None,
        account_alias: Optional[str] = None,
    ) -> BulkResult:
        """Create multiple journals with configurable error handling."""
        if options is None:
            options = BulkOptions()

        start_time = time.time()
        result = BulkResult(total=len(journals), successful=0, failed=0)

        if options.validate_before_execute:
            validation_errors = self._validate_journals(journals)
            if validation_errors and options.mode == BulkOperationMode.ATOMIC:
                for idx, error in validation_errors:
                    result.results.append(
                        OperationResult(
                            index=idx,
                            success=False,
                            error=f"Validation failed: {error}",
                        )
                    )
                result.failed = len(validation_errors)
                result.duration_ms = (time.time() - start_time) * 1000
                return result

        if options.dry_run:
            for idx in range(len(journals)):
                result.results.append(
                    OperationResult(
                        index=idx,
                        success=True,
                        uid=f"dry-run-journal-uid-{idx}",
                        duration_ms=0.1,
                    )
                )
            result.successful = len(journals)
        else:
            created_uids = []

            for batch_start in range(0, len(journals), options.max_parallel):
                batch_end = min(batch_start + options.max_parallel, len(journals))
                batch = journals[batch_start:batch_end]

                batch_results = self._execute_batch_create_journals(
                    calendar_uid, batch, batch_start, options, account_alias
                )

                for op_result in batch_results:
                    result.results.append(op_result)
                    if op_result.success:
                        result.successful += 1
                        created_uids.append(op_result.uid)
                    else:
                        result.failed += 1

                        if options.mode == BulkOperationMode.FAIL_FAST:
                            break
                        elif options.mode == BulkOperationMode.ATOMIC:
                            self._rollback_created_journals(calendar_uid, created_uids)
                            result.successful = 0
                            result.failed = len(journals)
                            break

                if (
                    options.mode == BulkOperationMode.FAIL_FAST and result.failed > 0
                ) or (options.mode == BulkOperationMode.ATOMIC and result.failed > 0):
                    break

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def _validate_events(self, events: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
        """Validate event data before execution."""
        errors = []

        for idx, event in enumerate(events):
            if not event.get("summary"):
                errors.append((idx, "Missing required field: summary"))
            if not event.get("dtstart"):
                errors.append((idx, "Missing required field: dtstart"))
            if not event.get("dtend"):
                errors.append((idx, "Missing required field: dtend"))

            try:
                start = datetime.fromisoformat(
                    str(event.get("dtstart", "")).replace("Z", "+00:00")
                )
                end = datetime.fromisoformat(
                    str(event.get("dtend", "")).replace("Z", "+00:00")
                )
                if end < start:
                    errors.append((idx, "End time before start time"))
            except Exception:
                errors.append((idx, "Invalid date format"))

        return errors

    def _validate_tasks(self, tasks: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
        """Validate task data before execution."""
        errors = []

        for idx, task in enumerate(tasks):
            if not task.get("summary"):
                errors.append((idx, "Missing required field: summary"))

            # Validate priority if provided
            priority = task.get("priority")
            if priority is not None:
                try:
                    priority_val = int(priority)
                    if priority_val < 1 or priority_val > 9:
                        errors.append((idx, "Priority must be between 1-9"))
                except (ValueError, TypeError):
                    errors.append((idx, "Priority must be an integer"))

            # Validate status if provided
            status = task.get("status")
            if status is not None:
                try:
                    TaskStatus(status)
                except ValueError:
                    valid_statuses = [s.value for s in TaskStatus]
                    errors.append(
                        (idx, f"Invalid status. Must be one of: {valid_statuses}")
                    )

            # Validate percent_complete if provided
            percent = task.get("percent_complete")
            if percent is not None:
                try:
                    percent_val = int(percent)
                    if percent_val < 0 or percent_val > 100:
                        errors.append((idx, "Percent complete must be between 0-100"))
                except (ValueError, TypeError):
                    errors.append((idx, "Percent complete must be an integer"))

            # Validate due date if provided
            due = task.get("due")
            if due is not None:
                try:
                    if isinstance(due, str):
                        datetime.fromisoformat(due.replace("Z", "+00:00"))
                except Exception:
                    errors.append((idx, "Invalid due date format"))

        return errors

    def _validate_journals(
        self, journals: List[Dict[str, Any]]
    ) -> List[Tuple[int, str]]:
        """Validate journal data before execution."""
        errors = []

        for idx, journal in enumerate(journals):
            if not journal.get("summary"):
                errors.append((idx, "Missing required field: summary"))

            # Validate dtstart if provided
            dtstart = journal.get("dtstart")
            if dtstart is not None:
                try:
                    if isinstance(dtstart, str):
                        datetime.fromisoformat(dtstart.replace("Z", "+00:00"))
                except Exception:
                    errors.append((idx, "Invalid dtstart date format"))

        return errors

    def _execute_batch_create(
        self,
        calendar_uid: str,
        batch: List[Dict],
        start_idx: int,
        options: BulkOptions,
        account_alias: Optional[str] = None,
    ) -> List[OperationResult]:
        """Execute a batch of create operations in parallel using ThreadPoolExecutor."""
        if not self.event_manager:
            raise ValueError("EventManager not provided to BulkOperationManager")

        def create_single_event(idx_event_tuple):
            idx, event = idx_event_tuple
            op_start = time.time()
            try:
                created_event = self.event_manager.create_event(
                    calendar_uid=calendar_uid,
                    summary=event.get("summary"),
                    start=event.get("dtstart"),
                    end=event.get("dtend"),
                    description=event.get("description"),
                    location=event.get("location"),
                    all_day=event.get("all_day", False),
                    alarm_minutes=event.get("alarm_minutes"),
                    recurrence_rule=event.get("recurrence_rule"),
                    attendees=event.get("attendees", []),
                    related_to=event.get("related_to", []),
                    account_alias=account_alias,
                )

                return OperationResult(
                    index=start_idx + idx,
                    success=True,
                    uid=created_event.uid,
                    duration_ms=(time.time() - op_start) * 1000,
                )
            except Exception as e:
                return OperationResult(
                    index=start_idx + idx,
                    success=False,
                    error=str(e),
                    duration_ms=(time.time() - op_start) * 1000,
                )

        # Use ThreadPoolExecutor with timeout control
        indexed_batch = list(enumerate(batch))
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_idx = {
                executor.submit(create_single_event, idx_event): idx
                for idx, idx_event in enumerate(indexed_batch)
            }

            results: List[Optional[OperationResult]] = [None] * len(batch)

            try:
                for future in concurrent.futures.as_completed(
                    future_to_idx, timeout=options.timeout_per_operation * len(batch)
                ):
                    try:
                        result = future.result(timeout=options.timeout_per_operation)
                        # Maintain original order based on batch index
                        batch_idx = result.index - start_idx
                        results[batch_idx] = result
                    except concurrent.futures.TimeoutError:
                        # Handle individual operation timeout
                        batch_idx = future_to_idx[future]
                        results[batch_idx] = OperationResult(
                            index=start_idx + batch_idx,
                            success=False,
                            error=f"Operation timeout after {options.timeout_per_operation}s",
                            duration_ms=options.timeout_per_operation * 1000,
                        )
                    except Exception as e:
                        # Handle executor-level exceptions
                        batch_idx = future_to_idx[future]
                        results[batch_idx] = OperationResult(
                            index=start_idx + batch_idx,
                            success=False,
                            error=f"Executor error: {e}",
                            duration_ms=0,
                        )
            except concurrent.futures.TimeoutError:
                # Handle batch-level timeout
                for idx, future in enumerate(future_to_idx.keys()):
                    if not future.done():
                        future.cancel()
                        if results[idx] is None:
                            results[idx] = OperationResult(
                                index=start_idx + idx,
                                success=False,
                                error="Batch operation timeout",
                                duration_ms=0,
                            )

        return [r for r in results if r is not None]

    def _execute_batch_create_tasks(
        self,
        calendar_uid: str,
        batch: List[Dict],
        start_idx: int,
        options: BulkOptions,
        account_alias: Optional[str] = None,
    ) -> List[OperationResult]:
        """Execute a batch of task create operations."""
        results = []

        # Use the provided task manager
        if not self.task_manager:
            raise ValueError("TaskManager not provided to BulkOperationManager")

        for idx, task in enumerate(batch):
            op_start = time.time()
            try:
                # Parse status if provided
                status = None
                if task.get("status"):
                    status = TaskStatus(task.get("status"))

                # Parse due date if provided as string
                due_dt = None
                if task.get("due"):
                    due_value = task.get("due")
                    if isinstance(due_value, str):
                        due_dt = parse_datetime(due_value)
                    else:
                        due_dt = due_value

                created_task = self.task_manager.create_task(
                    calendar_uid=calendar_uid,
                    summary=task.get("summary"),
                    description=task.get("description"),
                    due=due_dt,
                    priority=task.get("priority"),
                    status=status or TaskStatus.NEEDS_ACTION,
                    related_to=task.get("related_to", []),
                    account_alias=account_alias,
                )

                results.append(
                    OperationResult(
                        index=start_idx + idx,
                        success=True,
                        uid=created_task.uid,
                        duration_ms=(time.time() - op_start) * 1000,
                    )
                )
            except Exception as e:
                results.append(
                    OperationResult(
                        index=start_idx + idx,
                        success=False,
                        error=str(e),
                        duration_ms=(time.time() - op_start) * 1000,
                    )
                )

        return results

    def _execute_batch_create_journals(
        self,
        calendar_uid: str,
        batch: List[Dict],
        start_idx: int,
        options: BulkOptions,
        account_alias: Optional[str] = None,
    ) -> List[OperationResult]:
        """Execute a batch of journal create operations."""
        results = []

        # Use the provided journal manager
        if not self.journal_manager:
            raise ValueError("JournalManager not provided to BulkOperationManager")

        for idx, journal in enumerate(batch):
            op_start = time.time()
            try:
                # Parse dtstart if provided as string
                dtstart = journal.get("dtstart")
                if dtstart and isinstance(dtstart, str):
                    dtstart = parse_datetime(dtstart)

                # Create the journal using JournalManager
                created_journal = self.journal_manager.create_journal(
                    calendar_uid=calendar_uid,
                    summary=journal.get("summary"),
                    description=journal.get("description"),
                    dtstart=dtstart,
                    related_to=journal.get("related_to", []),
                    account_alias=account_alias,
                )

                results.append(
                    OperationResult(
                        index=start_idx + idx,
                        success=True,
                        uid=created_journal.uid,
                        duration_ms=(time.time() - op_start) * 1000,
                    )
                )
            except Exception as e:
                results.append(
                    OperationResult(
                        index=start_idx + idx,
                        success=False,
                        error=str(e),
                        duration_ms=(time.time() - op_start) * 1000,
                    )
                )

        return results

    def _rollback_created_events(self, calendar_uid: str, uids: List[str]):
        """Rollback created events in case of atomic operation failure."""
        # Delete all created events
        if self.event_manager:
            for uid in uids:
                try:
                    self.event_manager.delete_event(calendar_uid, uid)
                    logger.debug(f"Successfully rolled back event {uid}")
                except Exception as e:
                    logger.warning(f"Failed to rollback event {uid}: {e}")

    def _rollback_created_tasks(self, calendar_uid: str, uids: List[str]):
        """Rollback created tasks in case of atomic operation failure."""
        # Delete all created tasks
        if self.task_manager:
            for uid in uids:
                try:
                    self.task_manager.delete_task(calendar_uid, uid)
                    logger.debug(f"Successfully rolled back task {uid}")
                except Exception as e:
                    logger.warning(f"Failed to rollback task {uid}: {e}")

    def _rollback_created_journals(self, calendar_uid: str, uids: List[str]):
        """Rollback created journals in case of atomic operation failure."""
        # Delete all created journals
        if self.journal_manager:
            for uid in uids:
                try:
                    self.journal_manager.delete_journal(calendar_uid, uid)
                    logger.debug(f"Successfully rolled back journal {uid}")
                except Exception as e:
                    logger.warning(f"Failed to rollback journal {uid}: {e}")

    def bulk_delete_events(
        self, calendar_uid: str, event_uids: List[str], options: BulkOptions = None
    ) -> BulkResult:
        """Delete multiple events efficiently."""
        if options is None:
            options = BulkOptions()

        start_time = time.time()
        result = BulkResult(total=len(event_uids), successful=0, failed=0)

        if not self.event_manager:
            raise ValueError("EventManager not provided to BulkOperationManager")

        if options.dry_run:
            for idx in range(len(event_uids)):
                result.results.append(
                    OperationResult(
                        index=idx, success=True, uid=event_uids[idx], duration_ms=0.1
                    )
                )
            result.successful = len(event_uids)
        else:
            for batch_start in range(0, len(event_uids), options.max_parallel):
                batch_end = min(batch_start + options.max_parallel, len(event_uids))
                batch_uids = event_uids[batch_start:batch_end]

                for idx, uid in enumerate(batch_uids):
                    op_start = time.time()
                    try:
                        self.event_manager.delete_event(calendar_uid, uid)
                        result.results.append(
                            OperationResult(
                                index=batch_start + idx,
                                success=True,
                                uid=uid,
                                duration_ms=(time.time() - op_start) * 1000,
                            )
                        )
                        result.successful += 1
                    except Exception as e:
                        result.results.append(
                            OperationResult(
                                index=batch_start + idx,
                                success=False,
                                uid=uid,
                                error=str(e),
                                duration_ms=(time.time() - op_start) * 1000,
                            )
                        )
                        result.failed += 1

                        if options.mode == BulkOperationMode.FAIL_FAST:
                            break

                if options.mode == BulkOperationMode.FAIL_FAST and result.failed > 0:
                    break

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def bulk_delete_tasks(
        self, calendar_uid: str, task_uids: List[str], options: BulkOptions = None
    ) -> BulkResult:
        """Delete multiple tasks efficiently."""
        if options is None:
            options = BulkOptions()

        start_time = time.time()
        result = BulkResult(total=len(task_uids), successful=0, failed=0)

        if not self.task_manager:
            raise ValueError("TaskManager not provided to BulkOperationManager")

        if options.dry_run:
            for idx in range(len(task_uids)):
                result.results.append(
                    OperationResult(
                        index=idx, success=True, uid=task_uids[idx], duration_ms=0.1
                    )
                )
            result.successful = len(task_uids)
        else:
            for batch_start in range(0, len(task_uids), options.max_parallel):
                batch_end = min(batch_start + options.max_parallel, len(task_uids))
                batch_uids = task_uids[batch_start:batch_end]

                for idx, uid in enumerate(batch_uids):
                    op_start = time.time()
                    try:
                        self.task_manager.delete_task(calendar_uid, uid)
                        result.results.append(
                            OperationResult(
                                index=batch_start + idx,
                                success=True,
                                uid=uid,
                                duration_ms=(time.time() - op_start) * 1000,
                            )
                        )
                        result.successful += 1
                    except Exception as e:
                        result.results.append(
                            OperationResult(
                                index=batch_start + idx,
                                success=False,
                                uid=uid,
                                error=str(e),
                                duration_ms=(time.time() - op_start) * 1000,
                            )
                        )
                        result.failed += 1

                        if options.mode == BulkOperationMode.FAIL_FAST:
                            break

                if options.mode == BulkOperationMode.FAIL_FAST and result.failed > 0:
                    break

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def bulk_delete_journals(
        self, calendar_uid: str, journal_uids: List[str], options: BulkOptions = None
    ) -> BulkResult:
        """Delete multiple journals efficiently."""
        if options is None:
            options = BulkOptions()

        start_time = time.time()
        result = BulkResult(total=len(journal_uids), successful=0, failed=0)

        if not self.journal_manager:
            raise ValueError("JournalManager not provided to BulkOperationManager")

        if options.dry_run:
            for idx in range(len(journal_uids)):
                result.results.append(
                    OperationResult(
                        index=idx, success=True, uid=journal_uids[idx], duration_ms=0.1
                    )
                )
            result.successful = len(journal_uids)
        else:
            for batch_start in range(0, len(journal_uids), options.max_parallel):
                batch_end = min(batch_start + options.max_parallel, len(journal_uids))
                batch_uids = journal_uids[batch_start:batch_end]

                for idx, uid in enumerate(batch_uids):
                    op_start = time.time()
                    try:
                        self.journal_manager.delete_journal(calendar_uid, uid)
                        result.results.append(
                            OperationResult(
                                index=batch_start + idx,
                                success=True,
                                uid=uid,
                                duration_ms=(time.time() - op_start) * 1000,
                            )
                        )
                        result.successful += 1
                    except Exception as e:
                        result.results.append(
                            OperationResult(
                                index=batch_start + idx,
                                success=False,
                                uid=uid,
                                error=str(e),
                                duration_ms=(time.time() - op_start) * 1000,
                            )
                        )
                        result.failed += 1

                        if options.mode == BulkOperationMode.FAIL_FAST:
                            break

                if options.mode == BulkOperationMode.FAIL_FAST and result.failed > 0:
                    break

        result.duration_ms = (time.time() - start_time) * 1000
        return result
