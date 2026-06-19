"""
Unit tests for task management
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import caldav
import pytest
from icalendar import Calendar as iCalendar
from icalendar import Todo as iTodo

from chronos_mcp.calendars import CalendarManager
from chronos_mcp.exceptions import (
    CalendarNotFoundError,
    ChronosError,
    EventCreationError,
    EventDeletionError,
    TaskNotFoundError,
)
from chronos_mcp.models import TaskStatus
from chronos_mcp.tasks import TaskManager


class TestTaskManager:
    """Test task management functionality"""

    @pytest.fixture
    def mock_calendar_manager(self):
        """Mock CalendarManager"""
        manager = Mock(spec=CalendarManager)
        manager.accounts = Mock()
        manager.accounts.config = Mock()
        manager.accounts.config.config = Mock()
        manager.accounts.config.config.default_account = "test_account"
        return manager

    @pytest.fixture
    def mock_calendar(self):
        """Mock calendar object with full CalDAV feature support"""
        calendar = Mock()
        calendar.save_todo = Mock()
        calendar.save_event = Mock()
        calendar.todos = Mock()
        calendar.events = Mock()
        calendar.event_by_uid = Mock()
        return calendar

    @pytest.fixture
    def mock_calendar_basic(self):
        """Mock calendar object with basic CalDAV support (fallback mode)"""

        # Create a mock that only has specific methods
        class BasicCalendar:
            def __init__(self):
                self.save_event = Mock()
                self.events = Mock()
                # Explicitly no save_todo, todos, or event_by_uid methods

        return BasicCalendar()

    @pytest.fixture
    def sample_task_data(self):
        """Sample task data for testing"""
        return {
            "calendar_uid": "cal-123",
            "summary": "Test Task",
            "description": "Test task description",
            "due": datetime(2025, 12, 31, 23, 59, tzinfo=timezone.utc),
            "priority": 5,
            "status": TaskStatus.NEEDS_ACTION,
            "related_to": ["related-task-1", "related-task-2"],
            "account_alias": "test_account",
        }

    @pytest.fixture
    def sample_vtodo_ical(self):
        """Sample VTODO iCalendar data"""
        cal = iCalendar()
        task = iTodo()
        task.add("uid", "test-task-123")
        task.add("summary", "Test Task")
        task.add("description", "Test task description")
        task.add("dtstamp", datetime.now(timezone.utc))
        task.add("due", datetime(2025, 12, 31, 23, 59, tzinfo=timezone.utc))
        task.add("priority", 5)
        task.add("status", "NEEDS-ACTION")
        task.add("percent-complete", 0)
        task.add("related-to", "related-task-1")
        task.add("related-to", "related-task-2")
        cal.add_component(task)
        return cal.to_ical().decode("utf-8")

    @pytest.fixture
    def mock_caldav_task(self, sample_vtodo_ical):
        """Mock CalDAV task object"""
        task = Mock()
        task.data = sample_vtodo_ical
        task.delete = Mock()
        task.save = Mock()
        return task

    def test_init(self, mock_calendar_manager):
        """Test TaskManager initialization"""
        mgr = TaskManager(mock_calendar_manager)
        assert mgr.calendars == mock_calendar_manager

    def test_get_default_account_success(self, mock_calendar_manager):
        """Test _get_default_account returns configured default"""
        mgr = TaskManager(mock_calendar_manager)
        assert mgr._get_default_account() == "test_account"

    def test_get_default_account_failure(self, mock_calendar_manager):
        """Test _get_default_account handles exceptions gracefully"""
        mock_calendar_manager.accounts.config.config.default_account = None
        mgr = TaskManager(mock_calendar_manager)
        assert mgr._get_default_account() is None

    # Phase 1: Basic CRUD Operations (25% coverage target)

    @patch("chronos_mcp.tasks.uuid.uuid4")
    def test_create_task_minimal_success(self, mock_uuid, mock_calendar_manager, mock_calendar):
        """Test create_task with minimal parameters - modern server"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-task-123")

        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar

        mock_caldav_task = Mock()
        mock_calendar.save_todo.return_value = mock_caldav_task

        # Execute
        result = mgr.create_task(calendar_uid="cal-123", summary="Test Task")

        # Verify
        assert result is not None
        assert result.uid == "test-task-123"
        assert result.summary == "Test Task"
        assert result.status == TaskStatus.NEEDS_ACTION
        assert result.percent_complete == 0
        assert result.calendar_uid == "cal-123"

        mock_calendar_manager.get_calendar.assert_called_once()
        mock_calendar.save_todo.assert_called_once()

    @patch("chronos_mcp.tasks.uuid.uuid4")
    def test_create_task_full_parameters(
        self, mock_uuid, mock_calendar_manager, mock_calendar, sample_task_data
    ):
        """Test create_task with all parameters"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-task-123")

        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar

        mock_caldav_task = Mock()
        mock_calendar.save_todo.return_value = mock_caldav_task

        # Execute
        result = mgr.create_task(**sample_task_data)

        # Verify
        assert result is not None
        assert result.uid == "test-task-123"
        assert result.summary == sample_task_data["summary"]
        assert result.description == sample_task_data["description"]
        assert result.due == sample_task_data["due"]
        assert result.priority == sample_task_data["priority"]
        assert result.status == sample_task_data["status"]
        assert result.related_to == sample_task_data["related_to"]

    @patch("chronos_mcp.tasks.uuid.uuid4")
    def test_create_task_related_to_emits_reltype_parent(
        self, mock_uuid, mock_calendar_manager, mock_calendar
    ):
        """RELATED-TO on a subtask must be a typed, one-directional PARENT link.

        Regression test: an untyped RELATED-TO is treated as bidirectional by
        some clients (e.g. jtx Board), making the child appear as both a subtask
        and a parent. The child must declare RELTYPE=PARENT and nothing must be
        written to a second component (no reverse link on the parent).
        """
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="child-task-123")

        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.save_todo.return_value = Mock()

        mgr.create_task(
            calendar_uid="cal-123",
            summary="Child Task",
            related_to=["parent-task-uid"],
        )

        # Exactly one save, to the single (child) component — no parent mutation.
        mock_calendar.save_todo.assert_called_once()
        mock_calendar.save_event.assert_not_called()

        ical_data = mock_calendar.save_todo.call_args[0][0]
        assert ical_data.count("RELATED-TO") == 1
        assert "RELATED-TO;RELTYPE=PARENT:parent-task-uid" in ical_data
        # No untyped RELATED-TO line.
        assert "RELATED-TO:parent-task-uid" not in ical_data

    @patch("chronos_mcp.tasks.uuid.uuid4")
    def test_create_task_fallback_to_save_event(
        self, mock_uuid, mock_calendar_manager, mock_calendar
    ):
        """Test create_task falls back to save_event when save_todo fails"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-task-123")

        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar

        # Make save_todo fail
        mock_calendar.save_todo.side_effect = Exception("save_todo failed")
        mock_caldav_task = Mock()
        mock_calendar.save_event.return_value = mock_caldav_task

        # Execute
        result = mgr.create_task(calendar_uid="cal-123", summary="Test Task")

        # Verify
        assert result is not None
        mock_calendar.save_todo.assert_called_once()
        mock_calendar.save_event.assert_called_once()

    @patch("chronos_mcp.tasks.uuid.uuid4")
    def test_create_task_basic_server(self, mock_uuid, mock_calendar_manager, mock_calendar_basic):
        """Test create_task with basic server (no save_todo support)"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-task-123")

        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar_basic

        mock_caldav_task = Mock()
        mock_calendar_basic.save_event.return_value = mock_caldav_task

        # Execute
        result = mgr.create_task(calendar_uid="cal-123", summary="Test Task")

        # Verify
        assert result is not None
        assert result.summary == "Test Task"
        mock_calendar_basic.save_event.assert_called_once()
        # save_todo should not be called since it doesn't exist
        assert not hasattr(mock_calendar_basic, "save_todo")

    def test_get_task_success_event_by_uid(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """Test get_task success using event_by_uid method"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.return_value = mock_caldav_task

        # Execute
        result = mgr.get_task(task_uid="test-task-123", calendar_uid="cal-123")

        # Verify
        assert result is not None
        assert result.uid == "test-task-123"
        assert result.summary == "Test Task"
        mock_calendar.event_by_uid.assert_called_once_with("test-task-123")

    def test_list_tasks_success_todos_method(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """Test list_tasks success using todos() method"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.todos.return_value = [mock_caldav_task]

        # Execute
        result = mgr.list_tasks(calendar_uid="cal-123")

        # Verify
        assert len(result) == 1
        assert result[0].uid == "test-task-123"
        mock_calendar.todos.assert_called_once_with(include_completed=True)

    def test_list_tasks_with_status_filter(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """Test list_tasks with status filter"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.todos.return_value = [mock_caldav_task]

        # Execute
        result = mgr.list_tasks(calendar_uid="cal-123", status_filter=TaskStatus.NEEDS_ACTION)

        # Verify
        assert len(result) == 1
        assert result[0].status == TaskStatus.NEEDS_ACTION

    def test_list_tasks_includes_completed(self, mock_calendar_manager, mock_calendar):
        """Test list_tasks fetches completed tasks from server (issue #14)"""
        completed_vtodo = (
            "BEGIN:VTODO\r\n"
            "UID:completed-task-456\r\n"
            "SUMMARY:Done Task\r\n"
            "STATUS:COMPLETED\r\n"
            "PERCENT-COMPLETE:100\r\n"
            "DTSTAMP:20250101T000000Z\r\n"
            "END:VTODO\r\n"
        )
        completed_caldav = Mock()
        completed_caldav.data = completed_vtodo

        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.todos.return_value = [completed_caldav]

        result = mgr.list_tasks(calendar_uid="cal-123")

        assert len(result) == 1
        assert result[0].status == TaskStatus.COMPLETED
        mock_calendar.todos.assert_called_once_with(include_completed=True)

    def test_update_task_summary_only(self, mock_calendar_manager, mock_calendar, mock_caldav_task):
        """Test update_task updating only summary field"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.return_value = mock_caldav_task

        # Execute
        result = mgr.update_task(
            task_uid="test-task-123", calendar_uid="cal-123", summary="Updated Summary"
        )

        # Verify
        assert result is not None
        mock_caldav_task.save.assert_called_once()

    def test_update_task_related_to_emits_reltype_parent(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """update_task must also write RELATED-TO as a typed PARENT link."""
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.return_value = mock_caldav_task

        result = mgr.update_task(
            task_uid="test-task-123",
            calendar_uid="cal-123",
            related_to=["parent-task-uid"],
        )

        assert result is not None
        mock_caldav_task.save.assert_called_once()

        # The serialized payload that was saved back to the server.
        saved_ical = mock_caldav_task.data
        assert saved_ical.count("RELATED-TO") == 1
        assert "RELATED-TO;RELTYPE=PARENT:parent-task-uid" in saved_ical
        assert "RELATED-TO:parent-task-uid" not in saved_ical

    def test_delete_task_success_event_by_uid(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """Test delete_task success using event_by_uid"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.return_value = mock_caldav_task

        # Execute
        result = mgr.delete_task(calendar_uid="cal-123", task_uid="test-task-123")

        # Verify
        assert result is True
        mock_caldav_task.delete.assert_called_once()

    def test_parse_caldav_task_success(self, mock_calendar_manager, mock_caldav_task):
        """Test _parse_caldav_task successfully parses VTODO"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)

        # Execute
        result = mgr._parse_caldav_task(
            mock_caldav_task, calendar_uid="cal-123", account_alias="test_account"
        )

        # Verify
        assert result is not None
        assert result.uid == "test-task-123"
        assert result.summary == "Test Task"
        assert result.description == "Test task description"
        assert result.priority == 5
        assert result.status == TaskStatus.NEEDS_ACTION
        assert result.percent_complete == 0
        assert "related-task-1" in result.related_to
        assert "related-task-2" in result.related_to

    # Phase 2: Error Conditions (50% coverage target)

    def test_create_task_calendar_not_found(self, mock_calendar_manager):
        """Test create_task raises CalendarNotFoundError when calendar not found"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = None

        # Execute & Verify
        with pytest.raises(CalendarNotFoundError):
            mgr.create_task(calendar_uid="nonexistent-cal", summary="Test Task")

    def test_create_task_authorization_error(self, mock_calendar_manager, mock_calendar):
        """Test create_task handles CalDAV authorization errors"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.save_todo.side_effect = caldav.lib.error.AuthorizationError("Auth failed")
        mock_calendar.save_event.side_effect = caldav.lib.error.AuthorizationError("Auth failed")

        # Execute & Verify
        with pytest.raises(EventCreationError):
            mgr.create_task(calendar_uid="cal-123", summary="Test Task")

    def test_create_task_general_error(self, mock_calendar_manager, mock_calendar):
        """Test create_task handles general exceptions"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.save_todo.side_effect = Exception("Unexpected error")
        mock_calendar.save_event.side_effect = Exception("Unexpected error")

        # Execute & Verify
        with pytest.raises(EventCreationError):
            mgr.create_task(calendar_uid="cal-123", summary="Test Task")

    def test_get_task_calendar_not_found(self, mock_calendar_manager):
        """Test get_task raises CalendarNotFoundError when calendar not found"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = None

        # Execute & Verify
        with pytest.raises(CalendarNotFoundError):
            mgr.get_task(task_uid="test-task-123", calendar_uid="nonexistent-cal")

    def test_get_task_not_found_event_by_uid(self, mock_calendar_manager, mock_calendar):
        """Test get_task raises TaskNotFoundError when task not found via event_by_uid"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.side_effect = Exception("Task not found")
        mock_calendar.todos.return_value = []

        # Execute & Verify
        with pytest.raises(TaskNotFoundError):
            mgr.get_task(task_uid="nonexistent-task", calendar_uid="cal-123")

    def test_get_task_not_found_fallback_search(self, mock_calendar_manager, mock_calendar):
        """Test get_task raises TaskNotFoundError when task not found via fallback search"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.side_effect = Exception("Not found")
        mock_calendar.todos.return_value = []

        # Execute & Verify
        with pytest.raises(TaskNotFoundError):
            mgr.get_task(task_uid="nonexistent-task", calendar_uid="cal-123")

    def test_list_tasks_calendar_not_found(self, mock_calendar_manager):
        """Test list_tasks raises CalendarNotFoundError when calendar not found"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = None

        # Execute & Verify
        with pytest.raises(CalendarNotFoundError):
            mgr.list_tasks(calendar_uid="nonexistent-cal")

    def test_update_task_calendar_not_found(self, mock_calendar_manager):
        """Test update_task raises CalendarNotFoundError when calendar not found"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = None

        # Execute & Verify
        with pytest.raises(CalendarNotFoundError):
            mgr.update_task(
                task_uid="test-task-123",
                calendar_uid="nonexistent-cal",
                summary="Updated",
            )

    def test_update_task_not_found(self, mock_calendar_manager, mock_calendar):
        """Test update_task raises TaskNotFoundError when task not found"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.side_effect = Exception("Not found")
        mock_calendar.todos.return_value = []

        # Execute & Verify
        with pytest.raises(TaskNotFoundError):
            mgr.update_task(task_uid="nonexistent-task", calendar_uid="cal-123", summary="Updated")

    def test_delete_task_calendar_not_found(self, mock_calendar_manager):
        """Test delete_task raises CalendarNotFoundError when calendar not found"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = None

        # Execute & Verify
        with pytest.raises(CalendarNotFoundError):
            mgr.delete_task(calendar_uid="nonexistent-cal", task_uid="test-task-123")

    def test_delete_task_not_found(self, mock_calendar_manager, mock_calendar):
        """Test delete_task raises TaskNotFoundError when task not found"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.side_effect = Exception("Not found")
        mock_calendar.todos.return_value = []

        # Execute & Verify
        with pytest.raises(TaskNotFoundError):
            mgr.delete_task(calendar_uid="cal-123", task_uid="nonexistent-task")

    def test_delete_task_general_error(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """Test delete_task handles general errors during deletion"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.return_value = mock_caldav_task
        mock_caldav_task.delete.side_effect = Exception("Unexpected deletion error")

        # Execute & Verify - when task is found but deletion fails, raises EventDeletionError
        # (not TaskNotFoundError, since the task was successfully found)
        with pytest.raises(EventDeletionError):
            mgr.delete_task(calendar_uid="cal-123", task_uid="test-task-123")

    # Phase 3: Server Compatibility (70% coverage target)

    def test_get_task_fallback_to_todos_search(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """Test get_task falls back to searching todos when event_by_uid fails"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.side_effect = Exception("Method failed")
        mock_calendar.todos.return_value = [mock_caldav_task]

        # Execute
        result = mgr.get_task(task_uid="test-task-123", calendar_uid="cal-123")

        # Verify
        assert result is not None
        assert result.uid == "test-task-123"
        mock_calendar.event_by_uid.assert_called_once()
        mock_calendar.todos.assert_called_once_with(include_completed=True)

    def test_get_task_fallback_to_events_search(
        self, mock_calendar_manager, mock_calendar_basic, mock_caldav_task
    ):
        """Test get_task falls back to searching events when todos not available"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar_basic
        mock_calendar_basic.events.return_value = [mock_caldav_task]

        # Execute
        result = mgr.get_task(task_uid="test-task-123", calendar_uid="cal-123")

        # Verify
        assert result is not None
        assert result.uid == "test-task-123"
        mock_calendar_basic.events.assert_called_once()

    def test_list_tasks_fallback_to_events(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """Test list_tasks falls back to events when todos() fails"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.todos.side_effect = Exception("todos() failed")
        mock_calendar.events.return_value = [mock_caldav_task]

        # Execute
        result = mgr.list_tasks(calendar_uid="cal-123")

        # Verify
        assert len(result) == 1
        assert result[0].uid == "test-task-123"
        mock_calendar.todos.assert_called_once_with(include_completed=True)
        mock_calendar.events.assert_called_once()

    def test_list_tasks_basic_server_events_only(
        self, mock_calendar_manager, mock_calendar_basic, mock_caldav_task
    ):
        """Test list_tasks on basic server using events() only"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar_basic
        mock_calendar_basic.events.return_value = [mock_caldav_task]

        # Execute
        result = mgr.list_tasks(calendar_uid="cal-123")

        # Verify
        assert len(result) == 1
        assert result[0].uid == "test-task-123"
        mock_calendar_basic.events.assert_called_once()

    def test_update_task_fallback_search(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """Test update_task falls back to searching todos when event_by_uid fails"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.side_effect = Exception("Method failed")
        mock_calendar.todos.return_value = [mock_caldav_task]

        # Execute
        result = mgr.update_task(
            task_uid="test-task-123", calendar_uid="cal-123", summary="Updated Summary"
        )

        # Verify
        assert result is not None
        mock_caldav_task.save.assert_called_once()

    def test_delete_task_fallback_search(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """Test delete_task falls back to searching todos when event_by_uid fails"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.side_effect = Exception("Method failed")
        mock_calendar.todos.return_value = [mock_caldav_task]

        # Execute
        result = mgr.delete_task(calendar_uid="cal-123", task_uid="test-task-123")

        # Verify
        assert result is True
        mock_caldav_task.delete.assert_called_once()

    def test_delete_task_basic_server_events_search(
        self, mock_calendar_manager, mock_calendar_basic, mock_caldav_task
    ):
        """Test delete_task on basic server using events() search"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar_basic
        mock_calendar_basic.events.return_value = [mock_caldav_task]

        # Execute
        result = mgr.delete_task(calendar_uid="cal-123", task_uid="test-task-123")

        # Verify
        assert result is True
        mock_caldav_task.delete.assert_called_once()

    # Phase 4: Edge Cases and Validation (80% coverage target)

    def test_create_task_priority_validation(self, mock_calendar_manager, mock_calendar):
        """Test create_task validates priority range (1-9)"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_caldav_task = Mock()
        mock_calendar.save_todo.return_value = mock_caldav_task

        # Test invalid priority (outside 1-9 range)
        with patch("chronos_mcp.tasks.uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.__str__ = Mock(return_value="test-task-123")

            result = mgr.create_task(
                calendar_uid="cal-123",
                summary="Test Task",
                priority=10,  # Invalid priority
            )

            # Priority should be ignored for invalid values
            assert result is not None

    def test_update_task_all_fields(self, mock_calendar_manager, mock_calendar, mock_caldav_task):
        """Test update_task updating all possible fields"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.return_value = mock_caldav_task

        due_date = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        # Execute
        result = mgr.update_task(
            task_uid="test-task-123",
            calendar_uid="cal-123",
            summary="Updated Summary",
            description="Updated Description",
            due=due_date,
            priority=3,
            status=TaskStatus.IN_PROCESS,
            percent_complete=50,
            related_to=["new-related-1", "new-related-2"],
        )

        # Verify
        assert result is not None
        mock_caldav_task.save.assert_called_once()

    def test_update_task_clear_optional_fields(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """Test update_task can clear optional fields by setting to None"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.return_value = mock_caldav_task

        # Execute - clear description, due, priority, related_to
        result = mgr.update_task(
            task_uid="test-task-123",
            calendar_uid="cal-123",
            description="",  # Clear description
            due=None,  # Clear due date
            priority=None,  # Clear priority
            related_to=[],  # Clear related tasks
        )

        # Verify
        assert result is not None
        mock_caldav_task.save.assert_called_once()

    def test_update_task_invalid_priority_range(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """Test update_task handles invalid priority values"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.return_value = mock_caldav_task

        # Execute with invalid priority
        result = mgr.update_task(
            task_uid="test-task-123",
            calendar_uid="cal-123",
            priority=15,  # Invalid priority (>9)
        )

        # Verify - should still succeed but ignore invalid priority
        assert result is not None
        mock_caldav_task.save.assert_called_once()

    def test_update_task_percent_complete_validation(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """Test update_task validates percent_complete range (0-100)"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.return_value = mock_caldav_task

        # Execute with valid percent_complete
        result = mgr.update_task(
            task_uid="test-task-123", calendar_uid="cal-123", percent_complete=75
        )

        # Verify
        assert result is not None
        mock_caldav_task.save.assert_called_once()

    def test_update_task_parsing_error(
        self, mock_calendar_manager, mock_calendar, mock_caldav_task
    ):
        """Test update_task handles parsing errors gracefully"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.return_value = mock_caldav_task

        # Make iCalendar parsing fail
        mock_caldav_task.data = "invalid ical data"

        # Execute & Verify
        with pytest.raises(EventCreationError):
            mgr.update_task(task_uid="test-task-123", calendar_uid="cal-123", summary="Updated")

    def test_parse_caldav_task_malformed_data(self, mock_calendar_manager):
        """Test _parse_caldav_task handles malformed iCalendar data"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_caldav_event = Mock()
        mock_caldav_event.data = "invalid ical data"

        # Execute
        result = mgr._parse_caldav_task(
            mock_caldav_event, calendar_uid="cal-123", account_alias="test_account"
        )

        # Verify - should return None for malformed data
        assert result is None

    def test_parse_caldav_task_no_vtodo_component(self, mock_calendar_manager):
        """Test _parse_caldav_task handles iCalendar without VTODO component"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)

        # Create iCalendar with VEVENT instead of VTODO
        cal = iCalendar()
        from icalendar import Event as iEvent

        event = iEvent()
        event.add("uid", "test-event-123")
        event.add("summary", "Test Event")
        cal.add_component(event)

        mock_caldav_event = Mock()
        mock_caldav_event.data = cal.to_ical().decode("utf-8")

        # Execute
        result = mgr._parse_caldav_task(
            mock_caldav_event, calendar_uid="cal-123", account_alias="test_account"
        )

        # Verify - should return None since no VTODO component
        assert result is None

    def test_parse_caldav_task_missing_optional_fields(self, mock_calendar_manager):
        """Test _parse_caldav_task handles missing optional fields gracefully"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)

        # Create minimal VTODO with only required fields
        cal = iCalendar()
        task = iTodo()
        task.add("uid", "minimal-task-123")
        task.add("summary", "Minimal Task")
        task.add("dtstamp", datetime.now(timezone.utc))
        # No description, due, priority, etc.
        cal.add_component(task)

        mock_caldav_event = Mock()
        mock_caldav_event.data = cal.to_ical().decode("utf-8")

        # Execute
        result = mgr._parse_caldav_task(
            mock_caldav_event, calendar_uid="cal-123", account_alias="test_account"
        )

        # Verify - should handle missing fields gracefully
        assert result is not None
        assert result.uid == "minimal-task-123"
        assert result.summary == "Minimal Task"
        assert result.description is None
        assert result.due is None
        assert result.priority is None
        assert result.percent_complete == 0
        assert result.status == TaskStatus.NEEDS_ACTION
        assert result.related_to == []

    def test_parse_caldav_task_invalid_status_value(self, mock_calendar_manager):
        """Test _parse_caldav_task handles invalid status values gracefully"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)

        # Create a valid VTODO with an invalid status value
        cal = iCalendar()
        task = iTodo()
        task.add("uid", "invalid-status-task")
        task.add("summary", "Task with Invalid Status")
        task.add("dtstamp", datetime.now(timezone.utc))
        task.add("priority", 5)
        task.add("percent-complete", 25)
        task.add(
            "status", "UNKNOWN-STATUS"
        )  # This will be accepted by iCalendar but invalid for our enum
        cal.add_component(task)

        mock_caldav_event = Mock()
        mock_caldav_event.data = cal.to_ical().decode("utf-8")

        # Execute
        result = mgr._parse_caldav_task(
            mock_caldav_event, calendar_uid="cal-123", account_alias="test_account"
        )

        # Verify - should handle invalid status gracefully
        assert result is not None
        assert result.uid == "invalid-status-task"
        assert result.priority == 5
        assert result.percent_complete == 25
        assert (
            result.status == TaskStatus.NEEDS_ACTION
        )  # Should fallback to default for invalid status

    def test_get_task_general_error_handling(self, mock_calendar_manager, mock_calendar):
        """Test get_task handles unexpected errors gracefully"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.side_effect = RuntimeError("Unexpected error")
        mock_calendar.todos.side_effect = RuntimeError("Unexpected error")

        # Execute & Verify
        with pytest.raises(ChronosError):
            mgr.get_task(task_uid="test-task-123", calendar_uid="cal-123")

    def test_list_tasks_handles_parsing_errors(
        self, mock_calendar_manager, mock_calendar, sample_vtodo_ical
    ):
        """Test list_tasks continues when individual task parsing fails"""
        # Setup
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar

        # Create one valid and one invalid task
        valid_task = Mock()
        valid_task.data = sample_vtodo_ical

        invalid_task = Mock()
        invalid_task.data = "invalid ical data"

        mock_calendar.todos.return_value = [valid_task, invalid_task]

        # Execute
        result = mgr.list_tasks(calendar_uid="cal-123")

        # Verify - should return only the valid task
        assert len(result) == 1
        assert result[0].uid == "test-task-123"

    @patch("chronos_mcp.tasks.uuid.uuid4")
    def test_create_task_with_request_id(self, mock_uuid, mock_calendar_manager, mock_calendar):
        """Test create_task respects provided request_id"""
        # Setup
        mock_uuid.return_value.__str__ = Mock(return_value="test-task-123")
        mgr = TaskManager(mock_calendar_manager)
        mock_calendar_manager.get_calendar.return_value = mock_calendar
        mock_caldav_task = Mock()
        mock_calendar.save_todo.return_value = mock_caldav_task

        # Execute
        result = mgr.create_task(
            calendar_uid="cal-123", summary="Test Task", request_id="custom-request-id"
        )

        # Verify
        assert result is not None
        mock_calendar_manager.get_calendar.assert_called_with(
            "cal-123", None, request_id="custom-request-id"
        )
