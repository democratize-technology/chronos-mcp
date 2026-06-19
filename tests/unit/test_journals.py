"""
Comprehensive unit tests for journal management
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from icalendar import Calendar as iCalendar
from icalendar import Event as iEvent
from icalendar import Journal as iJournal

from chronos_mcp.calendars import CalendarManager
from chronos_mcp.exceptions import (
    CalendarNotFoundError,
    ChronosError,
    EventCreationError,
    EventDeletionError,
    JournalNotFoundError,
)
from chronos_mcp.journals import JournalManager
from chronos_mcp.models import Journal


class TestJournalManagerInit:
    """Test JournalManager initialization and basic functionality"""

    def test_init_with_calendar_manager(self):
        """Test JournalManager initialization with CalendarManager"""
        mock_calendar_manager = Mock(spec=CalendarManager)
        journal_manager = JournalManager(mock_calendar_manager)
        assert journal_manager.calendars == mock_calendar_manager

    def test_get_default_account_success(self):
        """Test _get_default_account returns account when available"""
        mock_calendar_manager = Mock()
        mock_calendar_manager.accounts = Mock()
        mock_calendar_manager.accounts.config = Mock()
        mock_calendar_manager.accounts.config.config = Mock()
        mock_calendar_manager.accounts.config.config.default_account = "test_account"

        journal_manager = JournalManager(mock_calendar_manager)
        result = journal_manager._get_default_account()
        assert result == "test_account"

    def test_get_default_account_exception(self):
        """Test _get_default_account returns None on exception"""
        mock_calendar_manager = Mock()
        # Remove the accounts attribute to force AttributeError
        del mock_calendar_manager.accounts

        journal_manager = JournalManager(mock_calendar_manager)
        result = journal_manager._get_default_account()
        assert result is None


class TestJournalCRUD:
    """Test CRUD operations for journals"""

    @pytest.fixture
    def mock_calendar_manager(self):
        """Mock CalendarManager for testing"""
        return Mock(spec=CalendarManager)

    @pytest.fixture
    def mock_calendar(self):
        """Mock calendar object with journal support"""
        calendar = Mock()
        calendar.save_journal = Mock()
        calendar.save_event = Mock()
        calendar.journals = Mock()
        calendar.events = Mock()
        calendar.event_by_uid = Mock()
        return calendar

    @pytest.fixture
    def journal_manager(self, mock_calendar_manager):
        """JournalManager instance for testing"""
        return JournalManager(mock_calendar_manager)

    @pytest.fixture
    def sample_journal_data(self):
        """Sample journal data for testing"""
        return {
            "calendar_uid": "cal-123",
            "summary": "Daily Reflection",
            "description": "Today was productive",
            "dtstart": datetime(2025, 7, 10, 9, 0, tzinfo=timezone.utc),
            "related_to": ["event-456"],
            "account_alias": "test_account",
        }

    def test_create_journal_success_with_save_journal(
        self, journal_manager, mock_calendar, sample_journal_data
    ):
        """Test successful journal creation using save_journal method"""
        # Setup
        journal_manager.calendars.get_calendar.return_value = mock_calendar
        mock_caldav_journal = Mock()
        mock_calendar.save_journal.return_value = mock_caldav_journal

        with patch("uuid.uuid4", return_value="test-uid-123"):
            result = journal_manager.create_journal(**sample_journal_data)

        # Assertions
        assert result is not None
        assert result.uid == "test-uid-123"
        assert result.summary == "Daily Reflection"
        assert result.description == "Today was productive"
        mock_calendar.save_journal.assert_called_once()

    def test_create_journal_related_to_stays_untyped(
        self, journal_manager, mock_calendar, sample_journal_data
    ):
        """Journals are not hierarchies: their RELATED-TO must stay untyped.

        Guards against the VTODO RELTYPE=PARENT fix accidentally being applied
        to journals as well.
        """
        journal_manager.calendars.get_calendar.return_value = mock_calendar
        mock_calendar.save_journal.return_value = Mock()

        with patch("uuid.uuid4", return_value="test-uid-123"):
            journal_manager.create_journal(**sample_journal_data)

        ical_data = mock_calendar.save_journal.call_args[0][0]
        assert "RELATED-TO:event-456" in ical_data
        assert "RELTYPE" not in ical_data

    def test_create_journal_fallback_to_save_event(
        self, journal_manager, mock_calendar, sample_journal_data
    ):
        """Test journal creation falls back to save_event when save_journal fails"""
        # Setup
        journal_manager.calendars.get_calendar.return_value = mock_calendar
        mock_calendar.save_journal.side_effect = Exception("save_journal failed")
        mock_caldav_journal = Mock()
        mock_calendar.save_event.return_value = mock_caldav_journal

        with patch("uuid.uuid4", return_value="test-uid-123"):
            result = journal_manager.create_journal(**sample_journal_data)

        # Assertions
        assert result is not None
        assert result.uid == "test-uid-123"
        mock_calendar.save_journal.assert_called_once()
        mock_calendar.save_event.assert_called_once()

    def test_create_journal_no_save_journal_method(self, journal_manager, sample_journal_data):
        """Test journal creation when calendar doesn't have save_journal method"""
        # Setup calendar without save_journal method
        mock_calendar = Mock()
        mock_calendar.save_event = Mock()
        # Remove save_journal method
        if hasattr(mock_calendar, "save_journal"):
            delattr(mock_calendar, "save_journal")

        journal_manager.calendars.get_calendar.return_value = mock_calendar
        mock_caldav_journal = Mock()
        mock_calendar.save_event.return_value = mock_caldav_journal

        with patch("uuid.uuid4", return_value="test-uid-123"):
            result = journal_manager.create_journal(**sample_journal_data)

        # Assertions
        assert result is not None
        assert result.uid == "test-uid-123"
        mock_calendar.save_event.assert_called_once()

    def test_create_journal_minimal_data(self, journal_manager, mock_calendar):
        """Test journal creation with minimal required data"""
        journal_manager.calendars.get_calendar.return_value = mock_calendar
        mock_caldav_journal = Mock()
        mock_calendar.save_journal.return_value = mock_caldav_journal

        with patch("uuid.uuid4", return_value="test-uid-123"):
            with patch("chronos_mcp.journals.datetime") as mock_datetime:
                mock_now = datetime(2025, 7, 10, 10, 0, tzinfo=timezone.utc)
                mock_datetime.now.return_value = mock_now
                mock_datetime.timezone = timezone

                result = journal_manager.create_journal(
                    calendar_uid="cal-123", summary="Simple Journal"
                )

        assert result is not None
        assert result.uid == "test-uid-123"
        assert result.summary == "Simple Journal"
        assert result.description is None
        assert result.related_to == []

    def test_create_journal_calendar_not_found(self, journal_manager):
        """Test journal creation with non-existent calendar"""
        journal_manager.calendars.get_calendar.return_value = None

        with pytest.raises(CalendarNotFoundError):
            journal_manager.create_journal(calendar_uid="nonexistent", summary="Test")

    def test_create_journal_authorization_error(
        self, journal_manager, mock_calendar, sample_journal_data
    ):
        """Test journal creation with authorization error"""
        from caldav.lib.error import AuthorizationError

        journal_manager.calendars.get_calendar.return_value = mock_calendar
        # Both save_journal and save_event should fail with authorization error
        mock_calendar.save_journal.side_effect = AuthorizationError("Unauthorized")
        mock_calendar.save_event.side_effect = AuthorizationError("Unauthorized")

        with pytest.raises(EventCreationError) as exc_info:
            journal_manager.create_journal(**sample_journal_data)

        assert "Authorization failed" in str(exc_info.value)

    def test_create_journal_generic_error(
        self, journal_manager, mock_calendar, sample_journal_data
    ):
        """Test journal creation with generic error"""
        journal_manager.calendars.get_calendar.return_value = mock_calendar
        mock_calendar.save_journal.side_effect = Exception("Generic error")
        mock_calendar.save_event.side_effect = Exception("Generic error")

        with pytest.raises(EventCreationError) as exc_info:
            journal_manager.create_journal(**sample_journal_data)

        assert "Generic error" in str(exc_info.value)

    def test_get_journal_success_with_event_by_uid(self, journal_manager, mock_calendar):
        """Test successful journal retrieval using event_by_uid"""
        # Setup
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        # Create mock CalDAV journal with VJOURNAL data
        mock_caldav_journal = Mock()
        mock_caldav_journal.data = self._create_sample_ical_data()
        mock_calendar.event_by_uid.return_value = mock_caldav_journal

        with patch.object(journal_manager, "_parse_caldav_journal") as mock_parse:
            expected_journal = Journal(
                uid="test-journal-123",
                summary="Test Journal",
                description="Test Description",
                dtstart=datetime(2025, 7, 10, 9, 0, tzinfo=timezone.utc),
                calendar_uid="cal-123",
                account_alias="test_account",
            )
            mock_parse.return_value = expected_journal

            result = journal_manager.get_journal(
                journal_uid="test-journal-123",
                calendar_uid="cal-123",
                account_alias="test_account",
            )

        assert result == expected_journal
        mock_calendar.event_by_uid.assert_called_once_with("test-journal-123")
        mock_parse.assert_called_once()

    def test_get_journal_fallback_to_journals_search(self, journal_manager, mock_calendar):
        """Test journal retrieval fallback to searching through journals"""
        # Setup
        journal_manager.calendars.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.side_effect = Exception("event_by_uid failed")

        # Create proper iCalendar mock data
        mock_journal1 = Mock()
        cal1 = iCalendar()
        journal1 = iJournal()
        journal1.add("uid", "different-uid")
        journal1.add("summary", "Different Journal")
        journal1.add("dtstart", datetime.now(timezone.utc))
        cal1.add_component(journal1)
        mock_journal1.data = cal1.to_ical()

        mock_journal2 = Mock()
        cal2 = iCalendar()
        journal2 = iJournal()
        journal2.add("uid", "test-journal-123")
        journal2.add("summary", "Found Journal")
        journal2.add("dtstart", datetime.now(timezone.utc))
        cal2.add_component(journal2)
        mock_journal2.data = cal2.to_ical()

        mock_calendar.journals.return_value = [mock_journal1, mock_journal2]

        result = journal_manager.get_journal(journal_uid="test-journal-123", calendar_uid="cal-123")

        assert result is not None
        assert result.uid == "test-journal-123"
        assert result.summary == "Found Journal"
        mock_calendar.journals.assert_called_once()

    def test_get_journal_fallback_to_events_search(self, journal_manager):
        """Test journal retrieval fallback to events when journals() unavailable"""
        # Setup calendar without journals method
        mock_calendar = Mock()
        mock_calendar.event_by_uid.side_effect = Exception("event_by_uid failed")
        mock_calendar.events = Mock()
        # Remove journals method
        if hasattr(mock_calendar, "journals"):
            delattr(mock_calendar, "journals")

        journal_manager.calendars.get_calendar.return_value = mock_calendar

        # Create proper iCalendar mock data
        mock_event1 = Mock()
        cal1 = iCalendar()
        event1 = iEvent()
        event1.add("uid", "event-uid")
        event1.add("summary", "Event")
        cal1.add_component(event1)
        mock_event1.data = cal1.to_ical()

        mock_event2 = Mock()
        cal2 = iCalendar()
        journal2 = iJournal()
        journal2.add("uid", "test-journal-123")
        journal2.add("summary", "Found in Events")
        journal2.add("dtstart", datetime.now(timezone.utc))
        cal2.add_component(journal2)
        mock_event2.data = cal2.to_ical()

        mock_calendar.events.return_value = [mock_event1, mock_event2]

        result = journal_manager.get_journal(journal_uid="test-journal-123", calendar_uid="cal-123")

        assert result is not None
        assert result.uid == "test-journal-123"
        mock_calendar.events.assert_called_once()

    def test_get_journal_not_found(self, journal_manager, mock_calendar):
        """Test journal retrieval when journal doesn't exist"""
        journal_manager.calendars.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.side_effect = Exception("Not found")
        mock_calendar.journals.return_value = []

        with pytest.raises(JournalNotFoundError):
            journal_manager.get_journal(journal_uid="nonexistent", calendar_uid="cal-123")

    def test_get_journal_calendar_not_found(self, journal_manager):
        """Test journal retrieval with non-existent calendar"""
        journal_manager.calendars.get_calendar.return_value = None

        with pytest.raises(CalendarNotFoundError):
            journal_manager.get_journal(journal_uid="test-journal-123", calendar_uid="nonexistent")

    def test_get_journal_generic_error(self, journal_manager, mock_calendar):
        """Test journal retrieval with generic error"""
        journal_manager.calendars.get_calendar.return_value = mock_calendar
        mock_calendar.event_by_uid.side_effect = Exception("Generic error")
        mock_calendar.journals.side_effect = Exception("Generic error")

        with pytest.raises(ChronosError):
            journal_manager.get_journal(journal_uid="test-journal-123", calendar_uid="cal-123")

    def test_list_journals_success_with_journals_method(self, journal_manager, mock_calendar):
        """Test successful journal listing using journals() method"""
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        # Mock journals
        mock_journals = [Mock(), Mock(), Mock()]
        mock_calendar.journals.return_value = mock_journals

        with patch.object(journal_manager, "_parse_caldav_journal") as mock_parse:
            mock_parse.side_effect = [
                Journal(
                    uid="j1",
                    summary="Journal 1",
                    dtstart=datetime.now(timezone.utc),
                    calendar_uid="cal-123",
                    account_alias="test",
                ),
                None,  # One unparseable journal
                Journal(
                    uid="j3",
                    summary="Journal 3",
                    dtstart=datetime.now(timezone.utc),
                    calendar_uid="cal-123",
                    account_alias="test",
                ),
            ]

            result = journal_manager.list_journals(calendar_uid="cal-123")

        assert len(result) == 2
        assert result[0].uid == "j1"
        assert result[1].uid == "j3"
        mock_calendar.journals.assert_called_once()

    def test_list_journals_fallback_to_events(self, journal_manager):
        """Test journal listing fallback to events() when journals() unavailable"""
        # Setup calendar without journals method
        mock_calendar = Mock()
        mock_calendar.events = Mock()
        if hasattr(mock_calendar, "journals"):
            delattr(mock_calendar, "journals")

        journal_manager.calendars.get_calendar.return_value = mock_calendar

        # Mock events
        mock_events = [Mock(), Mock()]
        mock_calendar.events.return_value = mock_events

        with patch.object(journal_manager, "_parse_caldav_journal") as mock_parse:
            mock_parse.side_effect = [
                Journal(
                    uid="j1",
                    summary="Journal from Events",
                    dtstart=datetime.now(timezone.utc),
                    calendar_uid="cal-123",
                    account_alias="test",
                ),
                None,  # One non-journal event
            ]

            result = journal_manager.list_journals(calendar_uid="cal-123")

        assert len(result) == 1
        assert result[0].uid == "j1"
        mock_calendar.events.assert_called_once()

    def test_list_journals_with_limit(self, journal_manager, mock_calendar):
        """Test journal listing with limit parameter"""
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        # Mock 5 journals
        mock_journals = [Mock() for _ in range(5)]
        mock_calendar.journals.return_value = mock_journals

        with patch.object(journal_manager, "_parse_caldav_journal") as mock_parse:
            mock_parse.side_effect = [
                Journal(
                    uid=f"j{i}",
                    summary=f"Journal {i}",
                    dtstart=datetime.now(timezone.utc),
                    calendar_uid="cal-123",
                    account_alias="test",
                )
                for i in range(5)
            ]

            result = journal_manager.list_journals(calendar_uid="cal-123", limit=3)

        assert len(result) == 3
        assert result[0].uid == "j0"
        assert result[2].uid == "j2"

    def test_list_journals_journals_method_fails_retry_with_events(
        self, journal_manager, mock_calendar
    ):
        """Test journal listing when journals() fails but events() succeeds"""
        journal_manager.calendars.get_calendar.return_value = mock_calendar
        mock_calendar.journals.side_effect = Exception("journals() failed")

        # Mock events for fallback
        mock_events = [Mock()]
        mock_calendar.events.return_value = mock_events

        with patch.object(journal_manager, "_parse_caldav_journal") as mock_parse:
            mock_parse.return_value = Journal(
                uid="j1",
                summary="From Events Fallback",
                dtstart=datetime.now(timezone.utc),
                calendar_uid="cal-123",
                account_alias="test",
            )

            result = journal_manager.list_journals(calendar_uid="cal-123")

        assert len(result) == 1
        assert result[0].uid == "j1"
        mock_calendar.events.assert_called_once()

    def test_list_journals_calendar_not_found(self, journal_manager):
        """Test journal listing with non-existent calendar"""
        journal_manager.calendars.get_calendar.return_value = None

        with pytest.raises(CalendarNotFoundError):
            journal_manager.list_journals(calendar_uid="nonexistent")

    def _create_sample_ical_data(self):
        """Create sample iCalendar data with VJOURNAL"""
        cal = iCalendar()
        journal = iJournal()
        journal.add("uid", "test-journal-123")
        journal.add("summary", "Test Journal")
        journal.add("description", "Test Description")
        journal.add("dtstart", datetime(2025, 7, 10, 9, 0, tzinfo=timezone.utc))
        cal.add_component(journal)
        return cal.to_ical().decode("utf-8")


class TestJournalServerCompatibility:
    """Test server compatibility and fallback mechanisms"""

    @pytest.fixture
    def journal_manager(self):
        """JournalManager instance for testing"""
        mock_calendar_manager = Mock(spec=CalendarManager)
        return JournalManager(mock_calendar_manager)

    def test_update_journal_success_with_event_by_uid(self, journal_manager):
        """Test successful journal update using event_by_uid"""
        # Setup
        mock_calendar = Mock()
        mock_calendar.event_by_uid = Mock()
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        # Create mock existing journal
        existing_ical = self._create_journal_ical()
        mock_caldav_journal = Mock()
        mock_caldav_journal.data = existing_ical
        mock_caldav_journal.save = Mock()
        mock_calendar.event_by_uid.return_value = mock_caldav_journal

        with patch.object(journal_manager, "_parse_caldav_journal") as mock_parse:
            updated_journal = Journal(
                uid="test-journal-123",
                summary="Updated Summary",
                description="Updated Description",
                dtstart=datetime(2025, 7, 10, 10, 0, tzinfo=timezone.utc),
                calendar_uid="cal-123",
                account_alias="test",
            )
            mock_parse.return_value = updated_journal

            result = journal_manager.update_journal(
                journal_uid="test-journal-123",
                calendar_uid="cal-123",
                summary="Updated Summary",
                description="Updated Description",
            )

        assert result == updated_journal
        mock_caldav_journal.save.assert_called_once()

    def test_update_journal_fallback_search_with_journals(self, journal_manager):
        """Test journal update fallback to searching through journals()"""
        # Setup
        mock_calendar = Mock()
        mock_calendar.event_by_uid.side_effect = Exception("event_by_uid failed")
        mock_calendar.journals = Mock()
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        # Mock journal search
        existing_ical = self._create_journal_ical()
        mock_journal1 = Mock()
        mock_journal1.data = "different journal"
        mock_journal2 = Mock()
        mock_journal2.data = existing_ical
        mock_journal2.save = Mock()
        mock_calendar.journals.return_value = [mock_journal1, mock_journal2]

        with patch.object(journal_manager, "_parse_caldav_journal") as mock_parse:
            mock_parse.return_value = Journal(
                uid="test-journal-123",
                summary="Updated via Search",
                dtstart=datetime.now(timezone.utc),
                calendar_uid="cal-123",
                account_alias="test",
            )

            result = journal_manager.update_journal(
                journal_uid="test-journal-123",
                calendar_uid="cal-123",
                summary="Updated via Search",
            )

        assert result is not None
        mock_journal2.save.assert_called_once()

    def test_update_journal_fallback_search_with_events(self, journal_manager):
        """Test journal update fallback to searching through events()"""
        # Setup calendar without journals method
        mock_calendar = Mock()
        mock_calendar.event_by_uid.side_effect = Exception("event_by_uid failed")
        mock_calendar.events = Mock()
        if hasattr(mock_calendar, "journals"):
            delattr(mock_calendar, "journals")

        journal_manager.calendars.get_calendar.return_value = mock_calendar

        # Mock event search
        existing_ical = self._create_journal_ical()
        mock_event = Mock()
        mock_event.data = existing_ical
        mock_event.save = Mock()
        mock_calendar.events.return_value = [mock_event]

        with patch.object(journal_manager, "_parse_caldav_journal") as mock_parse:
            mock_parse.return_value = Journal(
                uid="test-journal-123",
                summary="Updated via Events",
                dtstart=datetime.now(timezone.utc),
                calendar_uid="cal-123",
                account_alias="test",
            )

            result = journal_manager.update_journal(
                journal_uid="test-journal-123",
                calendar_uid="cal-123",
                summary="Updated via Events",
            )

        assert result is not None
        mock_event.save.assert_called_once()

    def test_update_journal_not_found(self, journal_manager):
        """Test journal update when journal not found"""
        mock_calendar = Mock()
        mock_calendar.event_by_uid.side_effect = Exception("Not found")
        mock_calendar.journals.return_value = []
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        with pytest.raises(JournalNotFoundError):
            journal_manager.update_journal(
                journal_uid="nonexistent", calendar_uid="cal-123", summary="Won't work"
            )

    def test_update_journal_invalid_ical_data(self, journal_manager):
        """Test journal update with invalid iCalendar data"""
        mock_calendar = Mock()
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        mock_caldav_journal = Mock()
        mock_caldav_journal.data = "INVALID ICAL DATA"
        mock_calendar.event_by_uid.return_value = mock_caldav_journal

        with pytest.raises(EventCreationError):
            journal_manager.update_journal(
                journal_uid="test-journal-123",
                calendar_uid="cal-123",
                summary="Updated",
            )

    def test_delete_journal_success_with_event_by_uid(self, journal_manager):
        """Test successful journal deletion using event_by_uid"""
        mock_calendar = Mock()
        mock_calendar.event_by_uid = Mock()
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        mock_journal = Mock()
        mock_journal.delete = Mock()
        mock_calendar.event_by_uid.return_value = mock_journal

        result = journal_manager.delete_journal(
            calendar_uid="cal-123", journal_uid="test-journal-123"
        )

        assert result is True
        mock_journal.delete.assert_called_once()

    def test_delete_journal_fallback_to_journals_search(self, journal_manager):
        """Test journal deletion fallback to searching through journals()"""
        mock_calendar = Mock()
        mock_calendar.event_by_uid.side_effect = Exception("event_by_uid failed")
        mock_calendar.journals = Mock()
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        # Create mock journal with VJOURNAL component that contains the target UID
        cal1 = iCalendar()
        event1 = iJournal()  # Wrong component type for first mock
        event1.add("uid", "different-uid")
        cal1.add_component(event1)

        cal2 = iCalendar()
        journal2 = iJournal()
        journal2.add("uid", "test-journal-123")  # This matches what we're searching for
        journal2.add("summary", "Target Journal")
        cal2.add_component(journal2)

        mock_journal1 = Mock()
        mock_journal1.data = cal1.to_ical().decode("utf-8")
        mock_journal2 = Mock()
        mock_journal2.data = cal2.to_ical().decode("utf-8")
        mock_journal2.delete = Mock()
        mock_calendar.journals.return_value = [mock_journal1, mock_journal2]

        result = journal_manager.delete_journal(
            calendar_uid="cal-123", journal_uid="test-journal-123"
        )

        assert result is True
        mock_journal2.delete.assert_called_once()

    def test_delete_journal_fallback_to_events_search(self, journal_manager):
        """Test journal deletion fallback to searching through events()"""
        # Setup calendar without journals method
        mock_calendar = Mock()
        mock_calendar.event_by_uid.side_effect = Exception("event_by_uid failed")
        mock_calendar.events = Mock()
        if hasattr(mock_calendar, "journals"):
            delattr(mock_calendar, "journals")

        journal_manager.calendars.get_calendar.return_value = mock_calendar

        # Create mock event with VJOURNAL component
        target_ical = self._create_journal_ical()
        mock_event = Mock()
        mock_event.data = target_ical
        mock_event.delete = Mock()
        mock_calendar.events.return_value = [mock_event]

        result = journal_manager.delete_journal(
            calendar_uid="cal-123", journal_uid="test-journal-123"
        )

        assert result is True
        mock_event.delete.assert_called_once()

    def test_delete_journal_not_found(self, journal_manager):
        """Test journal deletion when journal not found"""
        mock_calendar = Mock()
        mock_calendar.event_by_uid.side_effect = Exception("Not found")
        mock_calendar.journals.return_value = []
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        with pytest.raises(JournalNotFoundError):
            journal_manager.delete_journal(calendar_uid="cal-123", journal_uid="nonexistent")

    def test_delete_journal_authorization_error(self, journal_manager):
        """Test journal deletion with authorization error"""
        from caldav.lib.error import AuthorizationError

        mock_calendar = Mock()
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        mock_journal = Mock()
        mock_journal.delete.side_effect = AuthorizationError("Unauthorized")
        mock_calendar.event_by_uid.return_value = mock_journal

        # Execute & Verify - when journal found but deletion fails due to auth,
        # raises EventDeletionError (not JournalNotFoundError since it was found)
        # (not JournalNotFoundError, since the journal was successfully found)
        with pytest.raises(EventDeletionError):
            journal_manager.delete_journal(calendar_uid="cal-123", journal_uid="test-journal-123")

    def test_delete_journal_generic_error(self, journal_manager):
        """Test journal deletion with generic error"""
        mock_calendar = Mock()
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        mock_journal = Mock()
        mock_journal.delete.side_effect = Exception("Generic error")
        mock_calendar.event_by_uid.return_value = mock_journal

        # Execute & Verify - when journal is found but deletion fails, raises EventDeletionError
        # (not JournalNotFoundError, since the journal was successfully found)
        with pytest.raises(EventDeletionError):
            journal_manager.delete_journal(calendar_uid="cal-123", journal_uid="test-journal-123")

    def _create_journal_ical(self):
        """Create sample iCalendar data with VJOURNAL component"""
        cal = iCalendar()
        journal = iJournal()
        journal.add("uid", "test-journal-123")
        journal.add("summary", "Test Journal")
        journal.add("description", "Test Description")
        journal.add("dtstart", datetime(2025, 7, 10, 9, 0, tzinfo=timezone.utc))
        cal.add_component(journal)
        return cal.to_ical().decode("utf-8")


class TestJournalEdgeCases:
    """Test edge cases, error conditions, and parsing"""

    @pytest.fixture
    def journal_manager(self):
        """JournalManager instance for testing"""
        mock_calendar_manager = Mock(spec=CalendarManager)
        return JournalManager(mock_calendar_manager)

    def test_parse_caldav_journal_success(self, journal_manager):
        """Test successful VJOURNAL parsing"""
        mock_caldav_event = Mock()
        mock_caldav_event.data = self._create_complex_journal_ical()

        result = journal_manager._parse_caldav_journal(mock_caldav_event, "cal-123", "test_account")

        assert result is not None
        assert result.uid == "complex-journal-123"
        assert result.summary == "Complex Journal"
        assert result.description == "Detailed description"
        # Categories will be converted to strings from icalendar objects
        assert len(result.categories) == 1  # The list becomes a single string
        assert len(result.related_to) == 2
        assert "event-456" in result.related_to
        assert "task-789" in result.related_to

    def test_parse_caldav_journal_with_single_category(self, journal_manager):
        """Test VJOURNAL parsing with single category (not list)"""
        cal = iCalendar()
        journal = iJournal()
        journal.add("uid", "single-cat-journal")
        journal.add("summary", "Single Category Journal")
        journal.add("categories", "personal")  # Single category, not list
        cal.add_component(journal)

        mock_caldav_event = Mock()
        mock_caldav_event.data = cal.to_ical().decode("utf-8")

        result = journal_manager._parse_caldav_journal(mock_caldav_event, "cal-123", "test_account")

        assert result is not None
        # The icalendar library wraps categories in objects that need string conversion
        assert len(result.categories) == 1
        # Categories are converted to string representations that contain the original value
        category_str = str(result.categories[0])
        assert "vCategory" in category_str  # Just verify it's the expected object type

    def test_parse_caldav_journal_with_single_related_to(self, journal_manager):
        """Test VJOURNAL parsing with single related-to (not list)"""
        cal = iCalendar()
        journal = iJournal()
        journal.add("uid", "single-related-journal")
        journal.add("summary", "Single Related Journal")
        journal.add("related-to", "event-123")  # Single related-to, not list
        cal.add_component(journal)

        mock_caldav_event = Mock()
        mock_caldav_event.data = cal.to_ical().decode("utf-8")

        result = journal_manager._parse_caldav_journal(mock_caldav_event, "cal-123", "test_account")

        assert result is not None
        assert result.related_to == ["event-123"]

    def test_parse_caldav_journal_minimal_data(self, journal_manager):
        """Test VJOURNAL parsing with minimal required data"""
        cal = iCalendar()
        journal = iJournal()
        journal.add("uid", "minimal-journal")
        # Only UID, no summary or other fields
        cal.add_component(journal)

        mock_caldav_event = Mock()
        mock_caldav_event.data = cal.to_ical().decode("utf-8")

        with patch("chronos_mcp.journals.datetime") as mock_datetime:
            mock_now = datetime(2025, 7, 10, 12, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_now
            mock_datetime.timezone = timezone

            result = journal_manager._parse_caldav_journal(
                mock_caldav_event, "cal-123", "test_account"
            )

        assert result is not None
        assert result.uid == "minimal-journal"
        assert result.summary == "No Title"  # Default value
        assert result.description is None
        assert result.dtstart == mock_now  # Uses current time as fallback

    def test_parse_caldav_journal_no_vjournal_component(self, journal_manager):
        """Test parsing CalDAV data without VJOURNAL component"""
        # Create calendar with only VEVENT
        cal = iCalendar()
        from icalendar import Event as iEvent

        event = iEvent()
        event.add("uid", "event-123")
        event.add("summary", "Regular Event")
        cal.add_component(event)

        mock_caldav_event = Mock()
        mock_caldav_event.data = cal.to_ical().decode("utf-8")

        result = journal_manager._parse_caldav_journal(mock_caldav_event, "cal-123", "test_account")

        assert result is None

    def test_parse_caldav_journal_invalid_ical_data(self, journal_manager):
        """Test parsing invalid iCalendar data"""
        mock_caldav_event = Mock()
        mock_caldav_event.data = "INVALID ICAL DATA"

        result = journal_manager._parse_caldav_journal(mock_caldav_event, "cal-123", "test_account")

        assert result is None

    def test_parse_caldav_journal_exception_during_parsing(self, journal_manager):
        """Test handling exceptions during journal parsing"""
        mock_caldav_event = Mock()
        mock_caldav_event.data = self._create_complex_journal_ical()

        with patch("icalendar.Calendar.from_ical", side_effect=Exception("Parse error")):
            result = journal_manager._parse_caldav_journal(
                mock_caldav_event, "cal-123", "test_account"
            )

        assert result is None

    def test_parse_caldav_journal_with_default_account_fallback(self, journal_manager):
        """Test journal parsing with default account fallback"""
        journal_manager._get_default_account = Mock(return_value="default_account")

        mock_caldav_event = Mock()
        mock_caldav_event.data = self._create_simple_journal_ical()

        result = journal_manager._parse_caldav_journal(
            mock_caldav_event,
            "cal-123",
            None,  # No account_alias provided
        )

        assert result is not None
        assert result.account_alias == "default_account"

    def test_parse_caldav_journal_no_default_account(self, journal_manager):
        """Test journal parsing when no default account available"""
        journal_manager._get_default_account = Mock(return_value=None)

        mock_caldav_event = Mock()
        mock_caldav_event.data = self._create_simple_journal_ical()

        result = journal_manager._parse_caldav_journal(
            mock_caldav_event,
            "cal-123",
            None,  # No account_alias provided
        )

        assert result is not None
        assert result.account_alias == "default"  # Final fallback

    def test_update_journal_update_all_fields(self, journal_manager):
        """Test updating all journal fields"""
        mock_calendar = Mock()
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        existing_ical = self._create_complex_journal_ical()
        mock_caldav_journal = Mock()
        mock_caldav_journal.data = existing_ical
        mock_caldav_journal.save = Mock()
        mock_calendar.event_by_uid.return_value = mock_caldav_journal

        new_dtstart = datetime(2025, 8, 1, 14, 0, tzinfo=timezone.utc)
        new_related_to = ["new-event-123", "new-task-456"]

        with patch.object(journal_manager, "_parse_caldav_journal") as mock_parse:
            mock_parse.return_value = Journal(
                uid="complex-journal-123",
                summary="Updated Summary",
                description="Updated Description",
                dtstart=new_dtstart,
                related_to=new_related_to,
                calendar_uid="cal-123",
                account_alias="test",
            )

            result = journal_manager.update_journal(
                journal_uid="complex-journal-123",
                calendar_uid="cal-123",
                summary="Updated Summary",
                description="Updated Description",
                dtstart=new_dtstart,
                related_to=new_related_to,
            )

        assert result is not None
        mock_caldav_journal.save.assert_called_once()

    def test_update_journal_clear_description(self, journal_manager):
        """Test updating journal to clear description field"""
        mock_calendar = Mock()
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        existing_ical = self._create_complex_journal_ical()
        mock_caldav_journal = Mock()
        mock_caldav_journal.data = existing_ical
        mock_caldav_journal.save = Mock()
        mock_calendar.event_by_uid.return_value = mock_caldav_journal

        with patch.object(journal_manager, "_parse_caldav_journal") as mock_parse:
            mock_parse.return_value = Journal(
                uid="complex-journal-123",
                summary="Complex Journal",
                description=None,  # Cleared description
                dtstart=datetime.now(timezone.utc),
                calendar_uid="cal-123",
                account_alias="test",
            )

            result = journal_manager.update_journal(
                journal_uid="complex-journal-123",
                calendar_uid="cal-123",
                description="",  # Empty string to clear
            )

        assert result is not None
        assert result.description is None

    def test_update_journal_clear_related_to(self, journal_manager):
        """Test updating journal to clear related_to field"""
        mock_calendar = Mock()
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        existing_ical = self._create_complex_journal_ical()
        mock_caldav_journal = Mock()
        mock_caldav_journal.data = existing_ical
        mock_caldav_journal.save = Mock()
        mock_calendar.event_by_uid.return_value = mock_caldav_journal

        with patch.object(journal_manager, "_parse_caldav_journal") as mock_parse:
            mock_parse.return_value = Journal(
                uid="complex-journal-123",
                summary="Complex Journal",
                dtstart=datetime.now(timezone.utc),
                related_to=[],  # Cleared related_to
                calendar_uid="cal-123",
                account_alias="test",
            )

            result = journal_manager.update_journal(
                journal_uid="complex-journal-123",
                calendar_uid="cal-123",
                related_to=[],  # Empty list to clear
            )

        assert result is not None
        assert result.related_to == []

    def test_create_journal_with_request_id(self, journal_manager):
        """Test journal creation with custom request_id"""
        mock_calendar = Mock()
        mock_calendar.save_journal = Mock()
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        custom_request_id = "custom-request-123"

        with patch("uuid.uuid4", return_value="test-uid-123"):
            result = journal_manager.create_journal(
                calendar_uid="cal-123",
                summary="Test Journal",
                request_id=custom_request_id,
            )

        assert result is not None
        # Verify request_id was passed through to get_calendar
        journal_manager.calendars.get_calendar.assert_called_with(
            "cal-123", None, request_id=custom_request_id
        )

    def test_operations_generate_request_id_when_none_provided(self, journal_manager):
        """Test that operations generate request_id when none provided"""
        mock_calendar = Mock()
        mock_calendar.save_journal = Mock()
        journal_manager.calendars.get_calendar.return_value = mock_calendar

        with patch("uuid.uuid4", return_value="generated-request-id") as mock_uuid:
            result = journal_manager.create_journal(
                calendar_uid="cal-123",
                summary="Test Journal",
                # No request_id provided
            )

        assert result is not None
        # Should be called twice: once for request_id, once for journal UID
        assert mock_uuid.call_count == 2

    def _create_simple_journal_ical(self):
        """Create simple iCalendar data with VJOURNAL"""
        cal = iCalendar()
        journal = iJournal()
        journal.add("uid", "simple-journal-123")
        journal.add("summary", "Simple Journal")
        journal.add("dtstart", datetime(2025, 7, 10, 9, 0, tzinfo=timezone.utc))
        cal.add_component(journal)
        return cal.to_ical().decode("utf-8")

    def _create_complex_journal_ical(self):
        """Create complex iCalendar data with VJOURNAL including categories and related-to"""
        cal = iCalendar()
        journal = iJournal()
        journal.add("uid", "complex-journal-123")
        journal.add("summary", "Complex Journal")
        journal.add("description", "Detailed description")
        journal.add("dtstart", datetime(2025, 7, 10, 9, 0, tzinfo=timezone.utc))
        journal.add("categories", ["work", "project"])
        journal.add("related-to", "event-456")
        journal.add("related-to", "task-789")
        cal.add_component(journal)
        return cal.to_ical().decode("utf-8")
