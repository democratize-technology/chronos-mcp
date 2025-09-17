"""
Test configuration and fixtures for Chronos MCP
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import pytz

from chronos_mcp.config import ConfigManager
from chronos_mcp.models import Account, Calendar, Event
from chronos_mcp.rate_limiter import disable_rate_limiting


@pytest.fixture(autouse=True)
def disable_rate_limiting_for_tests():
    """Automatically disable rate limiting for all tests"""
    disable_rate_limiting()
    yield


@pytest.fixture
def temp_config_dir():
    """Create temporary config directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config_manager(temp_config_dir):
    """Mock ConfigManager with temp directory"""
    with patch("chronos_mcp.config.Path.home") as mock_home:
        mock_home.return_value = temp_config_dir
        config_mgr = ConfigManager()
        yield config_mgr


@pytest.fixture
def sample_account():
    """Sample account for testing"""
    return Account(
        alias="test_account",
        url="https://caldav.example.com",
        username="testuser",
        password="testpass",
        display_name="Test Account",
    )


@pytest.fixture
def sample_calendar():
    """Sample calendar for testing"""
    return Calendar(
        uid="cal-123",
        name="Test Calendar",
        description="A test calendar",
        color="#FF0000",
        account_alias="test_account",
        url="https://caldav.example.com/calendars/test",
    )


@pytest.fixture
def sample_event():
    """Sample event for testing"""
    return Event(
        uid="evt-123",
        summary="Test Event",
        description="Test Description",
        location="Test Location",
        start=datetime(2025, 7, 5, 10, 0, tzinfo=pytz.UTC),
        end=datetime(2025, 7, 5, 11, 0, tzinfo=pytz.UTC),
        all_day=False,
        calendar_uid="cal-123",
        account_alias="test_account",
    )


@pytest.fixture
def mock_caldav_client():
    """Mock CalDAV client"""
    with patch("caldav.DAVClient") as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance

        # Mock principal
        mock_principal = Mock()
        mock_instance.principal.return_value = mock_principal

        # Mock calendars
        mock_calendar = Mock()
        mock_calendar.name = "Test Calendar"
        mock_calendar.url = "https://caldav.example.com/cal"
        mock_principal.calendars.return_value = [mock_calendar]

        yield mock_instance
