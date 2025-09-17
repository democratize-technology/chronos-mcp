"""
Chronos MCP Server - Advanced CalDAV Management
"""

from fastmcp import FastMCP

from .accounts import AccountManager
from .bulk import BulkOperationManager
from .calendars import CalendarManager
from .config import ConfigManager
from .events import EventManager
from .journals import JournalManager
from .logging_config import setup_logging
from .tasks import TaskManager
from .tools import register_all_tools

logger = setup_logging()

mcp = FastMCP("chronos-mcp")

logger.info("Initializing Chronos MCP Server...")

try:
    config_manager = ConfigManager()
    account_manager = AccountManager(config_manager)
    calendar_manager = CalendarManager(account_manager)
    event_manager = EventManager(calendar_manager)
    task_manager = TaskManager(calendar_manager)
    journal_manager = JournalManager(calendar_manager)
    bulk_manager = BulkOperationManager(
        event_manager=event_manager,
        task_manager=task_manager,
        journal_manager=journal_manager,
    )
    logger.info("All managers initialized successfully")

    managers = {
        "config_manager": config_manager,
        "account_manager": account_manager,
        "calendar_manager": calendar_manager,
        "event_manager": event_manager,
        "task_manager": task_manager,
        "journal_manager": journal_manager,
        "bulk_manager": bulk_manager,
    }

    register_all_tools(mcp, managers)
    logger.info("All tools registered successfully")

except Exception as e:
    logger.error(f"Error initializing Chronos MCP Server: {e}")
    raise


# Export all tools for backwards compatibility
# This allows tests and existing code to import from server.py
from .tools.accounts import add_account, list_accounts, remove_account, test_account
from .tools.bulk import (
    bulk_create_events,
    bulk_create_journals,
    bulk_create_tasks,
    bulk_delete_events,
    bulk_delete_journals,
    bulk_delete_tasks,
)
from .tools.calendars import create_calendar, delete_calendar, list_calendars
from .tools.events import (
    create_event,
    create_recurring_event,
    delete_event,
    get_events_range,
    search_events,
    update_event,
)
from .tools.journals import (
    create_journal,
    delete_journal,
    list_journals,
    update_journal,
)
from .tools.tasks import create_task, delete_task, list_tasks, update_task

__all__ = [
    # Account tools
    "add_account",
    "list_accounts",
    "remove_account",
    "test_account",
    # Calendar tools
    "list_calendars",
    "create_calendar",
    "delete_calendar",
    # Event tools
    "create_event",
    "get_events_range",
    "delete_event",
    "update_event",
    "create_recurring_event",
    "search_events",
    # Task tools
    "create_task",
    "list_tasks",
    "update_task",
    "delete_task",
    # Journal tools
    "create_journal",
    "list_journals",
    "update_journal",
    "delete_journal",
    # Bulk tools
    "bulk_create_events",
    "bulk_delete_events",
    "bulk_create_tasks",
    "bulk_delete_tasks",
    "bulk_create_journals",
    "bulk_delete_journals",
]

def main():
    """Main function to run the MCP server."""
    logger.info("Initializing Chronos MCP Server...")

    try:
        logger.info("All managers initialized successfully")
        logger.info("All tools registered successfully")

        # Run the server - this blocks and keeps the server alive
        logger.info("Starting MCP server...")
        mcp.run()

    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

# Main entry point for running the server
if __name__ == "__main__":
    main()
