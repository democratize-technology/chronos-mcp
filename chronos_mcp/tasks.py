"""
Task operations for Chronos MCP
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

import caldav
from caldav import Event as CalDAVEvent
from icalendar import Calendar as iCalendar
from icalendar import Todo as iTodo

from .caldav_utils import get_item_with_fallback
from .calendars import CalendarManager
from .exceptions import (
    CalendarNotFoundError,
    ChronosError,
    EventCreationError,
    EventDeletionError,
    TaskNotFoundError,
)
from .logging_config import setup_logging
from .models import Task, TaskStatus
from .utils import ical_to_datetime

logger = setup_logging()


class TaskManager:
    """Manage calendar tasks (VTODO)"""

    def __init__(self, calendar_manager: CalendarManager):
        self.calendars = calendar_manager

    def _get_default_account(self) -> Optional[str]:
        try:
            return self.calendars.accounts.config.config.default_account
        except Exception:
            return None

    def create_task(
        self,
        calendar_uid: str,
        summary: str,
        description: Optional[str] = None,
        due: Optional[datetime] = None,
        priority: Optional[int] = None,
        status: TaskStatus = TaskStatus.NEEDS_ACTION,
        related_to: Optional[List[str]] = None,
        account_alias: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[Task]:
        """Create a new task - raises exceptions on failure"""
        request_id = request_id or str(uuid.uuid4())

        calendar = self.calendars.get_calendar(
            calendar_uid, account_alias, request_id=request_id
        )
        if not calendar:
            raise CalendarNotFoundError(
                calendar_uid, account_alias, request_id=request_id
            )

        try:
            cal = iCalendar()
            task = iTodo()

            # Generate UID if not provided
            task_uid = str(uuid.uuid4())

            task.add("uid", task_uid)
            task.add("summary", summary)
            task.add("dtstamp", datetime.now(timezone.utc))

            if description:
                task.add("description", description)
            if due:
                task.add("due", due)
            if priority is not None and 1 <= priority <= 9:
                task.add("priority", priority)
            task.add("status", status.value)
            task.add("percent-complete", 0)

            if related_to:
                for related_uid in related_to:
                    task.add("related-to", related_uid)

            cal.add_component(task)

            # Save to CalDAV server using component-specific method when available
            ical_data = cal.to_ical().decode("utf-8")

            if hasattr(calendar, "save_todo"):
                logger.debug(
                    "Using calendar.save_todo() for optimized task creation",
                    extra={"request_id": request_id},
                )
                try:
                    caldav_task = calendar.save_todo(ical_data)
                except Exception as e:
                    logger.warning(
                        f"calendar.save_todo() failed: {e}, falling back to save_event()",
                        extra={"request_id": request_id},
                    )
                    caldav_task = calendar.save_event(ical_data)
            else:
                logger.debug(
                    "Server doesn't support calendar.save_todo(), using calendar.save_event()",
                    extra={"request_id": request_id},
                )
                caldav_task = calendar.save_event(ical_data)

            task_model = Task(
                uid=task_uid,
                summary=summary,
                description=description,
                due=due,
                priority=priority,
                status=status,
                percent_complete=0,
                related_to=related_to or [],
                calendar_uid=calendar_uid,
                account_alias=account_alias or self._get_default_account() or "default",
            )

            return task_model

        except caldav.lib.error.AuthorizationError as e:
            logger.error(
                f"Authorization error creating task '{summary}': {e}",
                extra={"request_id": request_id},
            )
            raise EventCreationError(
                summary, "Authorization failed", request_id=request_id
            )
        except Exception as e:
            logger.error(
                f"Error creating task '{summary}': {e}",
                extra={"request_id": request_id},
            )
            raise EventCreationError(summary, str(e), request_id=request_id)

    def get_task(
        self,
        task_uid: str,
        calendar_uid: str,
        account_alias: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[Task]:
        """Get a specific task by UID"""
        request_id = request_id or str(uuid.uuid4())

        calendar = self.calendars.get_calendar(
            calendar_uid, account_alias, request_id=request_id
        )
        if not calendar:
            raise CalendarNotFoundError(
                calendar_uid, account_alias, request_id=request_id
            )

        try:
            # Use utility function to find task with automatic fallback
            caldav_task = get_item_with_fallback(
                calendar, task_uid, "task", request_id=request_id
            )
            return self._parse_caldav_task(caldav_task, calendar_uid, account_alias)
        except ValueError:
            # get_item_with_fallback raises ValueError when not found
            raise TaskNotFoundError(task_uid, calendar_uid, request_id=request_id)

        except TaskNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error getting task '{task_uid}': {e}",
                extra={"request_id": request_id},
            )
            raise ChronosError(f"Failed to get task: {str(e)}", request_id=request_id)

    def list_tasks(
        self,
        calendar_uid: str,
        status_filter: Optional[TaskStatus] = None,
        account_alias: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> List[Task]:
        """List all tasks in a calendar"""
        request_id = request_id or str(uuid.uuid4())

        calendar = self.calendars.get_calendar(
            calendar_uid, account_alias, request_id=request_id
        )
        if not calendar:
            raise CalendarNotFoundError(
                calendar_uid, account_alias, request_id=request_id
            )

        tasks = []
        try:
            # Try component-specific method first for better performance
            if hasattr(calendar, "todos"):
                try:
                    logger.debug(
                        "Using calendar.todos() for server-side filtering",
                        extra={"request_id": request_id},
                    )
                    todos = calendar.todos()

                    for caldav_todo in todos:
                        task_data = self._parse_caldav_task(
                            caldav_todo, calendar_uid, account_alias
                        )
                        if task_data:
                            tasks.append(task_data)

                except Exception as e:
                    logger.warning(
                        f"calendar.todos() failed: {e}, falling back to calendar.events()",
                        extra={"request_id": request_id},
                    )
                    # Fall through to fallback method
                    raise
            else:
                # Fallback method for servers without todos() support
                logger.debug(
                    "Server doesn't support calendar.todos(), using calendar.events() with client-side filtering",
                    extra={"request_id": request_id},
                )
                events = calendar.events()

                for caldav_event in events:
                    task_data = self._parse_caldav_task(
                        caldav_event, calendar_uid, account_alias
                    )
                    if task_data:
                        tasks.append(task_data)

        except Exception as e:
            # If todos() method failed, try the fallback approach
            if hasattr(calendar, "todos"):
                try:
                    logger.info(
                        "Retrying with calendar.events() fallback method",
                        extra={"request_id": request_id},
                    )
                    events = calendar.events()

                    for caldav_event in events:
                        task_data = self._parse_caldav_task(
                            caldav_event, calendar_uid, account_alias
                        )
                        if task_data:
                            tasks.append(task_data)
                except Exception as fallback_error:
                    logger.error(
                        f"Error listing tasks (both methods failed): {fallback_error}",
                        extra={"request_id": request_id},
                    )
            else:
                logger.error(
                    f"Error listing tasks: {e}", extra={"request_id": request_id}
                )

        # Filter by status if requested
        if status_filter:
            tasks = [task for task in tasks if task.status == status_filter]
            logger.debug(
                f"Filtered tasks by status {status_filter.value}: {len(tasks)} tasks",
                extra={"request_id": request_id},
            )

        return tasks

    def update_task(
        self,
        task_uid: str,
        calendar_uid: str,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        due: Optional[datetime] = None,
        priority: Optional[int] = None,
        status: Optional[TaskStatus] = None,
        percent_complete: Optional[int] = None,
        related_to: Optional[List[str]] = None,
        account_alias: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[Task]:
        """Update an existing task - raises exceptions on failure"""
        request_id = request_id or str(uuid.uuid4())

        calendar = self.calendars.get_calendar(
            calendar_uid, account_alias, request_id=request_id
        )
        if not calendar:
            raise CalendarNotFoundError(
                calendar_uid, account_alias, request_id=request_id
            )

        try:
            # Use utility function to find task with automatic fallback
            try:
                caldav_task = get_item_with_fallback(
                    calendar, task_uid, "task", request_id=request_id
                )
            except ValueError:
                raise TaskNotFoundError(task_uid, calendar_uid, request_id=request_id)

            # Parse existing task data
            ical = iCalendar.from_ical(caldav_task.data)
            existing_task = None

            for component in ical.walk():
                if component.name == "VTODO":
                    existing_task = component
                    break

            if not existing_task:
                raise EventCreationError(
                    f"Task {task_uid}",
                    "Could not parse existing task data",
                    request_id=request_id,
                )

            # Update only provided fields
            if summary is not None:
                existing_task["SUMMARY"] = summary

            if description is not None:
                if description:
                    existing_task["DESCRIPTION"] = description
                elif "DESCRIPTION" in existing_task:
                    del existing_task["DESCRIPTION"]

            if due is not None:
                if "DUE" in existing_task:
                    del existing_task["DUE"]
                if due:
                    existing_task.add("DUE", due)

            if priority is not None:
                if priority and 1 <= priority <= 9:
                    existing_task["PRIORITY"] = priority
                elif "PRIORITY" in existing_task:
                    del existing_task["PRIORITY"]

            if status is not None:
                existing_task["STATUS"] = status.value

            if percent_complete is not None:
                if 0 <= percent_complete <= 100:
                    existing_task["PERCENT-COMPLETE"] = percent_complete

            # Handle RELATED-TO property updates
            if related_to is not None:
                # Remove all existing RELATED-TO properties
                if "RELATED-TO" in existing_task:
                    del existing_task["RELATED-TO"]

                # Add new RELATED-TO properties if provided
                if related_to:
                    for related_uid in related_to:
                        existing_task.add("RELATED-TO", related_uid)

            # Update last-modified timestamp
            if "LAST-MODIFIED" in existing_task:
                del existing_task["LAST-MODIFIED"]
            existing_task.add("LAST-MODIFIED", datetime.now(timezone.utc).replace(microsecond=0))

            # Save the updated task
            caldav_task.data = ical.to_ical().decode("utf-8")
            caldav_task.save()

            # Parse and return the updated task
            return self._parse_caldav_task(caldav_task, calendar_uid, account_alias)

        except TaskNotFoundError:
            raise
        except EventCreationError:
            raise
        except Exception as e:
            logger.error(
                f"Error updating task '{task_uid}': {e}",
                extra={"request_id": request_id},
            )
            raise EventCreationError(
                task_uid, f"Failed to update task: {str(e)}", request_id=request_id
            )

    def delete_task(
        self,
        calendar_uid: str,
        task_uid: str,
        account_alias: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> bool:
        """Delete a task by UID - raises exceptions on failure"""
        request_id = request_id or str(uuid.uuid4())

        calendar = self.calendars.get_calendar(
            calendar_uid, account_alias, request_id=request_id
        )
        if not calendar:
            raise CalendarNotFoundError(
                calendar_uid, account_alias, request_id=request_id
            )

        try:
            # Use utility function to find task with automatic fallback
            task = get_item_with_fallback(
                calendar, task_uid, "task", request_id=request_id
            )
            task.delete()
            logger.info(
                f"Deleted task '{task_uid}'",
                extra={"request_id": request_id},
            )
            return True
        except ValueError:
            # get_item_with_fallback raises ValueError when not found
            raise TaskNotFoundError(task_uid, calendar_uid, request_id=request_id)

        except TaskNotFoundError:
            raise
        except caldav.lib.error.AuthorizationError as e:
            logger.error(
                f"Authorization error deleting task '{task_uid}': {e}",
                extra={"request_id": request_id},
            )
            raise EventDeletionError(
                task_uid, "Authorization failed", request_id=request_id
            )
        except Exception as e:
            logger.error(
                f"Error deleting task '{task_uid}': {e}",
                extra={"request_id": request_id},
            )
            raise EventDeletionError(task_uid, str(e), request_id=request_id)

    def _parse_caldav_task(
        self, caldav_event: CalDAVEvent, calendar_uid: str, account_alias: Optional[str]
    ) -> Optional[Task]:
        """Parse CalDAV VTODO to Task model"""
        try:
            # Parse iCalendar data
            ical = iCalendar.from_ical(caldav_event.data)

            for component in ical.walk():
                if component.name == "VTODO":
                    # Parse date/time values
                    due_dt = None
                    completed_dt = None

                    if component.get("due"):
                        due_dt = ical_to_datetime(component.get("due"))
                    if component.get("completed"):
                        completed_dt = ical_to_datetime(component.get("completed"))

                    # Parse priority
                    priority = None
                    if component.get("priority"):
                        try:
                            priority = int(component.get("priority"))
                        except (ValueError, TypeError):
                            priority = None

                    # Parse percent complete
                    percent_complete = 0
                    if component.get("percent-complete"):
                        try:
                            percent_complete = int(component.get("percent-complete"))
                        except (ValueError, TypeError):
                            percent_complete = 0

                    # Parse status
                    status = TaskStatus.NEEDS_ACTION
                    if component.get("status"):
                        try:
                            status = TaskStatus(str(component.get("status")))
                        except ValueError:
                            status = TaskStatus.NEEDS_ACTION

                    # Parse RELATED-TO properties
                    related_to = []
                    if component.get("related-to"):
                        related_prop = component.get("related-to")
                        if isinstance(related_prop, list):
                            related_to = [str(r) for r in related_prop]
                        else:
                            related_to = [str(related_prop)]

                    # Parse basic task data
                    task = Task(
                        uid=str(component.get("uid", "")),
                        summary=str(component.get("summary", "No Title")),
                        description=(
                            str(component.get("description", ""))
                            if component.get("description")
                            else None
                        ),
                        due=due_dt,
                        completed=completed_dt,
                        priority=priority,
                        status=status,
                        percent_complete=percent_complete,
                        related_to=related_to,
                        calendar_uid=calendar_uid,
                        account_alias=account_alias
                        or self._get_default_account()
                        or "default",
                    )

                    return task

        except Exception as e:
            logger.error(f"Error parsing task: {e}")

        return None
