"""
Calendar operations for Chronos MCP
"""

import uuid
from typing import List, Optional

import caldav
from caldav import Calendar as CalDAVCalendar

from .accounts import AccountManager
from .exceptions import (
    AccountNotFoundError,
    CalendarCreationError,
    CalendarDeletionError,
    CalendarNotFoundError,
    ErrorHandler,
)
from .logging_config import setup_logging
from .models import Calendar

logger = setup_logging()


class CalendarManager:
    """Manage calendar operations"""

    def __init__(self, account_manager: AccountManager):
        self.accounts = account_manager

    def list_calendars(
        self, account_alias: Optional[str] = None, request_id: Optional[str] = None
    ) -> List[Calendar]:
        """List all calendars for an account - raises exceptions on failure"""
        request_id = request_id or str(uuid.uuid4())

        principal = self.accounts.get_principal(account_alias)
        if not principal:
            raise AccountNotFoundError(
                account_alias
                or self.accounts.config.config.default_account
                or "default",
                request_id=request_id,
            )

        calendars = []
        try:
            for cal in principal.calendars():
                # Extract calendar properties
                cal_info = Calendar(
                    uid=(
                        str(cal.url).split("/")[-2]
                        if str(cal.url).endswith("/")
                        else str(cal.url).split("/")[-1]
                    ),
                    name=cal.name or "Unnamed Calendar",
                    description=None,  # Will need to fetch from properties
                    color=None,  # Will need to fetch from properties
                    account_alias=account_alias
                    or self.accounts.config.config.default_account,
                    url=str(cal.url),
                    read_only=False,  # Will need to check permissions
                )
                calendars.append(cal_info)

        except Exception as e:
            logger.error(f"Error listing calendars: {e}")

        return calendars

    def create_calendar(
        self,
        name: str,
        description: Optional[str] = None,
        color: Optional[str] = None,
        account_alias: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[Calendar]:
        """Create a new calendar - raises exceptions on failure"""
        request_id = request_id or str(uuid.uuid4())

        principal = self.accounts.get_principal(account_alias)
        if not principal:
            raise AccountNotFoundError(
                account_alias
                or self.accounts.config.config.default_account
                or "default",
                request_id=request_id,
            )

        try:
            cal_id = name.lower().replace(" ", "_")
            cal = principal.make_calendar(name=name, cal_id=cal_id)

            # Note: description and color properties would need CalDAV server support
            # for setting calendar properties beyond name

            return Calendar(
                uid=cal_id,
                name=name,
                description=description,
                color=color,
                account_alias=account_alias
                or self.accounts.config.config.default_account,
                url=str(cal.url),
                read_only=False,
            )

        except caldav.lib.error.AuthorizationError as e:
            logger.error(
                f"Authorization error creating calendar '{name}': {e}",
                extra={"request_id": request_id},
            )
            raise CalendarCreationError(
                name, "Authorization failed", request_id=request_id
            )
        except Exception as e:
            logger.error(
                f"Error creating calendar '{name}': {e}",
                extra={"request_id": request_id},
            )
            raise CalendarCreationError(name, str(e), request_id=request_id)

    def delete_calendar(
        self,
        calendar_uid: str,
        account_alias: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> bool:
        """Delete a calendar - raises exceptions on failure"""
        request_id = request_id or str(uuid.uuid4())

        principal = self.accounts.get_principal(account_alias)
        if not principal:
            raise AccountNotFoundError(
                account_alias
                or self.accounts.config.config.default_account
                or "default",
                request_id=request_id,
            )

        try:
            # Find calendar by UID
            for cal in principal.calendars():
                cal_id = (
                    str(cal.url).split("/")[-2]
                    if str(cal.url).endswith("/")
                    else str(cal.url).split("/")[-1]
                )
                if cal_id == calendar_uid:
                    cal.delete()
                    logger.info(
                        f"Deleted calendar '{calendar_uid}'",
                        extra={"request_id": request_id},
                    )
                    return True

            # Calendar not found
            raise CalendarNotFoundError(
                calendar_uid, account_alias, request_id=request_id
            )

        except CalendarNotFoundError:
            raise  # Re-raise our own exception
        except caldav.lib.error.AuthorizationError as e:
            logger.error(
                f"Authorization error deleting calendar '{calendar_uid}': {e}",
                extra={"request_id": request_id},
            )
            raise CalendarDeletionError(
                calendar_uid, "Authorization failed", request_id=request_id
            )
        except Exception as e:
            logger.error(
                f"Error deleting calendar '{calendar_uid}': {e}",
                extra={"request_id": request_id},
            )
            raise CalendarDeletionError(calendar_uid, str(e), request_id=request_id)

    @ErrorHandler.safe_operation(logger, default_return=None)
    def get_calendar(
        self,
        calendar_uid: str,
        account_alias: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[CalDAVCalendar]:
        """Get CalDAV calendar object by UID - internal utility method"""
        principal = self.accounts.get_principal(account_alias)
        if not principal:
            return None

        try:
            for cal in principal.calendars():
                cal_id = (
                    str(cal.url).split("/")[-2]
                    if str(cal.url).endswith("/")
                    else str(cal.url).split("/")[-1]
                )
                if cal_id == calendar_uid:
                    return cal
        except Exception as e:
            logger.error(f"Error getting calendar: {e}")

        return None
