"""
Journal operations for Chronos MCP
"""

import sys
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from icalendar import Journal as iJournal, Calendar as iCalendar
from caldav import Event as CalDAVEvent
import caldav

from .models import Journal
from .calendars import CalendarManager
from .utils import parse_datetime, datetime_to_ical, ical_to_datetime
from .logging_config import setup_logging
from .exceptions import (
    CalendarNotFoundError,
    EventNotFoundError,
    EventCreationError,
    EventDeletionError,
    CalDAVError,
    ChronosError,
    ErrorHandler
)

# Set up logging
logger = setup_logging()


class JournalManager:
    """Manage calendar journals (VJOURNAL)"""
    
    def __init__(self, calendar_manager: CalendarManager):
        self.calendars = calendar_manager
    
    def _get_default_account(self) -> Optional[str]:
        """Get default account alias"""
        try:
            return self.calendars.accounts.config.config.default_account
        except Exception:
            return None

    def create_journal(self, 
                      calendar_uid: str,
                      summary: str,
                      description: Optional[str] = None,
                      dtstart: Optional[datetime] = None,
                      related_to: Optional[List[str]] = None,
                      account_alias: Optional[str] = None,
                      request_id: Optional[str] = None) -> Optional[Journal]:
        """Create a new journal entry - raises exceptions on failure"""
        request_id = request_id or str(uuid.uuid4())
        
        calendar = self.calendars.get_calendar(calendar_uid, account_alias, request_id=request_id)
        if not calendar:
            raise CalendarNotFoundError(calendar_uid, account_alias, request_id=request_id)
            
        try:
            # Use current time if dtstart not provided
            if dtstart is None:
                dtstart = datetime.now(timezone.utc)
            
            # Create iCalendar journal
            cal = iCalendar()
            journal = iJournal()
            
            # Generate UID if not provided
            journal_uid = str(uuid.uuid4())
            
            # Set required properties
            journal.add('uid', journal_uid)
            journal.add('summary', summary)
            journal.add('dtstart', dtstart)
            journal.add('dtstamp', datetime.now(timezone.utc))
            
            # Set optional properties
            if description:
                journal.add('description', description)
            
            # Add RELATED-TO properties
            if related_to:
                for related_uid in related_to:
                    journal.add('related-to', related_uid)
                
            # Add journal to calendar
            cal.add_component(journal)
            
            # Save to CalDAV server using component-specific method when available
            ical_data = cal.to_ical().decode('utf-8')
            
            if hasattr(calendar, 'save_journal'):
                logger.debug(f"Using calendar.save_journal() for optimized journal creation", extra={"request_id": request_id})
                try:
                    caldav_journal = calendar.save_journal(ical_data)
                except Exception as e:
                    logger.warning(f"calendar.save_journal() failed: {e}, falling back to save_event()", extra={"request_id": request_id})
                    caldav_journal = calendar.save_event(ical_data)
            else:
                logger.debug(f"Server doesn't support calendar.save_journal(), using calendar.save_event()", extra={"request_id": request_id})
                caldav_journal = calendar.save_event(ical_data)
            
            # Return Journal model
            journal_model = Journal(
                uid=journal_uid,
                summary=summary,
                description=description,
                dtstart=dtstart,
                related_to=related_to or [],
                calendar_uid=calendar_uid,
                account_alias=account_alias or self._get_default_account() or "default"
            )
            
            return journal_model
            
        except caldav.lib.error.AuthorizationError as e:
            logger.error(f"Authorization error creating journal '{summary}': {e}", extra={"request_id": request_id})
            raise EventCreationError(summary, "Authorization failed", request_id=request_id)
        except Exception as e:
            logger.error(f"Error creating journal '{summary}': {e}", extra={"request_id": request_id})
            raise EventCreationError(summary, str(e), request_id=request_id)

    def get_journal(self, 
                   journal_uid: str, 
                   calendar_uid: str, 
                   account_alias: Optional[str] = None,
                   request_id: Optional[str] = None) -> Optional[Journal]:
        """Get a specific journal by UID"""
        request_id = request_id or str(uuid.uuid4())
        
        calendar = self.calendars.get_calendar(calendar_uid, account_alias, request_id=request_id)
        if not calendar:
            raise CalendarNotFoundError(calendar_uid, account_alias, request_id=request_id)
            
        try:
            # Method 1: Try event_by_uid if available
            if hasattr(calendar, 'event_by_uid'):
                try:
                    caldav_journal = calendar.event_by_uid(journal_uid)
                    return self._parse_caldav_journal(caldav_journal, calendar_uid, account_alias)
                except Exception as e:
                    logger.warning(f"event_by_uid failed: {e}, trying fallback method")
            
            # Method 2: Fallback - search through all journals
            try:
                if hasattr(calendar, 'journals'):
                    journals = calendar.journals()
                else:
                    # If journals() not available, use events() and filter
                    journals = calendar.events()
                    
                for journal in journals:
                    if journal_uid in journal.data:
                        journal_data = self._parse_caldav_journal(journal, calendar_uid, account_alias)
                        if journal_data and journal_data.uid == journal_uid:
                            return journal_data
            except Exception as e:
                logger.warning(f"Fallback search failed: {e}", extra={"request_id": request_id})
            
            # Journal not found
            raise EventNotFoundError(journal_uid, calendar_uid, request_id=request_id)
                
        except EventNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting journal '{journal_uid}': {e}", extra={"request_id": request_id})
            raise ChronosError(f"Failed to get journal: {str(e)}", request_id=request_id)

    def list_journals(self, 
                     calendar_uid: str,
                     account_alias: Optional[str] = None,
                     request_id: Optional[str] = None) -> List[Journal]:
        """List all journals in a calendar"""
        request_id = request_id or str(uuid.uuid4())
        
        calendar = self.calendars.get_calendar(calendar_uid, account_alias, request_id=request_id)
        if not calendar:
            raise CalendarNotFoundError(calendar_uid, account_alias, request_id=request_id)
            
        journals = []
        try:
            # Try component-specific method first for better performance
            if hasattr(calendar, 'journals'):
                try:
                    logger.debug(f"Using calendar.journals() for server-side filtering", extra={"request_id": request_id})
                    journal_objects = calendar.journals()
                    
                    for caldav_journal in journal_objects:
                        journal_data = self._parse_caldav_journal(caldav_journal, calendar_uid, account_alias)
                        if journal_data:
                            journals.append(journal_data)
                            
                except Exception as e:
                    logger.warning(f"calendar.journals() failed: {e}, falling back to calendar.events()", extra={"request_id": request_id})
                    # Fall through to fallback method
                    raise
            else:
                # Fallback method for servers without journals() support
                logger.debug(f"Server doesn't support calendar.journals(), using calendar.events() with client-side filtering", extra={"request_id": request_id})
                events = calendar.events()
                
                for caldav_event in events:
                    journal_data = self._parse_caldav_journal(caldav_event, calendar_uid, account_alias)
                    if journal_data:
                        journals.append(journal_data)
                    
        except Exception as e:
            # If journals() method failed, try the fallback approach
            if hasattr(calendar, 'journals'):
                try:
                    logger.info(f"Retrying with calendar.events() fallback method", extra={"request_id": request_id})
                    events = calendar.events()
                    
                    for caldav_event in events:
                        journal_data = self._parse_caldav_journal(caldav_event, calendar_uid, account_alias)
                        if journal_data:
                            journals.append(journal_data)
                except Exception as fallback_error:
                    logger.error(f"Error listing journals (both methods failed): {fallback_error}", extra={"request_id": request_id})
            else:
                logger.error(f"Error listing journals: {e}", extra={"request_id": request_id})
            
        return journals

    def update_journal(self,
                      journal_uid: str,
                      calendar_uid: str,
                      summary: Optional[str] = None,
                      description: Optional[str] = None,
                      dtstart: Optional[datetime] = None,
                      related_to: Optional[List[str]] = None,
                      account_alias: Optional[str] = None,
                      request_id: Optional[str] = None) -> Optional[Journal]:
        """Update an existing journal - raises exceptions on failure"""
        request_id = request_id or str(uuid.uuid4())
        
        calendar = self.calendars.get_calendar(calendar_uid, account_alias, request_id=request_id)
        if not calendar:
            raise CalendarNotFoundError(calendar_uid, account_alias, request_id=request_id)
            
        try:
            # Find the existing journal
            caldav_journal = None
            
            # Method 1: Try event_by_uid if available
            if hasattr(calendar, 'event_by_uid'):
                try:
                    caldav_journal = calendar.event_by_uid(journal_uid)
                except Exception as e:
                    logger.warning(f"event_by_uid failed for update: {e}, trying fallback")
                    
            # Method 2: Fallback - search through all journals
            if not caldav_journal:
                try:
                    if hasattr(calendar, 'journals'):
                        journals = calendar.journals()
                    else:
                        # If journals() not available, use events() and filter
                        journals = calendar.events()
                        
                    for journal in journals:
                        if journal_uid in journal.data:
                            caldav_journal = journal
                            break
                except Exception as e:
                    logger.warning(f"Fallback search in update failed: {e}", extra={"request_id": request_id})
                        
            if not caldav_journal:
                raise EventNotFoundError(journal_uid, calendar_uid, request_id=request_id)
                
            # Parse existing journal data
            ical = iCalendar.from_ical(caldav_journal.data)
            existing_journal = None
            
            for component in ical.walk():
                if component.name == "VJOURNAL":
                    existing_journal = component
                    break
                    
            if not existing_journal:
                raise EventCreationError(
                    f"Journal {journal_uid}",
                    "Could not parse existing journal data",
                    request_id=request_id
                )
                
            # Update only provided fields
            if summary is not None:
                existing_journal['SUMMARY'] = summary
                
            if description is not None:
                if description:
                    existing_journal['DESCRIPTION'] = description
                elif 'DESCRIPTION' in existing_journal:
                    del existing_journal['DESCRIPTION']
                    
            if dtstart is not None:
                if 'DTSTART' in existing_journal:
                    del existing_journal['DTSTART']
                existing_journal.add('DTSTART', dtstart)
                    
            # Handle RELATED-TO property updates
            if related_to is not None:
                # Remove all existing RELATED-TO properties
                if 'RELATED-TO' in existing_journal:
                    del existing_journal['RELATED-TO']
                
                # Add new RELATED-TO properties if provided
                if related_to:
                    for related_uid in related_to:
                        existing_journal.add('RELATED-TO', related_uid)
                    
            # Update last-modified timestamp
            if 'LAST-MODIFIED' in existing_journal:
                del existing_journal['LAST-MODIFIED']
            existing_journal.add('LAST-MODIFIED', datetime.now(timezone.utc))
            
            # Save the updated journal
            caldav_journal.data = ical.to_ical().decode('utf-8')
            caldav_journal.save()
            
            # Parse and return the updated journal
            return self._parse_caldav_journal(caldav_journal, calendar_uid, account_alias)
            
        except EventNotFoundError:
            raise
        except EventCreationError:
            raise
        except Exception as e:
            logger.error(f"Error updating journal '{journal_uid}': {e}", extra={"request_id": request_id})
            raise EventCreationError(
                journal_uid,
                f"Failed to update journal: {str(e)}",
                request_id=request_id
            )

    def delete_journal(self, 
                      calendar_uid: str, 
                      journal_uid: str, 
                      account_alias: Optional[str] = None,
                      request_id: Optional[str] = None) -> bool:
        """Delete a journal by UID - raises exceptions on failure"""
        request_id = request_id or str(uuid.uuid4())
        
        calendar = self.calendars.get_calendar(calendar_uid, account_alias, request_id=request_id)
        if not calendar:
            raise CalendarNotFoundError(calendar_uid, account_alias, request_id=request_id)
            
        try:
            # Method 1: Try event_by_uid if available
            if hasattr(calendar, 'event_by_uid'):
                try:
                    journal = calendar.event_by_uid(journal_uid)
                    journal.delete()
                    logger.info(f"Deleted journal '{journal_uid}' using event_by_uid")
                    return True
                except Exception as e:
                    logger.warning(f"event_by_uid failed: {e}, trying fallback method")
            
            # Method 2: Fallback - get all journals and filter
            try:
                if hasattr(calendar, 'journals'):
                    journals = calendar.journals()
                else:
                    # If journals() not available, use events() and filter
                    journals = calendar.events()
                    
                for journal in journals:
                    # Parse the journal to check UID and type
                    ical = iCalendar.from_ical(journal.data)
                    for component in ical.walk():
                        if component.name == "VJOURNAL":
                            if str(component.get('uid', '')) == journal_uid:
                                journal.delete()
                                logger.info(f"Deleted journal '{journal_uid}'", extra={"request_id": request_id})
                                return True
            except Exception as e:
                logger.warning(f"Fallback delete failed: {e}", extra={"request_id": request_id})
            
            # Journal not found
            raise EventNotFoundError(journal_uid, calendar_uid, request_id=request_id)
                
        except EventNotFoundError:
            raise
        except caldav.lib.error.AuthorizationError as e:
            logger.error(f"Authorization error deleting journal '{journal_uid}': {e}", extra={"request_id": request_id})
            raise EventDeletionError(journal_uid, "Authorization failed", request_id=request_id)
        except Exception as e:
            logger.error(f"Error deleting journal '{journal_uid}': {e}", extra={"request_id": request_id})
            raise EventDeletionError(journal_uid, str(e), request_id=request_id)

    def _parse_caldav_journal(self, caldav_event: CalDAVEvent, calendar_uid: str, account_alias: Optional[str]) -> Optional[Journal]:
        """Parse CalDAV VJOURNAL to Journal model"""
        try:
            # Parse iCalendar data
            ical = iCalendar.from_ical(caldav_event.data)
            
            for component in ical.walk():
                # Debug logging to see what components we find
                logger.debug(f"Found component: {component.name}")
                if component.name == "VJOURNAL":
                    # Parse date/time values
                    dtstart_dt = None
                    if component.get('dtstart'):
                        dtstart_dt = ical_to_datetime(component.get('dtstart'))
                    
                    # Parse categories
                    categories = []
                    if component.get('categories'):
                        cat_value = component.get('categories')
                        if isinstance(cat_value, list):
                            categories = [str(cat) for cat in cat_value]
                        else:
                            categories = [str(cat_value)]
                    
                    # Parse RELATED-TO properties
                    related_to = []
                    if component.get('related-to'):
                        related_prop = component.get('related-to')
                        if isinstance(related_prop, list):
                            related_to = [str(r) for r in related_prop]
                        else:
                            related_to = [str(related_prop)]
                    
                    # Parse basic journal data
                    journal = Journal(
                        uid=str(component.get('uid', '')),
                        summary=str(component.get('summary', 'No Title')),
                        description=str(component.get('description', '')) if component.get('description') else None,
                        dtstart=dtstart_dt or datetime.now(timezone.utc),
                        categories=categories,
                        related_to=related_to,
                        calendar_uid=calendar_uid,
                        account_alias=account_alias or self._get_default_account() or "default"
                    )
                    
                    return journal
                    
        except Exception as e:
            logger.error(f"Error parsing journal: {e}")
            
        return None