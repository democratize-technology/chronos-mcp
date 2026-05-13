# Chronos MCP - Advanced CalDAV Management Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP 2.0+](https://img.shields.io/badge/FastMCP-2.0+-green.svg)](https://github.com/jlowin/fastmcp)
[![CalDAV](https://img.shields.io/badge/CalDAV-RFC4791-orange.svg)](https://tools.ietf.org/html/rfc4791)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Model Context Protocol (MCP) server for CalDAV calendar management, built with FastMCP 2.0. Chronos provides advanced calendar and event management capabilities with multi-account support.

## 🚀 Features

- **Multi-account Support**: Manage multiple CalDAV servers simultaneously
- **Full CRUD Operations**: Create, read, update, and delete calendars and events ✅
- **Advanced Event Management**: 
  - Recurring events with RRULE support ✅
  - Event updates with partial field modifications ✅
  - Attendee management and invitations (JSON format) ✅
  - Reminders and alarms (VALARM) ✅
  - Timezone-aware operations
- **Advanced Search**: 
  - Full-text search across events ✅
  - Field-specific search (title, description, location) ✅
  - Multiple match types (contains, starts_with, exact, regex) ✅
  - Date range filtering ✅
  - Relevance ranking algorithm ✅
- **Bulk Operations**:
  - Create multiple events in parallel ✅
  - Delete multiple events efficiently ✅
  - Atomic operations with rollback ✅
  - Configurable error handling modes ✅
- **Security Hardening**:
  - Comprehensive input validation ✅
  - XSS and injection prevention ✅
  - Path traversal protection ✅
  - RFC-compliant validation ✅
- **Task Management**: Full VTODO support ✅
- **Journal Entries**: Full VJOURNAL support ✅
- **Import/Export**: iCalendar format support (coming soon)
- **Sync Capabilities**: Calendar synchronization between accounts (coming soon)


## 📋 Task Management (VTODO)

Chronos MCP provides comprehensive support for CalDAV tasks:

- **Create tasks** with due dates, priorities, and descriptions
- **Track progress** with percentage completion (0-100%)
- **Manage status**: NEEDS-ACTION, IN-PROCESS, COMPLETED, CANCELLED
- **Create subtasks** using related_to relationships
- **Bulk operations** for efficient task management

```bash
# Example: Create a task
mcp call create_task '{
  "calendar_uid": "my-calendar",
  "summary": "Complete project documentation",
  "due": "2025-02-01T15:00:00Z",
  "priority": 2
}'
```


## 📓 Journal Entries (VJOURNAL)

Keep detailed records with CalDAV journal entries:

- **Create journal entries** with timestamps and rich descriptions
- **Link related entries** using related_to relationships
- **Organize with categories** for better searchability
- **Update and manage** existing journal entries

```bash
# Example: Create a journal entry
mcp call create_journal '{
  "calendar_uid": "my-calendar",
  "summary": "Team Meeting Notes",
  "description": "Discussed Q1 objectives..."
}'
```

For detailed usage, see [VTODO/VJOURNAL Guide](docs/VTODO_VJOURNAL_GUIDE.md).


## 🔐 Security

### Secure Password Storage (New!)

Chronos MCP now supports secure password storage using your system's keyring (via python-keyring). When available, passwords are automatically stored in:
- **macOS**: Keychain Access
- **Windows**: Windows Credential Locker
- **Linux**: Secret Service (GNOME Keyring, KWallet, etc.)

### Migration to Secure Storage

If you have existing accounts with passwords stored in plain text, migrate them to secure storage:

```bash
# Check what will be migrated (dry run)
python scripts/migrate_to_keyring.py --dry-run

# Perform actual migration
python scripts/migrate_to_keyring.py
```

The migration script will:
1. Read existing passwords from `~/.chronos/accounts.json`
2. Store them securely in your system keyring
3. Create a backup of the original configuration
4. Remove passwords from the JSON file

### Fallback Behavior

If keyring is not available (e.g., SSH sessions, containers), Chronos MCP will:
- Warn about the security implications
- Fall back to storing passwords in the configuration file
- Automatically attempt to migrate passwords to keyring when it becomes available

### Legacy Security Warning

**Note**: If keyring is not installed or available, passwords will be stored in plain text at `~/.chronos/accounts.json`. Install keyring support with:

```bash
pip install "chronos-mcp[secure]"  # or just: pip install keyring
```

## Installation

### Standard Installation
```bash
pip install -e .
```

### Secure Installation (Recommended)
Includes keyring support for secure password storage:
```bash
pip install -e ".[secure]"
```

Or if you already have Chronos installed:
```bash
pip install keyring>=24.0.0
```

## Configuration

### Environment Variables (Default Account)
```bash
CALDAV_BASE_URL=http://<YOUR_CALDAV_SERVER>:5232
CALDAV_USERNAME=<YOUR_USERNAME>
CALDAV_PASSWORD=<YOUR_PASSWORD>
```

### Multi-Account Configuration

Create `~/.chronos/accounts.json`:
```json
{
  "accounts": {
    "personal": {
      "url": "http://<YOUR_CALDAV_SERVER>:5232",
      "username": "<YOUR_USERNAME>",
      "display_name": "Personal Calendar"
    },
    "work": {
      "url": "https://caldav.company.com",
      "username": "user",
      "display_name": "Work Calendar"
    }
  },
  "default_account": "personal"
}
```

**Note**: Passwords are not included in the JSON when using keyring. They will be:
- Prompted for on first use and stored securely
- Migrated from existing configuration using `scripts/migrate_to_keyring.py`
- Only stored in JSON if keyring is unavailable (with a warning)

## Usage

### Running the Server
```bash
./run_chronos.sh
```

### Basic Operations

List all configured accounts:
```
list_accounts()
```


### Example Tool Usage

Create an event with reminder:
```python
chronos:create_event(
    calendar_uid="assistant",
    summary="Team Meeting",
    start="2025-07-08T14:00:00",
    end="2025-07-08T15:00:00",
    location="Conference Room",
    alarm_minutes="15"  # Note: Pass as string!
)
```

Create recurring event with attendees:
```python
chronos:create_event(
    calendar_uid="work",
    summary="Weekly Standup",
    start="2025-07-07T09:00:00", 
    end="2025-07-07T09:30:00",
    recurrence_rule="FREQ=WEEKLY;BYDAY=MO,WE,FR",
    attendees_json='[{"email": "team@example.com", "name": "Team"}]'
)
```

Delete an event:
```python
chronos:delete_event(
    calendar_uid="assistant",
    event_uid="abc-123-def-456"
)
```

Update an event (partial update):
```python
chronos:update_event(
    calendar_uid="assistant",
    event_uid="abc-123-def-456",
    location="Room 202",  # Update location
    alarm_minutes="30"    # Change reminder to 30 minutes
    # Other fields remain unchanged
)
```

## Documentation

- [Usage](#usage) - Basic tool usage examples
- [Architecture Guide](docs/ARCHITECTURE.md) - System design and components
- [RRULE Guide](docs/RRULE_GUIDE.md) - Recurring events documentation
- [VTODO/VJOURNAL Guide](docs/VTODO_VJOURNAL_GUIDE.md) - Task and journal entry documentation

## Known Issues

See [GitHub Issues](https://github.com/democratize-technology/chronos-mcp/issues) for current limitations and workarounds.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
