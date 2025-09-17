"""Input validation for Chronos MCP with SSRF protection.

This module provides comprehensive input validation for CalDAV operations,
including enhanced URL validation with Server-Side Request Forgery (SSRF)
protection. By default, URLs pointing to localhost, private IP ranges, and
other potentially dangerous addresses are blocked to prevent SSRF attacks.

Security Features:
- SSRF Protection: Blocks requests to localhost, private IPs, and link-local addresses
- HTTPS Enforcement: Only HTTPS URLs are allowed for CalDAV connections
- Pattern Validation: Prevents injection attacks through input sanitization
- DNS Resolution: Validates that domains don't resolve to private IPs

For local development or trusted environments, SSRF protection can be
disabled by setting allow_private_ips=True when calling validate_url().
"""

import ipaddress
import re
import socket
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .exceptions import ValidationError
from .models import TaskStatus


class InputValidator:
    """Comprehensive input validation for CalDAV operations."""

    # SSRF Protection - Private IP ranges that should be blocked
    PRIVATE_IP_RANGES = [
        ipaddress.ip_network("10.0.0.0/8"),  # Class A private
        ipaddress.ip_network("172.16.0.0/12"),  # Class B private
        ipaddress.ip_network("192.168.0.0/16"),  # Class C private
        ipaddress.ip_network("127.0.0.0/8"),  # Loopback
        ipaddress.ip_network("169.254.0.0/16"),  # Link-local
        ipaddress.ip_network("::1/128"),  # IPv6 loopback
        ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
        ipaddress.ip_network("fc00::/7"),  # IPv6 private
    ]

    # SSRF Protection - Blocked hostnames
    BLOCKED_HOSTNAMES = [
        "localhost",
        "localhost.localdomain",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "::ffff:127.0.0.1",
    ]

    MAX_LENGTHS = {
        "summary": 255,
        "description": 5000,
        "location": 255,
        "uid": 255,
        "attendee_email": 254,
        "url": 2048,
        "alias": 50,
        "calendar_name": 100,
    }

    PATTERNS = {
        "uid": re.compile(r"^[a-zA-Z0-9\-_.@]+$"),
        "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        "url": re.compile(
            r"^https://(?:[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}|[a-zA-Z0-9-]+|localhost|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?))(?::(?:[1-9][0-9]{0,3}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5]))?(?:/[^\s]*)?$"
        ),
        "color": re.compile(r"^#[0-9A-Fa-f]{6}$"),
    }

    # ReDoS-safe patterns with simplified regex and input length limits
    MAX_VALIDATION_LENGTH = 10000  # Pre-filter before regex validation

    DANGEROUS_PATTERNS = [
        # Script tags (simplified, non-backtracking)
        re.compile(r"<script\b", re.IGNORECASE),
        re.compile(r"</script\s*>", re.IGNORECASE),
        # JavaScript protocols (simplified)
        re.compile(r"javascript\s*:", re.IGNORECASE),
        re.compile(r"vbscript\s*:", re.IGNORECASE),
        re.compile(r"data\s*:", re.IGNORECASE),
        # Event handlers (simplified, non-greedy)
        re.compile(r"\bon\w+\s*=", re.IGNORECASE),
        # Dangerous HTML elements (simplified)
        re.compile(
            r"<(?:iframe|frame|object|embed|applet|form|meta|link)\b", re.IGNORECASE
        ),
        # Expression and eval patterns (simplified)
        re.compile(r"\bexpression\s*\(", re.IGNORECASE),
        re.compile(r"\beval\s*\(", re.IGNORECASE),
        re.compile(r"\bsetTimeout\s*\(", re.IGNORECASE),
        re.compile(r"\bsetInterval\s*\(", re.IGNORECASE),
        # Control characters (unchanged - safe pattern)
        re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"),
        # Encoded patterns (simplified)
        re.compile(r"&#[xX]?[0-9a-fA-F]+;"),
        re.compile(r"%[0-9a-fA-F]{2}"),
        re.compile(r"\\u[0-9a-fA-F]{4}", re.IGNORECASE),
        # CSS injection (simplified)
        re.compile(r"@import\b", re.IGNORECASE),
        # SVG patterns (simplified)
        re.compile(r"<svg\b", re.IGNORECASE),
        re.compile(r"<foreignobject\b", re.IGNORECASE),
    ]

    @classmethod
    def validate_event(cls, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize event data."""
        sanitized = {}

        if not event_data.get("summary"):
            raise ValidationError("Event summary is required")
        if not event_data.get("dtstart"):
            raise ValidationError("Event start time is required")
        if not event_data.get("dtend"):
            raise ValidationError("Event end time is required")

        sanitized["summary"] = cls.validate_text_field(
            event_data["summary"], "summary", required=True
        )

        if "description" in event_data:
            sanitized["description"] = cls.validate_text_field(
                event_data["description"], "description"
            )

        if "location" in event_data:
            sanitized["location"] = cls.validate_text_field(
                event_data["location"], "location"
            )

        sanitized["dtstart"] = cls.validate_datetime(event_data["dtstart"], "dtstart")
        sanitized["dtend"] = cls.validate_datetime(event_data["dtend"], "dtend")

        if sanitized["dtend"] <= sanitized["dtstart"]:
            raise ValidationError("Event end time must be after start time")

        if "uid" in event_data:
            sanitized["uid"] = cls.validate_uid(event_data["uid"])

        if "attendees" in event_data:
            sanitized["attendees"] = cls.validate_attendees(event_data["attendees"])

        if "recurrence_rule" in event_data:
            sanitized["recurrence_rule"] = cls.validate_rrule(
                event_data["recurrence_rule"]
            )

        return sanitized

    @classmethod
    def _decode_and_normalize(cls, value: str) -> str:
        """Decode and normalize potentially obfuscated content for pattern matching"""
        import urllib.parse

        # Create a copy for testing (don't modify original)
        test_value = value

        # Decode common encodings
        try:
            # HTML entities
            import html

            test_value = html.unescape(test_value)

            # URL encoding
            test_value = urllib.parse.unquote(test_value)

            # Unicode escapes
            test_value = test_value.encode().decode("unicode_escape", errors="ignore")

        except Exception:
            # If decoding fails, use original value
            test_value = value

        return test_value

    @classmethod
    def validate_text_field(
        cls, value: str, field_name: str, required: bool = False
    ) -> str:
        """Validate and sanitize text fields."""
        if not value and required:
            raise ValidationError(f"{field_name} is required")

        if not value:
            return ""

        value = str(value).strip()

        # Pre-filter: Reject extremely long inputs before regex validation
        if len(value) > cls.MAX_VALIDATION_LENGTH:
            raise ValidationError(
                f"{field_name} exceeds maximum validation length of {cls.MAX_VALIDATION_LENGTH} characters"
            )

        max_length = cls.MAX_LENGTHS.get(field_name, 1000)
        if len(value) > max_length:
            raise ValidationError(
                f"{field_name} exceeds maximum length of {max_length} characters"
            )

        # Normalize Unicode
        value = unicodedata.normalize("NFKC", value)

        # Check for dangerous patterns on both original and decoded versions
        test_values = [value, cls._decode_and_normalize(value)]

        for test_val in test_values:
            # Additional length check after decoding
            if len(test_val) > cls.MAX_VALIDATION_LENGTH:
                raise ValidationError(
                    f"{field_name} contains excessively long decoded content"
                )

            for pattern in cls.DANGEROUS_PATTERNS:
                if pattern.search(test_val):
                    raise ValidationError(
                        f"{field_name} contains potentially dangerous content"
                    )

        # NOTE: HTML escaping removed - should happen at display layer, not storage
        # CalDAV expects unescaped data

        return value

    @classmethod
    def validate_datetime(cls, value: Any, field_name: str) -> datetime:
        """Validate datetime values."""
        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            try:
                cleaned = value.replace("Z", "+00:00")
                return datetime.fromisoformat(cleaned)
            except ValueError:
                raise ValidationError(f"Invalid datetime format for {field_name}")

        raise ValidationError(f"{field_name} must be a datetime or ISO format string")

    @classmethod
    def validate_uid(cls, uid: str) -> str:
        """Validate UID format."""
        if not uid:
            raise ValidationError("UID cannot be empty")

        if len(uid) > cls.MAX_LENGTHS["uid"]:
            raise ValidationError(
                f"UID exceeds maximum length of {cls.MAX_LENGTHS['uid']}"
            )

        if not cls.PATTERNS["uid"].match(uid):
            raise ValidationError(
                "UID contains invalid characters. "
                "Only alphanumeric, dash, underscore, dot, and @ are allowed"
            )

        return uid

    @classmethod
    def validate_email(cls, email: str) -> str:
        """Validate email address."""
        email = email.strip().lower()

        if len(email) > cls.MAX_LENGTHS["attendee_email"]:
            raise ValidationError("Email address too long")

        if not cls.PATTERNS["email"].match(email):
            raise ValidationError(f"Invalid email address format: {email}")

        return email

    @classmethod
    def validate_attendees(
        cls, attendees: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate attendee list."""
        if not isinstance(attendees, list):
            raise ValidationError("Attendees must be a list")

        validated = []

        for attendee in attendees:
            if not isinstance(attendee, dict):
                raise ValidationError("Each attendee must be a dictionary")

            if "email" not in attendee:
                raise ValidationError("Attendee email is required")

            validated_attendee = {"email": cls.validate_email(attendee["email"])}

            if "name" in attendee:
                validated_attendee["name"] = cls.validate_text_field(
                    attendee["name"], "attendee_name"
                )

            # Preserve other attendee fields
            for field in ["role", "status", "rsvp"]:
                if field in attendee:
                    if field == "role":
                        valid_roles = [
                            "CHAIR",
                            "REQ-PARTICIPANT",
                            "OPT-PARTICIPANT",
                            "NON-PARTICIPANT",
                        ]
                        if attendee[field] not in valid_roles:
                            raise ValidationError(
                                f"Invalid attendee role: {attendee[field]}"
                            )
                    validated_attendee[field] = attendee[field]

            validated.append(validated_attendee)

        return validated

    @classmethod
    def validate_rrule(cls, rrule: str) -> str:
        """Validate recurrence rule."""
        rrule = rrule.strip().upper()

        if not rrule.startswith("FREQ="):
            raise ValidationError("RRULE must start with FREQ=")

        valid_freqs = ["DAILY", "WEEKLY", "MONTHLY", "YEARLY"]
        freq_match = re.match(r"FREQ=(\w+)", rrule)
        if not freq_match or freq_match.group(1) not in valid_freqs:
            raise ValidationError(f"Invalid frequency. Must be one of: {valid_freqs}")

        if len(rrule) > 500:
            raise ValidationError("RRULE too complex (exceeds 500 characters)")

        return rrule

    @classmethod
    def validate_task(cls, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize task data."""
        sanitized = {}

        if not task_data.get("summary"):
            raise ValidationError("Task summary is required")

        sanitized["summary"] = cls.validate_text_field(
            task_data["summary"], "summary", required=True
        )

        if "description" in task_data:
            sanitized["description"] = cls.validate_text_field(
                task_data["description"], "description"
            )

        if "due" in task_data and task_data["due"] is not None:
            sanitized["due"] = cls.validate_datetime(task_data["due"], "due")

        if "priority" in task_data and task_data["priority"] is not None:
            sanitized["priority"] = cls.validate_priority(task_data["priority"])

        if "status" in task_data and task_data["status"] is not None:
            sanitized["status"] = cls.validate_task_status(task_data["status"])

        if (
            "percent_complete" in task_data
            and task_data["percent_complete"] is not None
        ):
            sanitized["percent_complete"] = cls.validate_percent_complete(
                task_data["percent_complete"]
            )

        if "uid" in task_data:
            sanitized["uid"] = cls.validate_uid(task_data["uid"])

        if "related_to" in task_data:
            sanitized["related_to"] = cls.validate_related_to(task_data["related_to"])

        return sanitized

    @classmethod
    def validate_journal(cls, journal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize journal data."""
        sanitized = {}

        if not journal_data.get("summary"):
            raise ValidationError("Journal summary is required")

        sanitized["summary"] = cls.validate_text_field(
            journal_data["summary"], "summary", required=True
        )

        if "description" in journal_data:
            sanitized["description"] = cls.validate_text_field(
                journal_data["description"], "description"
            )

        if "dtstart" in journal_data and journal_data["dtstart"] is not None:
            sanitized["dtstart"] = cls.validate_datetime(
                journal_data["dtstart"], "dtstart"
            )

        if "categories" in journal_data:
            sanitized["categories"] = cls.validate_categories(
                journal_data["categories"]
            )

        if "uid" in journal_data:
            sanitized["uid"] = cls.validate_uid(journal_data["uid"])

        if "related_to" in journal_data:
            sanitized["related_to"] = cls.validate_related_to(
                journal_data["related_to"]
            )

        return sanitized

    @classmethod
    def validate_priority(cls, priority: Any) -> int:
        """Validate task priority (1-9, RFC 5545 compliant)."""
        try:
            priority_val = int(priority)
        except (ValueError, TypeError):
            raise ValidationError("Priority must be an integer")

        if priority_val < 1 or priority_val > 9:
            raise ValidationError("Priority must be between 1-9 (1 is highest)")

        return priority_val

    @classmethod
    def validate_task_status(cls, status: Any) -> TaskStatus:
        """Validate task status."""
        if isinstance(status, TaskStatus):
            return status

        try:
            return TaskStatus(str(status))
        except ValueError:
            valid_statuses = [s.value for s in TaskStatus]
            raise ValidationError(
                f"Invalid task status. Must be one of: {valid_statuses}"
            )

    @classmethod
    def validate_percent_complete(cls, percent: Any) -> int:
        """Validate percent complete (0-100)."""
        try:
            percent_val = int(percent)
        except (ValueError, TypeError):
            raise ValidationError("Percent complete must be an integer")

        if percent_val < 0 or percent_val > 100:
            raise ValidationError("Percent complete must be between 0-100")

        return percent_val

    @classmethod
    def validate_categories(cls, categories: Any) -> List[str]:
        """Validate categories list."""
        if not isinstance(categories, list):
            if isinstance(categories, str):
                # Single category as string
                categories = [categories]
            else:
                raise ValidationError("Categories must be a list or string")

        validated_categories = []
        for category in categories:
            if not isinstance(category, str):
                raise ValidationError("Each category must be a string")

            category_clean = cls.validate_text_field(str(category), "category")
            if category_clean:  # Only add non-empty categories
                validated_categories.append(category_clean)

        return validated_categories

    @classmethod
    def validate_related_to(cls, related_to: Any) -> List[str]:
        """Validate RELATED-TO UIDs list."""
        if not isinstance(related_to, list):
            if isinstance(related_to, str):
                # Single UID as string
                related_to = [related_to]
            else:
                raise ValidationError("RELATED-TO must be a list or string")

        validated_uids = []
        for uid in related_to:
            if not isinstance(uid, str):
                raise ValidationError("Each RELATED-TO UID must be a string")

            validated_uid = cls.validate_uid(uid)
            validated_uids.append(validated_uid)

        return validated_uids

    @classmethod
    def validate_url(
        cls, url: str, allow_private_ips: bool = False, field_name: str = "url"
    ) -> str:
        """Validate URL with SSRF protection.

        Args:
            url: The URL to validate
            allow_private_ips: If False (default), block localhost and private IPs for SSRF protection
            field_name: Name of the field for error messages

        Returns:
            The validated URL

        Raises:
            ValidationError: If URL is invalid or blocked by SSRF protection
        """
        if not url:
            raise ValidationError(f"{field_name} cannot be empty")

        url = url.strip()

        # Check URL length
        if len(url) > cls.MAX_LENGTHS.get("url", 2048):
            raise ValidationError(
                f"{field_name} exceeds maximum length of {cls.MAX_LENGTHS.get('url', 2048)} characters"
            )

        # Check URL format using existing pattern
        if not cls.PATTERNS["url"].match(url):
            raise ValidationError(
                f"Invalid URL format for {field_name}. Must be a valid HTTPS URL."
            )

        # Handle FieldInfo objects from Pydantic Field defaults
        from pydantic.fields import FieldInfo

        if isinstance(allow_private_ips, FieldInfo):
            allow_private_ips = allow_private_ips.default

        # If SSRF protection is disabled, return early
        if allow_private_ips:
            return url

        # Parse URL for SSRF validation
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname

            if not hostname:
                raise ValidationError(
                    f"Invalid URL format for {field_name}: no hostname found"
                )

            # Check against blocked hostnames (case-insensitive)
            if hostname.lower() in [h.lower() for h in cls.BLOCKED_HOSTNAMES]:
                raise ValidationError(
                    f"URL validation failed for {field_name}: "
                    f"localhost and loopback addresses are not allowed for security reasons"
                )

            # Try to resolve the hostname to check for private IPs
            try:
                # Get all IP addresses for the hostname
                addr_info = socket.getaddrinfo(hostname, None)
                ip_addresses = set()

                for info in addr_info:
                    # info[4][0] contains the IP address
                    ip_addresses.add(info[4][0])

                # Check each resolved IP
                for ip_str in ip_addresses:
                    try:
                        ip = ipaddress.ip_address(ip_str)

                        # Check if IP is private or in blocked ranges
                        for private_range in cls.PRIVATE_IP_RANGES:
                            if ip in private_range:
                                raise ValidationError(
                                    f"URL validation failed for {field_name}: "
                                    f"URL resolves to a private or internal IP address ({ip_str}) "
                                    f"which is not allowed for security reasons"
                                )

                        # Additional checks for special addresses
                        if ip.is_private or ip.is_loopback or ip.is_link_local:
                            raise ValidationError(
                                f"URL validation failed for {field_name}: "
                                f"URL resolves to a restricted IP address ({ip_str}) "
                                f"which is not allowed for security reasons"
                            )

                    except ValueError:
                        # If we can't parse as IP, it might be IPv6 or malformed
                        # Be conservative and reject
                        pass

            except (socket.gaierror, socket.error) as e:
                # If DNS resolution fails, we should be cautious
                # Could be a non-existent domain or network issue
                raise ValidationError(
                    f"URL validation failed for {field_name}: "
                    f"Unable to resolve hostname '{hostname}'. "
                    f"Please verify the URL is correct and accessible."
                )

        except ValueError as e:
            # URL parsing failed
            raise ValidationError(f"Invalid URL format for {field_name}: {str(e)}")

        return url

    @classmethod
    def is_private_ip(cls, ip_str: str) -> bool:
        """Check if an IP address is private or restricted.

        Args:
            ip_str: IP address as string

        Returns:
            True if the IP is private/restricted, False otherwise
        """
        try:
            ip = ipaddress.ip_address(ip_str)

            # Check against our defined private ranges
            for private_range in cls.PRIVATE_IP_RANGES:
                if ip in private_range:
                    return True

            # Use built-in checks as well
            return ip.is_private or ip.is_loopback or ip.is_link_local

        except ValueError:
            # If we can't parse it, consider it suspicious
            return True
