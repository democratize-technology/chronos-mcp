"""
Comprehensive SSRF (Server-Side Request Forgery) protection tests for URL validation.

This module tests the enhanced URL validation that prevents SSRF attacks by blocking
requests to localhost, private IP ranges, and other potentially dangerous addresses.
"""

import socket
from unittest.mock import MagicMock, patch

import pytest

from chronos_mcp.exceptions import ValidationError
from chronos_mcp.validation import InputValidator


class TestSSRFProtection:
    """Test suite for SSRF protection in URL validation"""

    def test_validate_url_blocks_localhost(self):
        """Test that localhost in various forms is blocked by default"""
        validator = InputValidator()

        localhost_urls = [
            "https://localhost/caldav",
            "https://localhost:8443/caldav",
            "https://LOCALHOST/caldav",  # Case variations
            "https://localhost.localdomain/caldav",
            "https://127.0.0.1/caldav",
            "https://127.0.0.1:8443/caldav",
            "https://127.0.0.2/caldav",  # Other loopback addresses
            "https://127.255.255.255/caldav",  # End of loopback range
        ]

        for url in localhost_urls:
            with pytest.raises(ValidationError) as exc_info:
                validator.validate_url(url)

            error_msg = str(exc_info.value)
            assert (
                "not allowed for security reasons" in error_msg
                or "Unable to resolve hostname" in error_msg
            ), f"URL should be blocked for SSRF protection: {url}"

    def test_validate_url_blocks_ipv6_localhost(self):
        """Test that IPv6 localhost addresses are blocked"""
        validator = InputValidator()

        ipv6_localhost_urls = [
            "https://[::1]/caldav",
            "https://[::1]:8443/caldav",
            "https://[::ffff:127.0.0.1]/caldav",  # IPv4-mapped IPv6
        ]

        for url in ipv6_localhost_urls:
            # IPv6 URLs might not match our pattern or fail in resolution
            with pytest.raises(ValidationError):
                validator.validate_url(url)

    def test_validate_url_blocks_private_ipv4_ranges(self):
        """Test that private IPv4 ranges are blocked"""
        validator = InputValidator()

        private_ip_urls = [
            # Class A private (10.0.0.0/8)
            "https://10.0.0.1/caldav",
            "https://10.255.255.255/caldav",
            "https://10.1.2.3:8443/caldav",
            # Class B private (172.16.0.0/12)
            "https://172.16.0.1/caldav",
            "https://172.31.255.255/caldav",
            "https://172.20.10.5:8443/caldav",
            # Class C private (192.168.0.0/16)
            "https://192.168.0.1/caldav",
            "https://192.168.1.1/caldav",
            "https://192.168.255.255/caldav",
            "https://192.168.1.100:8443/caldav",
        ]

        for url in private_ip_urls:
            with pytest.raises(ValidationError) as exc_info:
                validator.validate_url(url)

            error_msg = str(exc_info.value)
            assert (
                "private or internal IP address" in error_msg
                or "Unable to resolve hostname" in error_msg
            ), f"Private IP should be blocked: {url}"

    def test_validate_url_blocks_link_local_addresses(self):
        """Test that link-local addresses are blocked"""
        validator = InputValidator()

        link_local_urls = [
            "https://169.254.0.1/caldav",
            "https://169.254.169.254/caldav",  # AWS metadata endpoint
            "https://169.254.255.255/caldav",
        ]

        for url in link_local_urls:
            with pytest.raises(ValidationError) as exc_info:
                validator.validate_url(url)

            error_msg = str(exc_info.value)
            assert (
                "private or internal IP address" in error_msg
                or "restricted IP address" in error_msg
                or "Unable to resolve hostname" in error_msg
            ), f"Link-local address should be blocked: {url}"

    def test_validate_url_blocks_zero_address(self):
        """Test that 0.0.0.0 is blocked"""
        validator = InputValidator()

        with pytest.raises(ValidationError):
            validator.validate_url("https://0.0.0.0/caldav")

    @patch("socket.getaddrinfo")
    def test_validate_url_blocks_domains_resolving_to_private_ips(
        self, mock_getaddrinfo
    ):
        """Test that domains resolving to private IPs are blocked"""
        validator = InputValidator()

        # Mock a domain that resolves to a private IP
        test_cases = [
            ("internal.example.com", "192.168.1.100"),
            ("dev.local", "10.0.0.50"),
            ("staging.app", "172.16.0.10"),
            ("metadata.local", "169.254.169.254"),
        ]

        for domain, private_ip in test_cases:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (private_ip, 443))
            ]

            url = f"https://{domain}/caldav"
            with pytest.raises(ValidationError) as exc_info:
                validator.validate_url(url)

            error_msg = str(exc_info.value)
            assert (
                "private or internal IP address" in error_msg
                or "restricted IP address" in error_msg
            ), f"Domain resolving to {private_ip} should be blocked: {url}"

    @patch("socket.getaddrinfo")
    def test_validate_url_allows_public_ips(self, mock_getaddrinfo):
        """Test that public IP addresses are allowed"""
        validator = InputValidator()

        # Mock domains resolving to public IPs
        public_test_cases = [
            ("caldav.example.com", "93.184.216.34"),  # Public IP
            ("calendar.company.org", "8.8.8.8"),  # Google DNS
            ("sync.service.io", "1.1.1.1"),  # Cloudflare DNS
        ]

        for domain, public_ip in public_test_cases:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (public_ip, 443))
            ]

            url = f"https://{domain}/caldav"
            result = validator.validate_url(url)
            assert result == url, f"Public IP should be allowed: {url}"

    def test_validate_url_with_allow_private_ips_flag(self):
        """Test that private IPs are allowed when flag is set"""
        validator = InputValidator()

        # URLs that would normally be blocked
        private_urls = [
            "https://localhost/caldav",
            "https://127.0.0.1/caldav",
            "https://192.168.1.100/caldav",
            "https://10.0.0.50/caldav",
            "https://172.16.0.10/caldav",
        ]

        for url in private_urls:
            # Should pass when allow_private_ips=True
            result = validator.validate_url(url, allow_private_ips=True)
            assert result == url, f"Private IP should be allowed with flag: {url}"

            # Should fail when allow_private_ips=False (default)
            with pytest.raises(ValidationError):
                validator.validate_url(url, allow_private_ips=False)

    @patch("socket.getaddrinfo")
    def test_validate_url_dns_rebinding_protection(self, mock_getaddrinfo):
        """Test protection against DNS rebinding attacks"""
        validator = InputValidator()

        # Simulate DNS rebinding - domain resolves to multiple IPs including private
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),  # Public
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("192.168.1.1", 443),
            ),  # Private
        ]

        url = "https://evil.example.com/caldav"
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_url(url)

        error_msg = str(exc_info.value)
        assert (
            "private or internal IP address" in error_msg
        ), "Should block domain with mixed public/private IPs"

    @patch("socket.getaddrinfo")
    def test_validate_url_handles_dns_resolution_failures(self, mock_getaddrinfo):
        """Test handling of DNS resolution failures"""
        validator = InputValidator()

        # Simulate DNS resolution failure
        mock_getaddrinfo.side_effect = socket.gaierror("Name or service not known")

        url = "https://nonexistent.example.com/caldav"
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_url(url)

        error_msg = str(exc_info.value)
        assert (
            "Unable to resolve hostname" in error_msg
        ), "Should handle DNS resolution failure gracefully"

    def test_validate_url_enforces_https(self):
        """Test that only HTTPS URLs are allowed"""
        validator = InputValidator()

        # HTTP and other protocols should be rejected
        invalid_protocols = [
            "http://example.com/caldav",
            "ftp://example.com/caldav",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
        ]

        for url in invalid_protocols:
            with pytest.raises(ValidationError) as exc_info:
                validator.validate_url(url)

            error_msg = str(exc_info.value)
            assert (
                "Invalid URL format" in error_msg
                or "Must be a valid HTTPS URL" in error_msg
            ), f"Non-HTTPS protocol should be rejected: {url}"

    def test_validate_url_length_limits(self):
        """Test URL length validation"""
        validator = InputValidator()

        # Create a URL that exceeds the maximum length
        long_path = "a" * 3000
        long_url = f"https://example.com/{long_path}"

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_url(long_url)

        error_msg = str(exc_info.value)
        assert "exceeds maximum length" in error_msg

    def test_is_private_ip_method(self):
        """Test the is_private_ip helper method"""
        validator = InputValidator()

        # Private IPs should return True
        private_ips = [
            "127.0.0.1",
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "169.254.1.1",
            "::1",
        ]

        for ip in private_ips:
            assert validator.is_private_ip(ip), f"Should identify {ip} as private"

        # Public IPs should return False
        public_ips = [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",
        ]

        for ip in public_ips:
            assert not validator.is_private_ip(ip), f"Should identify {ip} as public"

        # Invalid IPs should return True (fail-safe)
        invalid_ips = [
            "not-an-ip",
            "999.999.999.999",
            "",
        ]

        for ip in invalid_ips:
            assert validator.is_private_ip(
                ip
            ), f"Should treat invalid IP {ip} as suspicious"

    def test_validate_url_field_name_in_errors(self):
        """Test that custom field names appear in error messages"""
        validator = InputValidator()

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_url(
                "https://127.0.0.1/caldav", field_name="caldav_server"
            )

        error_msg = str(exc_info.value)
        assert "caldav_server" in error_msg, "Custom field name should appear in error"

    def test_validate_url_empty_url(self):
        """Test handling of empty URLs"""
        validator = InputValidator()

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_url("")

        error_msg = str(exc_info.value)
        assert "cannot be empty" in error_msg

    def test_validate_url_whitespace_handling(self):
        """Test that URLs with whitespace are properly handled"""
        validator = InputValidator()

        # Leading/trailing whitespace should be stripped
        result = validator.validate_url(
            "  https://example.com/caldav  ", allow_private_ips=True
        )
        assert result == "https://example.com/caldav"

        # Whitespace in URL should fail validation
        with pytest.raises(ValidationError):
            validator.validate_url("https://example .com/caldav")

    @patch("socket.getaddrinfo")
    def test_validate_url_ipv6_private_ranges(self, mock_getaddrinfo):
        """Test that private IPv6 ranges are blocked"""
        validator = InputValidator()

        # Mock domain resolving to private IPv6
        test_cases = [
            ("::1", "IPv6 loopback"),
            ("fe80::1", "IPv6 link-local"),
            ("fc00::1", "IPv6 unique local"),
            ("fd00::1", "IPv6 unique local"),
        ]

        for ipv6_addr, description in test_cases:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ipv6_addr, 443, 0, 0))
            ]

            url = "https://ipv6.example.com/caldav"
            with pytest.raises(ValidationError) as exc_info:
                validator.validate_url(url)

            error_msg = str(exc_info.value)
            assert (
                "private or internal IP address" in error_msg
                or "restricted IP address" in error_msg
            ), f"{description} should be blocked: {ipv6_addr}"

    def test_validate_url_special_case_addresses(self):
        """Test that special case addresses are handled correctly"""
        validator = InputValidator()

        special_addresses = [
            "https://0.0.0.0/caldav",  # Wildcard address
            "https://255.255.255.255/caldav",  # Broadcast address
        ]

        for url in special_addresses:
            with pytest.raises(ValidationError):
                validator.validate_url(url)


class TestBackwardCompatibility:
    """Test backward compatibility of the URL validation enhancement"""

    def test_pattern_still_accessible(self):
        """Test that the URL pattern is still accessible for existing code"""
        validator = InputValidator()

        # The pattern should still exist and work
        assert validator.PATTERNS["url"] is not None
        assert validator.PATTERNS["url"].match("https://example.com/caldav")
        assert not validator.PATTERNS["url"].match("http://example.com/caldav")

    def test_default_ssrf_protection_enabled(self):
        """Test that SSRF protection is enabled by default"""
        validator = InputValidator()

        # By default, private IPs should be blocked
        with pytest.raises(ValidationError):
            validator.validate_url("https://192.168.1.1/caldav")

        # But can be disabled for backward compatibility
        result = validator.validate_url(
            "https://192.168.1.1/caldav", allow_private_ips=True
        )
        assert result == "https://192.168.1.1/caldav"

    def test_validate_url_optional_parameters(self):
        """Test that all parameters have sensible defaults"""
        validator = InputValidator()

        # Should work with just URL (uses defaults)
        result = validator.validate_url("https://example.com/caldav")
        assert result == "https://example.com/caldav"

        # Can override field_name
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_url("invalid-url", field_name="custom_field")
        assert "custom_field" in str(exc_info.value)


class TestRealWorldScenarios:
    """Test real-world CalDAV server URLs and SSRF attack vectors"""

    @patch("socket.getaddrinfo")
    def test_common_caldav_servers_allowed(self, mock_getaddrinfo):
        """Test that common CalDAV servers are allowed"""
        validator = InputValidator()

        # Mock public IP resolution
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]

        caldav_urls = [
            "https://caldav.fastmail.com/dav/calendars/user/",
            "https://calendar.google.com/calendar/dav/",
            "https://outlook.office365.com/owa/calendar/",
            "https://dav.icloud.com/calendar/",
            "https://nextcloud.example.com/remote.php/dav/calendars/",
            "https://owncloud.example.org/remote.php/caldav/",
        ]

        for url in caldav_urls:
            result = validator.validate_url(url)
            assert result == url, f"Common CalDAV URL should be allowed: {url}"

    @patch("socket.getaddrinfo")
    def test_ssrf_attack_vectors_blocked(self, mock_getaddrinfo):
        """Test that common SSRF attack vectors are blocked"""
        validator = InputValidator()

        # Test various SSRF attack patterns
        attack_vectors = [
            # Direct private IPs
            (
                "https://169.254.169.254/latest/meta-data/",
                "169.254.169.254",
                "AWS metadata",
            ),
            ("https://metadata.google.internal/", "169.254.169.254", "GCP metadata"),
            # Encoded variations (should be caught by pattern validation)
            # These won't match our HTTPS pattern anyway
        ]

        for url, ip, description in attack_vectors:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))
            ]

            with pytest.raises(ValidationError) as exc_info:
                validator.validate_url(url)

            error_msg = str(exc_info.value)
            assert (
                "not allowed for security reasons" in error_msg
                or "Unable to resolve hostname" in error_msg
            ), f"SSRF vector should be blocked: {description}"

    def test_local_development_with_flag(self):
        """Test that local development can still work with explicit flag"""
        validator = InputValidator()

        # Local development URLs that might be legitimately used
        local_dev_urls = [
            "https://localhost:8443/caldav",
            "https://127.0.0.1:3000/api/caldav",
            "https://192.168.1.100:8080/dav",
            "https://10.0.0.50:443/calendar",
        ]

        for url in local_dev_urls:
            # Blocked by default
            with pytest.raises(ValidationError):
                validator.validate_url(url)

            # Allowed with flag for local development
            result = validator.validate_url(url, allow_private_ips=True)
            assert result == url, f"Should allow local URL with flag: {url}"
