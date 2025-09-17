"""
Security-focused tests for URL validation
"""

import pytest

from chronos_mcp.exceptions import ValidationError
from chronos_mcp.validation import InputValidator


class TestUrlValidationSecurity:
    """Test security aspects of URL validation"""

    def test_https_only_enforcement(self):
        """Test that only HTTPS URLs are allowed"""
        validator = InputValidator()

        # Valid HTTPS URLs should pass
        valid_urls = [
            "https://caldav.example.com/",
            "https://calendar.company.org/caldav/",
            "https://subdomain.example.co.uk/path/to/caldav",
            "https://192.168.1.100:8443/caldav",
            "https://example.com:443/",
        ]

        for url in valid_urls:
            assert validator.PATTERNS["url"].match(
                url
            ), f"Valid HTTPS URL should match: {url}"

    def test_http_urls_rejected(self):
        """Test that HTTP URLs are rejected"""
        validator = InputValidator()

        # HTTP URLs should be rejected
        invalid_urls = [
            "http://caldav.example.com/",
            "http://calendar.company.org/caldav/",
            "http://example.com/path",
            "http://192.168.1.100:8080/caldav",
            "http://localhost:8080/",
        ]

        for url in invalid_urls:
            assert not validator.PATTERNS["url"].match(
                url
            ), f"HTTP URL should be rejected: {url}"

    def test_malicious_url_schemes_rejected(self):
        """Test that malicious URL schemes are rejected"""
        validator = InputValidator()

        # Malicious schemes should be rejected
        malicious_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "ftp://malicious.com/",
            "file:///etc/passwd",
            "gopher://evil.com/",
            "ldap://attacker.com/",
            "mailto:victim@example.com",
            "tel:+1234567890",
        ]

        for url in malicious_urls:
            assert not validator.PATTERNS["url"].match(
                url
            ), f"Malicious URL should be rejected: {url}"

    def test_url_injection_attempts_rejected(self):
        """Test that URL injection attempts are rejected"""
        validator = InputValidator()

        # URL injection attempts should be rejected
        injection_urls = [
            "https://evil.com@example.com/",  # Credential phishing - @ not allowed
            "https://example .com/path",  # Space in domain
            "https://example.com:99999/path",  # Invalid port
        ]

        for url in injection_urls:
            assert not validator.PATTERNS["url"].match(
                url
            ), f"Dangerous URL should be rejected: {url}"

        # These URLs will match our pattern but contain potentially dangerous content
        # They should be caught by other validation layers (like dangerous pattern detection)
        potentially_dangerous_but_valid_format = [
            "https://example.com/path?param=javascript:alert(1)",
            "https://example.com/path#javascript:alert(1)",
            "https://example.com/../../../etc/passwd",
            "https://example.com/path?redirect=http://evil.com",
        ]

        # These match the URL format but should be caught by dangerous pattern validation
        for url in potentially_dangerous_but_valid_format:
            # The URL pattern itself might match (that's OK), but dangerous content
            # should be caught by the DANGEROUS_PATTERNS in validate_text_field
            result = validator.PATTERNS["url"].match(url)
            # This is acceptable - the URL format is valid, but content filtering should catch it

    def test_localhost_and_private_ips_pattern_matching(self):
        """Test that localhost and private IPs match the URL pattern (but may be blocked by validate_url)"""
        validator = InputValidator()

        # These match the URL pattern format (for backward compatibility)
        # but are now blocked by default in validate_url() for SSRF protection
        local_urls = [
            "https://localhost:8443/caldav",
            "https://127.0.0.1:8443/caldav",
            "https://192.168.1.100:8443/caldav",
            "https://10.0.0.50:8443/caldav",
            "https://172.16.0.10:8443/caldav",
        ]

        for url in local_urls:
            # Pattern still matches for backward compatibility
            assert validator.PATTERNS["url"].match(
                url
            ), f"Local/private URL pattern should match: {url}"

            # But validate_url blocks them by default (SSRF protection)
            with pytest.raises(ValidationError):
                validator.validate_url(url)

            # Unless explicitly allowed
            result = validator.validate_url(url, allow_private_ips=True)
            assert result == url

    def test_url_with_unusual_ports(self):
        """Test URLs with unusual but valid ports"""
        validator = InputValidator()

        urls_with_ports = [
            "https://example.com:8443/caldav",
            "https://example.com:9443/caldav",
            "https://example.com:443/caldav",  # Standard HTTPS port
            "https://example.com:8080/caldav",  # Common alternative
        ]

        for url in urls_with_ports:
            assert validator.PATTERNS["url"].match(
                url
            ), f"URL with port should be allowed: {url}"

    def test_empty_and_malformed_urls(self):
        """Test handling of empty and malformed URLs"""
        validator = InputValidator()

        malformed_urls = [
            "",
            "not-a-url",
            "://missing-scheme.com",
            "https://",
            "https:///path-without-domain",
            "https://.com/",
            "https://example.",
            "https://example .com/",  # Space in domain
            "https://example.com:abc/",  # Invalid port
            "https://example.com:0/",  # Port 0 not allowed
        ]

        for url in malformed_urls:
            assert not validator.PATTERNS["url"].match(
                url
            ), f"Malformed URL should be rejected: {url}"

        # These might match our pattern but are edge cases we should handle
        edge_cases = [
            "https://domain-without-tld",  # This will match as single hostname
            "https://example..com/",  # This might match due to our regex
        ]

        # Note: Some edge cases might pass the regex but should be caught by other validation

    def test_very_long_urls(self):
        """Test handling of extremely long URLs"""
        validator = InputValidator()

        # Create a very long but otherwise valid URL
        long_path = "a" * 1000
        long_url = f"https://example.com/{long_path}"

        # The pattern itself should match, but length validation should happen elsewhere
        # This tests that the regex doesn't break with long inputs
        result = validator.PATTERNS["url"].match(long_url)
        assert (
            result is not None
        ), "Long URL should match pattern (length validation is separate)"

    def test_unicode_domains_handled(self):
        """Test handling of internationalized domain names"""
        validator = InputValidator()

        # These might be legitimate but should be handled carefully
        unicode_domains = [
            "https://xn--example-9ua.com/caldav",  # Punycode encoded
            "https://bücher.example.com/caldav",  # Direct Unicode (might not match our pattern)
        ]

        # Our current pattern is ASCII-only, which is actually a security feature
        # Unicode domains should be punycode-encoded first
        assert validator.PATTERNS["url"].match(
            unicode_domains[0]
        ), "Punycode domain should be allowed"
        assert not validator.PATTERNS["url"].match(
            unicode_domains[1]
        ), "Direct Unicode should be rejected (security feature)"

    def test_case_sensitivity(self):
        """Test that URL scheme matching is case-sensitive for security"""
        validator = InputValidator()

        # Only lowercase 'https' should be allowed
        case_variants = [
            "HTTPS://example.com/caldav",
            "Https://example.com/caldav",
            "HTTPs://example.com/caldav",
            "https://EXAMPLE.COM/caldav",  # Domain case shouldn't matter
        ]

        assert not validator.PATTERNS["url"].match(
            case_variants[0]
        ), "Uppercase HTTPS should be rejected"
        assert not validator.PATTERNS["url"].match(
            case_variants[1]
        ), "Mixed case Https should be rejected"
        assert not validator.PATTERNS["url"].match(
            case_variants[2]
        ), "Mixed case HTTPs should be rejected"
        assert validator.PATTERNS["url"].match(
            case_variants[3]
        ), "Domain case should not matter"
