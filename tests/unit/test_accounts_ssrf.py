"""
Test SSRF protection in accounts module
"""

import pytest
from unittest.mock import MagicMock, patch

from chronos_mcp.exceptions import ValidationError
from chronos_mcp.tools import accounts


class TestAccountsSSRFProtection:
    """Test that the accounts module properly uses SSRF protection"""

    @pytest.mark.asyncio
    async def test_add_account_blocks_private_ips_by_default(self):
        """Test that add_account blocks private IPs by default"""

        # Directly set managers in the accounts module
        accounts._managers["config_manager"] = MagicMock()
        accounts._managers["account_manager"] = MagicMock()

        # Try to add account with localhost URL
        result = await accounts.add_account(
            alias="local-test",
            url="https://localhost:8443/caldav",
            username="user",
            password="pass",
        )

        # The decorator catches ValidationError and returns error response
        assert result["success"] is False
        assert "not allowed for security reasons" in result["error"]

        # Try with private IP
        result = await accounts.add_account(
            alias="private-test",
            url="https://192.168.1.100/caldav",
            username="user",
            password="pass",
        )

        assert result["success"] is False
        assert "not allowed for security reasons" in result["error"]

    @pytest.mark.asyncio
    async def test_add_account_allows_private_ips_with_flag(self):
        """Test that add_account allows private IPs when explicitly enabled"""

        # Mock the managers
        mock_config_manager = MagicMock()
        mock_account_manager = MagicMock()
        mock_account_manager.test_account.return_value = {
            "connected": True,
            "calendars": [],
        }

        accounts._managers["config_manager"] = mock_config_manager
        accounts._managers["account_manager"] = mock_account_manager

        # Should work with allow_local=True
        result = await accounts.add_account(
            alias="local-dev",
            url="https://localhost:8443/caldav",
            username="user",
            password="pass",
            allow_local=True,  # Explicitly allow local IPs
        )

        assert result["success"] is True
        assert "local-dev" in result["message"]

        # Verify the account was added
        assert mock_config_manager.add_account.called

    @pytest.mark.asyncio
    async def test_add_account_allows_public_urls(self):
        """Test that add_account allows public URLs by default"""

        # Mock the managers
        mock_config_manager = MagicMock()
        mock_account_manager = MagicMock()
        mock_account_manager.test_account.return_value = {
            "connected": True,
            "calendars": ["Calendar1"],
        }

        accounts._managers["config_manager"] = mock_config_manager
        accounts._managers["account_manager"] = mock_account_manager

        # Mock DNS resolution to return public IP
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            # Mock public IP resolution
            mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]

            result = await accounts.add_account(
                alias="public-caldav",
                url="https://caldav.example.com/dav",
                username="user",
                password="pass",
            )

            assert result["success"] is True
            assert "public-caldav" in result["message"]
            assert mock_config_manager.add_account.called

    @pytest.mark.asyncio
    async def test_add_account_validates_url_format(self):
        """Test that add_account validates URL format"""

        accounts._managers["config_manager"] = MagicMock()
        accounts._managers["account_manager"] = MagicMock()

        # Invalid URL format
        result = await accounts.add_account(
            alias="bad-url", url="not-a-url", username="user", password="pass"
        )

        assert result["success"] is False
        assert "Invalid URL format" in result["error"]

        # HTTP instead of HTTPS
        result = await accounts.add_account(
            alias="http-url",
            url="http://example.com/caldav",
            username="user",
            password="pass",
        )

        assert result["success"] is False
        assert (
            "Invalid URL format" in result["error"]
            or "Must be a valid HTTPS URL" in result["error"]
        )
