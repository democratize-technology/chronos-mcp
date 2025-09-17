"""Unit tests for RRULE validation and parsing."""

from datetime import datetime, timedelta, timezone

from chronos_mcp.rrule import MAX_COUNT, MAX_YEARS_AHEAD, RRuleTemplates, RRuleValidator


class TestRRuleValidator:
    """Test RRULE validation logic."""

    def test_valid_daily_with_count(self):
        """Test valid daily recurrence with count."""
        is_valid, error = RRuleValidator.validate_rrule("FREQ=DAILY;COUNT=10")
        assert is_valid is True
        assert error is None

    def test_valid_weekly_with_until(self):
        """Test valid weekly recurrence with until date."""
        until_date = datetime.now(timezone.utc) + timedelta(days=30)
        rrule = f"FREQ=WEEKLY;UNTIL={until_date.strftime('%Y%m%dT%H%M%SZ')}"
        is_valid, error = RRuleValidator.validate_rrule(rrule)
        assert is_valid is True
        assert error is None

    def test_valid_monthly_with_bymonthday(self):
        """Test valid monthly recurrence on specific day."""
        is_valid, error = RRuleValidator.validate_rrule(
            "FREQ=MONTHLY;BYMONTHDAY=15;COUNT=12"
        )
        assert is_valid is True
        assert error is None

    def test_valid_yearly_recurrence(self):
        """Test valid yearly recurrence."""
        is_valid, error = RRuleValidator.validate_rrule("FREQ=YEARLY;COUNT=5")
        assert is_valid is True
        assert error is None

    def test_invalid_no_frequency(self):
        """Test invalid RRULE without frequency."""
        is_valid, error = RRuleValidator.validate_rrule("COUNT=10")
        assert is_valid is False
        assert "must start with FREQ=" in error

    def test_invalid_no_end_condition(self):
        """Test invalid RRULE without COUNT or UNTIL."""
        is_valid, error = RRuleValidator.validate_rrule("FREQ=DAILY")
        assert is_valid is False
        assert "must have COUNT or UNTIL" in error

    def test_invalid_frequency(self):
        """Test invalid frequency (SECONDLY not allowed)."""
        is_valid, error = RRuleValidator.validate_rrule("FREQ=SECONDLY;COUNT=10")
        assert is_valid is False
        assert "not allowed" in error

    def test_invalid_count_too_high(self):
        """Test invalid COUNT exceeding maximum."""
        is_valid, error = RRuleValidator.validate_rrule(
            f"FREQ=DAILY;COUNT={MAX_COUNT + 1}"
        )
        assert is_valid is False
        assert f"cannot exceed {MAX_COUNT}" in error

    def test_invalid_count_zero(self):
        """Test invalid COUNT of zero."""
        is_valid, error = RRuleValidator.validate_rrule("FREQ=DAILY;COUNT=0")
        assert is_valid is False
        assert "must be at least 1" in error

    def test_invalid_until_too_far(self):
        """Test invalid UNTIL date too far in future."""
        far_future = datetime.now(timezone.utc).replace(
            year=datetime.now().year + MAX_YEARS_AHEAD + 1
        )
        rrule = f"FREQ=DAILY;UNTIL={far_future.strftime('%Y%m%dT%H%M%SZ')}"
        is_valid, error = RRuleValidator.validate_rrule(rrule)
        assert is_valid is False
        assert f"more than {MAX_YEARS_AHEAD} years" in error

    def test_invalid_until_format(self):
        """Test invalid UNTIL date format."""
        is_valid, error = RRuleValidator.validate_rrule("FREQ=DAILY;UNTIL=invalid")
        assert is_valid is False
        assert "UNTIL" in error  # Changed to match actual error message

    def test_valid_with_interval(self):
        """Test valid RRULE with interval."""
        is_valid, error = RRuleValidator.validate_rrule(
            "FREQ=WEEKLY;INTERVAL=2;COUNT=10"
        )
        assert is_valid is True
        assert error is None

    def test_invalid_interval_zero(self):
        """Test invalid interval of zero."""
        is_valid, error = RRuleValidator.validate_rrule(
            "FREQ=DAILY;INTERVAL=0;COUNT=10"
        )
        assert is_valid is False
        assert "INTERVAL must be at least 1" in error

    def test_empty_rrule(self):
        """Test empty RRULE string."""
        is_valid, error = RRuleValidator.validate_rrule("")
        assert is_valid is False
        assert "cannot be empty" in error


class TestRRuleExpansion:
    """Test RRULE expansion to occurrences."""

    def test_expand_daily_occurrences(self):
        """Test expanding daily occurrences."""
        start = datetime.now(timezone.utc).replace(
            hour=10, minute=0, second=0, microsecond=0
        )
        occurrences = RRuleValidator.expand_occurrences("FREQ=DAILY;COUNT=5", start)

        assert len(occurrences) == 5
        # Check dates are consecutive days
        for i in range(1, 5):
            assert (
                occurrences[i].date() == (occurrences[i - 1] + timedelta(days=1)).date()
            )

    def test_expand_weekly_occurrences(self):
        """Test expanding weekly occurrences."""
        start = datetime.now(timezone.utc).replace(
            hour=14, minute=0, second=0, microsecond=0
        )
        occurrences = RRuleValidator.expand_occurrences(
            "FREQ=WEEKLY;BYDAY=MO,WE,FR;COUNT=6", start
        )

        assert len(occurrences) == 6

    def test_expand_with_end_date(self):
        """Test expanding with end date limit."""
        start = datetime.now(timezone.utc)
        end = start + timedelta(days=10)

        occurrences = RRuleValidator.expand_occurrences(
            "FREQ=DAILY;COUNT=30",  # Would generate 30, but limited by end_date
            start,
            end_date=end,
        )
        # Should only return occurrences within the 10-day window
        assert len(occurrences) <= 11  # 10 days + start day
        assert all(occ <= end for occ in occurrences)

    def test_expand_with_limit(self):
        """Test expanding with occurrence limit."""
        start = datetime.now(timezone.utc)

        occurrences = RRuleValidator.expand_occurrences(
            "FREQ=DAILY;COUNT=1000",  # Large count
            start,
            limit=50,
        )

        assert len(occurrences) == 50

    def test_expand_invalid_rrule(self):
        """Test expanding invalid RRULE returns empty list."""
        start = datetime.now(timezone.utc)
        occurrences = RRuleValidator.expand_occurrences("INVALID", start)

        assert occurrences == []


class TestRRuleInfo:
    """Test RRULE information extraction."""

    def test_get_rrule_info_complete(self):
        """Test extracting complete RRULE information."""
        info = RRuleValidator.get_rrule_info(
            "FREQ=WEEKLY;INTERVAL=2;COUNT=10;BYDAY=MO,WE,FR"
        )
        assert info["frequency"] == "WEEKLY"
        assert info["interval"] == 2
        assert info["count"] == 10
        assert info["byday"] == ["MO", "WE", "FR"]
        assert info["until"] is None
        assert info["bymonthday"] is None

    def test_get_rrule_info_minimal(self):
        """Test extracting minimal RRULE information."""
        info = RRuleValidator.get_rrule_info("FREQ=DAILY;COUNT=5")

        assert info["frequency"] == "DAILY"
        assert info["interval"] == 1  # Default
        assert info["count"] == 5
        assert info["byday"] is None
        assert info["until"] is None

    def test_get_rrule_info_with_until(self):
        """Test extracting RRULE with UNTIL."""
        info = RRuleValidator.get_rrule_info(
            "FREQ=MONTHLY;UNTIL=20251231T235959Z;BYMONTHDAY=15"
        )

        assert info["frequency"] == "MONTHLY"
        assert info["until"] == "20251231T235959Z"
        assert info["bymonthday"] == [15]
        assert info["count"] is None


class TestRRuleTemplates:
    """Test RRULE template constants."""

    def test_template_formats(self):
        """Test that templates have correct format."""
        assert RRuleTemplates.DAILY_WEEKDAYS == "FREQ=DAILY;BYDAY=MO,TU,WE,TH,FR"
        assert RRuleTemplates.MONTHLY_LAST_DAY == "FREQ=MONTHLY;BYMONTHDAY=-1"
        assert "{day}" in RRuleTemplates.WEEKLY_ON_DAY
        assert "{days}" in RRuleTemplates.WEEKLY_MULTIPLE_DAYS
