import pytest

from memoria.utils.exceptions import SecurityError
from memoria.utils.security_audit import (
    DatabaseSecurityAuditor,
    SecureQueryBuilder,
    SecurityLevel,
    VulnerabilityType,
)


@pytest.fixture
def fresh_auditor() -> DatabaseSecurityAuditor:
    """Provide a fresh auditor instance for each test."""

    return DatabaseSecurityAuditor()


def test_audit_detects_critical_sql_injection(
    fresh_auditor: DatabaseSecurityAuditor,
) -> None:
    query = "SELECT * FROM users WHERE name = 'admin' OR 1=1; DROP TABLE users;"

    is_safe, findings = fresh_auditor.validate_query_safety(query, context="unit_test")

    assert not is_safe, "Injection payload should be reported as unsafe"
    severities = {
        (finding.vulnerability_type, finding.severity) for finding in findings
    }
    assert (
        VulnerabilityType.SQL_INJECTION,
        SecurityLevel.CRITICAL,
    ) in severities, "Critical SQL injection finding was not produced"


def test_audit_report_accumulates_findings(
    fresh_auditor: DatabaseSecurityAuditor,
) -> None:
    # Record a medium severity issue via SELECT *
    fresh_auditor.audit_query("SELECT * FROM accounts", context="report_select")

    # Record a critical DDL attempt
    fresh_auditor.audit_query("DROP TABLE accounts", context="report_drop")

    report = fresh_auditor.generate_audit_report()

    assert report.total_queries_audited == 2
    assert report.critical_count == 1
    assert report.medium_count == 1
    assert (
        report.overall_risk_score >= 30
    ), "Risk score should reflect accumulated findings"


def test_secure_query_builder_blocks_sensitive_columns(
    fresh_auditor: DatabaseSecurityAuditor,
) -> None:
    builder = SecureQueryBuilder(fresh_auditor)

    with pytest.raises(SecurityError):
        builder.build_safe_select("users", ["password"], {}, limit=5)


def test_secure_query_builder_builds_safe_insert(
    fresh_auditor: DatabaseSecurityAuditor,
) -> None:
    builder = SecureQueryBuilder(fresh_auditor)

    query, params = builder.build_safe_insert("memories", {"title": "Hello"})

    assert query == "INSERT INTO memories (title) VALUES (?)"
    assert params == ["Hello"]
