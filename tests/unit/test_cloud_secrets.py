"""Credential handling and redaction (``design-cloud-adapter.md`` §8)."""

from __future__ import annotations

import logging

import pytest

from spectraquant.cloud.secrets import (
    REDACTION_PLACEHOLDER,
    SECRET_ENV_VARS,
    MissingCredentials,
    RedactingFilter,
    assert_no_secret,
    has_secret,
    install_redacting_handler,
    redact,
    redact_json,
    redact_mapping,
    require_credentials,
    scrub_lines,
    secret_values,
)

SECRET = "kaggle_key_value_that_is_long_enough"


def test_secret_env_vars_are_the_documented_five() -> None:
    assert set(SECRET_ENV_VARS) == {
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "HF_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "SPECTRAQUANT_ALLOW_PAID",
    }


def test_redact_replaces_the_value_and_keeps_the_name() -> None:
    env = {"KAGGLE_KEY": SECRET}

    scrubbed = redact(f"using key {SECRET} now", env)

    assert SECRET not in scrubbed
    assert scrubbed == f"using key {REDACTION_PLACEHOLDER.format(name='KAGGLE_KEY')} now"


def test_redaction_is_idempotent() -> None:
    env = {"KAGGLE_KEY": SECRET}

    once = redact(SECRET, env)
    twice = redact(once, env)

    assert once == twice


def test_short_values_and_control_flags_are_never_redacted() -> None:
    env = {"KAGGLE_USERNAME": "ab", "SPECTRAQUANT_ALLOW_PAID": "1"}

    assert secret_values(env) == {}
    assert redact("ab and 1 stay intact", env) == "ab and 1 stay intact"


def test_a_paid_flag_never_corrupts_output() -> None:
    env = {"SPECTRAQUANT_ALLOW_PAID": "1"}

    assert redact("step 1 of 1: run 1", env) == "step 1 of 1: run 1"


def test_redact_mapping_scrubs_nested_structures() -> None:
    env = {"HF_TOKEN": SECRET}
    payload = {"a": [f"x{SECRET}y"], "b": {"c": SECRET}}

    scrubbed = redact_mapping(payload, env)

    assert SECRET not in str(scrubbed)
    assert scrubbed["b"]["c"] == REDACTION_PLACEHOLDER.format(name="HF_TOKEN")


def test_redact_json_produces_valid_json() -> None:
    env = {"HF_TOKEN": SECRET}
    text = redact_json({"token": SECRET}, env)

    assert SECRET not in text
    assert "[REDACTED:HF_TOKEN]" in text


def test_has_secret_and_assert_no_secret() -> None:
    env = {"HF_TOKEN": SECRET}

    assert has_secret(f"prefix {SECRET}", env) is True
    assert has_secret("nothing here", env) is False
    with pytest.raises(PermissionError, match="HF_TOKEN"):
        assert_no_secret(SECRET, env, where="a generated notebook")


def test_require_credentials_names_the_missing_variable() -> None:
    with pytest.raises(MissingCredentials, match="KAGGLE_KEY"):
        require_credentials(("KAGGLE_USERNAME", "KAGGLE_KEY"), {"KAGGLE_USERNAME": "someone"})


def test_require_credentials_returns_values_without_logging() -> None:
    env = {"KAGGLE_USERNAME": "someone", "KAGGLE_KEY": SECRET}

    assert require_credentials(("KAGGLE_KEY",), env) == {"KAGGLE_KEY": SECRET}


def test_redacting_filter_scrubs_log_records() -> None:
    env = {"KAGGLE_KEY": SECRET}
    record = logging.LogRecord(
        name="spectraquant.cloud",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="pushing with %s",
        args=(SECRET,),
        exc_info=None,
    )

    assert RedactingFilter(env).filter(record) is True

    assert SECRET not in record.getMessage()


def test_install_redacting_handler_attaches_a_filter(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("spectraquant.cloud.adapters.test")

    handler = install_redacting_handler(logger, {"HF_TOKEN": SECRET})

    assert handler is not None


def test_scrub_lines_redacts_each_line() -> None:
    env = {"HF_TOKEN": SECRET}

    assert scrub_lines([f"a {SECRET}", "b"], env) == [
        f"a {REDACTION_PLACEHOLDER.format(name='HF_TOKEN')}",
        "b",
    ]


def test_default_environment_is_read_from_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", SECRET)

    assert secret_values() == {"HF_TOKEN": SECRET}
    assert SECRET not in redact(f"token={SECRET}")
