from __future__ import annotations

import sqlite3

import pytest

from whispertome.security.vault import SecretVault, VaultLocked


def _vault(tmp_path):
    return SecretVault(tmp_path / "secrets.sqlite")


def test_create_unlock_and_roundtrip(tmp_path):
    v = _vault(tmp_path)
    assert v.is_initialized() is False
    v.create("hunter2")
    assert v.is_initialized() is True
    assert v.is_unlocked is True

    v.set_secret("OPENAI_API_KEY", "sk-secret-123")
    assert v.get_secret("OPENAI_API_KEY") == "sk-secret-123"
    assert v.names() == ["OPENAI_API_KEY"]


def test_unlock_with_correct_and_wrong_password(tmp_path):
    _vault(tmp_path).create("correct-horse")
    # Fresh instance simulates a new process.
    v2 = _vault(tmp_path)
    assert v2.is_unlocked is False
    assert v2.unlock("wrong") is False
    assert v2.is_unlocked is False
    assert v2.unlock("correct-horse") is True
    assert v2.is_unlocked is True


def test_locked_operations_raise(tmp_path):
    _vault(tmp_path).create("pw")
    v2 = _vault(tmp_path)  # not unlocked
    with pytest.raises(VaultLocked):
        v2.get_secret("OPENAI_API_KEY")
    with pytest.raises(VaultLocked):
        v2.set_secret("X", "y")


def test_secret_is_encrypted_at_rest(tmp_path):
    v = _vault(tmp_path)
    v.create("pw")
    v.set_secret("OPENAI_API_KEY", "sk-plaintext-marker")
    raw = (tmp_path / "secrets.sqlite").read_bytes()
    assert b"sk-plaintext-marker" not in raw  # never stored in the clear
    assert b"pw" not in raw                    # password not stored


def test_same_value_gets_unique_nonce(tmp_path):
    v = _vault(tmp_path)
    v.create("pw")
    v.set_secret("A", "same-value")
    v.set_secret("B", "same-value")
    with sqlite3.connect(tmp_path / "secrets.sqlite") as conn:
        a = conn.execute("SELECT nonce, ciphertext FROM secrets WHERE name='A'").fetchone()
        b = conn.execute("SELECT nonce, ciphertext FROM secrets WHERE name='B'").fetchone()
    assert a[0] != b[0]            # distinct nonces
    assert a[1] != b[1]            # so identical plaintext encrypts differently


def test_tamper_detection(tmp_path):
    v = _vault(tmp_path)
    v.create("pw")
    v.set_secret("K", "value")
    with sqlite3.connect(tmp_path / "secrets.sqlite") as conn:
        conn.execute("UPDATE secrets SET ciphertext = ? WHERE name='K'", (b"\x00\x00\x00\x00\x00",))
        conn.commit()
    with pytest.raises(ValueError):
        v.get_secret("K")


def test_change_password_rekeys(tmp_path):
    v = _vault(tmp_path)
    v.create("old-pw")
    v.set_secret("OPENAI_API_KEY", "sk-keep-me")
    assert v.change_password("old-pw", "new-pw") is True

    v2 = _vault(tmp_path)
    assert v2.unlock("old-pw") is False
    assert v2.unlock("new-pw") is True
    assert v2.get_secret("OPENAI_API_KEY") == "sk-keep-me"


def test_change_password_rejects_wrong_old(tmp_path):
    v = _vault(tmp_path)
    v.create("real")
    assert v.change_password("nope", "whatever") is False


def test_create_twice_refused(tmp_path):
    v = _vault(tmp_path)
    v.create("pw")
    with pytest.raises(RuntimeError):
        _vault(tmp_path).create("pw2")
