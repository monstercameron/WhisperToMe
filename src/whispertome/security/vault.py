"""Password-protected secret vault (SQLite, stdlib-only crypto).

Stores credentials (e.g. API keys) encrypted at rest, gated by a user password. This is *basic*
local-at-rest protection: it stops a casual reader of the .env / disk from lifting plaintext keys,
and requires the password to unlock. It is not a defence against a determined attacker who controls
the machine while the app runs.

Design (all stdlib — works in the frozen exe; no `cryptography` dependency):
  - KDF: scrypt(password, salt) -> 96 bytes split into enc_key | mac_key | verifier (memory-hard).
  - Password check: the stored `verifier` segment is compared in constant time on unlock — this is
    the "password hash".
  - Per-secret encryption: a unique random 16-byte nonce; keystream = HKDF-Expand(enc_key, nonce);
    ciphertext = plaintext XOR keystream; then encrypt-then-MAC: tag = HMAC-SHA256(mac_key,
    nonce||ciphertext), verified in constant time before decryption (tamper detection).
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_bytes
from threading import RLock
from typing import Iterator

LOGGER = logging.getLogger(__name__)

# scrypt cost. n*r*128 ~= 16 MB of work per guess — enough to slow offline brute force, fast enough
# for an interactive unlock. Stored alongside the salt so params can evolve without breaking old vaults.
_SCRYPT_N = 1 << 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_MAXMEM = 64 * 1024 * 1024
_DKLEN = 96  # 32 enc + 32 mac + 32 verifier
_SALT_BYTES = 16
_NONCE_BYTES = 16


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> tuple[bytes, bytes, bytes]:
    dk = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=_DKLEN, maxmem=_SCRYPT_MAXMEM
    )
    return dk[0:32], dk[32:64], dk[64:96]  # enc_key, mac_key, verifier


def _hkdf_expand(key: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 HKDF-Expand with HMAC-SHA256 — a standard way to stretch a key into a keystream."""
    out = bytearray()
    block = b""
    counter = 1
    while len(out) < length:
        block = hmac.new(key, block + info + bytes([counter]), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def _xor(data: bytes, keystream: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, keystream))


def default_vault_path(project_root: Path) -> Path:
    return project_root / "artifacts" / "secrets.sqlite"


class VaultLocked(RuntimeError):
    """Raised when a secret operation is attempted before unlock()."""


class SecretVault:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = RLock()
        self._enc_key: bytes | None = None
        self._mac_key: bytes | None = None
        with self._lock:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connection() as conn:
                self._init_schema(conn)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vault_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                salt BLOB NOT NULL, n INTEGER NOT NULL, r INTEGER NOT NULL, p INTEGER NOT NULL,
                verifier BLOB NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS secrets (
                name TEXT PRIMARY KEY,
                nonce BLOB NOT NULL, ciphertext BLOB NOT NULL, tag BLOB NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

    # -- lifecycle ----------------------------------------------------------

    def is_initialized(self) -> bool:
        with self._lock, self._connection() as conn:
            row = conn.execute("SELECT 1 FROM vault_meta WHERE id = 1").fetchone()
        return row is not None

    @property
    def is_unlocked(self) -> bool:
        return self._enc_key is not None and self._mac_key is not None

    def create(self, password: str) -> None:
        """Initialize the vault with a password. Unlocks it for immediate use."""
        if not password:
            raise ValueError("password is required")
        with self._lock:
            if self.is_initialized():
                raise RuntimeError("vault already initialized")
            salt = token_bytes(_SALT_BYTES)
            enc_key, mac_key, verifier = _derive(password, salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
            with self._connection() as conn:
                conn.execute(
                    "INSERT INTO vault_meta (id, salt, n, r, p, verifier, created_at) "
                    "VALUES (1, ?, ?, ?, ?, ?, ?)",
                    (salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P, verifier, _now_iso()),
                )
            self._enc_key, self._mac_key = enc_key, mac_key
            LOGGER.info("vault_created path=%s", self._db_path)

    def unlock(self, password: str) -> bool:
        """Verify the password (constant-time) and cache the derived keys. False if wrong."""
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT salt, n, r, p, verifier FROM vault_meta WHERE id = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("vault is not initialized")
        salt, n, r, p, verifier = row
        enc_key, mac_key, candidate = _derive(password, salt, n, r, p)
        if not hmac.compare_digest(candidate, verifier):
            LOGGER.info("vault_unlock_failed")
            return False
        self._enc_key, self._mac_key = enc_key, mac_key
        return True

    def lock(self) -> None:
        self._enc_key = None
        self._mac_key = None

    def change_password(self, old_password: str, new_password: str) -> bool:
        """Re-key the vault: verify old, re-encrypt every secret under the new password."""
        if not new_password:
            raise ValueError("new password is required")
        with self._lock:
            if not self.unlock(old_password):
                return False
            plain = {name: self.get_secret(name) for name in self.names()}
            with self._connection() as conn:
                conn.execute("DELETE FROM vault_meta WHERE id = 1")
                conn.execute("DELETE FROM secrets")
            self.lock()
            self.create(new_password)
            for name, value in plain.items():
                if value is not None:
                    self.set_secret(name, value)
            return True

    # -- secrets ------------------------------------------------------------

    def _require_keys(self) -> tuple[bytes, bytes]:
        if self._enc_key is None or self._mac_key is None:
            raise VaultLocked("vault is locked; call unlock() first")
        return self._enc_key, self._mac_key

    def set_secret(self, name: str, value: str) -> None:
        enc_key, mac_key = self._require_keys()
        name = name.strip()
        if not name:
            raise ValueError("secret name is required")
        plaintext = value.encode("utf-8")
        nonce = token_bytes(_NONCE_BYTES)
        ciphertext = _xor(plaintext, _hkdf_expand(enc_key, nonce, len(plaintext)))
        tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT INTO secrets (name, nonce, ciphertext, tag, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
                "nonce=excluded.nonce, ciphertext=excluded.ciphertext, tag=excluded.tag, "
                "updated_at=excluded.updated_at",
                (name, nonce, ciphertext, tag, _now_iso()),
            )

    def get_secret(self, name: str) -> str | None:
        enc_key, mac_key = self._require_keys()
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT nonce, ciphertext, tag FROM secrets WHERE name = ?", (name.strip(),)
            ).fetchone()
        if row is None:
            return None
        nonce, ciphertext, tag = row
        expected = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, tag):
            raise ValueError(f"secret {name!r} failed authentication (wrong key or tampered)")
        plaintext = _xor(ciphertext, _hkdf_expand(enc_key, nonce, len(ciphertext)))
        return plaintext.decode("utf-8")

    def names(self) -> list[str]:
        with self._lock, self._connection() as conn:
            rows = conn.execute("SELECT name FROM secrets ORDER BY name").fetchall()
        return [r[0] for r in rows]
