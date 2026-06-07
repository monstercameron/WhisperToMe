from __future__ import annotations

import os

import pytest

from whispertome import onboarding as ob
from whispertome.config import load_config
from whispertome.security.vault import SecretVault, default_vault_path

# Onboarding writes os.environ directly; delenv the full set so monkeypatch restores it and the
# tests stay order-independent.
_VARS = (
    "OPENAI_API_KEY", "openai", "CEREBRAS_API_KEY", "cerebras",
    "WHISPERTOME_LLM_PROVIDER", "WHISPERTOME_USER_NAME",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)


def test_write_env_merges_and_preserves(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# keep me\nOPENAI_MODEL=gpt-5.4\nWHISPERTOME_LLM_PROVIDER=openai\n", "utf-8")
    ob._write_env(tmp_path, {"WHISPERTOME_LLM_PROVIDER": "cerebras", "WHISPERTOME_USER_NAME": "Ada"})
    text = env.read_text(encoding="utf-8")
    assert "# keep me" in text
    assert "OPENAI_MODEL=gpt-5.4" in text
    assert "WHISPERTOME_LLM_PROVIDER=cerebras" in text       # updated in place
    assert text.count("WHISPERTOME_LLM_PROVIDER=") == 1
    assert "WHISPERTOME_USER_NAME=Ada" in text                # appended
    assert os.environ["WHISPERTOME_LLM_PROVIDER"] == "cerebras"


def test_scrub_env_removes_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-secret\nOPENAI_MODEL=gpt-5.4\n", "utf-8")
    os.environ["OPENAI_API_KEY"] = "sk-secret"
    ob._scrub_env(tmp_path, ("OPENAI_API_KEY",))
    text = env.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in text
    assert "OPENAI_MODEL=gpt-5.4" in text
    assert "OPENAI_API_KEY" not in os.environ


def test_persist_setup_vaults_key_not_plaintext(tmp_path):
    vault = SecretVault(default_vault_path(tmp_path))
    ob._persist_setup(tmp_path, vault, name="Ada", provider="cerebras",
                      api_key="csk-xyz", password="pw")
    # Key lives only in the vault (encrypted), not in .env.
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "csk-xyz" not in env_text
    assert "WHISPERTOME_USER_NAME=Ada" in env_text
    assert "WHISPERTOME_LLM_PROVIDER=cerebras" in env_text
    assert vault.get_secret("CEREBRAS_API_KEY") == "csk-xyz"
    # config picks the key up from the injected environment.
    config = load_config(tmp_path, require_openai_key=True)
    assert config.user_name == "Ada"
    assert config.cerebras.api_key == "csk-xyz"


def test_ensure_credentials_ready_legacy_plaintext(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-legacy\n", "utf-8")
    # No vault, but a plaintext key exists -> ready, no prompt.
    assert ob.ensure_credentials_ready(tmp_path, gui=False) is True


def test_ensure_credentials_ready_unlocks_vault(tmp_path, monkeypatch):
    vault = SecretVault(default_vault_path(tmp_path))
    vault.create("s3cret")
    vault.set_secret("OPENAI_API_KEY", "sk-vaulted")
    vault.lock()
    monkeypatch.setattr("getpass.getpass", lambda *_a, **_k: "s3cret")

    assert ob.ensure_credentials_ready(tmp_path, gui=False) is True
    assert os.environ["OPENAI_API_KEY"] == "sk-vaulted"  # injected after unlock


def test_ensure_credentials_ready_wrong_password(tmp_path, monkeypatch):
    vault = SecretVault(default_vault_path(tmp_path))
    vault.create("right")
    vault.set_secret("OPENAI_API_KEY", "sk-vaulted")
    monkeypatch.setattr("getpass.getpass", lambda *_a, **_k: "wrong")

    assert ob.ensure_credentials_ready(tmp_path, gui=False) is False
    assert os.environ.get("OPENAI_API_KEY") != "sk-vaulted"


def test_inherited_env_skips_unlock_prompt(tmp_path, monkeypatch):
    # Desktop host unlocks once and injects the key; the spawned child inherits it and must NOT
    # prompt again even though the vault file exists.
    vault = SecretVault(default_vault_path(tmp_path))
    vault.create("pw")
    vault.set_secret("OPENAI_API_KEY", "sk-vaulted")

    def _boom(*_a, **_k):
        raise AssertionError("must not prompt when the key is already in the environment")

    monkeypatch.setattr("getpass.getpass", _boom)
    os.environ["OPENAI_API_KEY"] = "sk-inherited"  # as if inherited from the parent process
    assert ob.ensure_credentials_ready(tmp_path, gui=False) is True


def test_secure_existing_env_migrates(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-migrate\nOPENAI_MODEL=gpt-5.4\n", "utf-8")
    monkeypatch.setattr("getpass.getpass", lambda *_a, **_k: "newpw")

    assert ob.secure_existing_env(tmp_path, gui=False) is True

    # Plaintext key removed from .env, other config preserved.
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "sk-migrate" not in env_text
    assert "OPENAI_MODEL=gpt-5.4" in env_text
    # Key now decryptable from the vault.
    vault = SecretVault(default_vault_path(tmp_path))
    assert vault.unlock("newpw") is True
    assert vault.get_secret("OPENAI_API_KEY") == "sk-migrate"
