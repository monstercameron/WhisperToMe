"""First-run onboarding + credential unlock.

Collects the minimal config a new user needs (name, OpenAI/Cerebras key, and a password) and stores
it securely: the API key is encrypted in a password-protected SQLite vault (see security/vault.py);
the non-secret name/provider go to the project `.env`. On later launches the vault is unlocked with
the password and the decrypted key is injected into the process environment so the rest of the app
reads it transparently. Stdlib only so it works inside the frozen exe.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from whispertome.security.vault import SecretVault, default_vault_path

LOGGER = logging.getLogger(__name__)

_NAME_KEY = "WHISPERTOME_USER_NAME"
_PROVIDER_KEY = "WHISPERTOME_LLM_PROVIDER"
_PROVIDERS = ("openai", "cerebras")
# Vault entry name per provider; config also accepts these via _env.
SECRET_KEY_NAMES = {"openai": "OPENAI_API_KEY", "cerebras": "CEREBRAS_API_KEY"}
_LEGACY_ALIASES = {"openai": ("OPENAI_API_KEY", "openai"), "cerebras": ("CEREBRAS_API_KEY", "cerebras")}


# -- .env helpers -----------------------------------------------------------

def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip()
    return values


def _write_env(project_root: Path, updates: dict[str, str]) -> None:
    """Merge updates into .env preserving existing lines/comments/order; mirror to os.environ."""
    path = project_root / ".env"
    remaining = dict(updates)
    lines: list[str] = []
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in remaining:
                    lines.append(f"{key}={remaining.pop(key)}")
                    continue
            lines.append(raw)
    for key, val in remaining.items():
        lines.append(f"{key}={val}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for key, val in updates.items():
        os.environ[key] = val


def _scrub_env(project_root: Path, keys: tuple[str, ...]) -> None:
    """Remove the given keys from .env (used after migrating plaintext secrets into the vault)."""
    path = project_root / ".env"
    if not path.exists():
        return
    kept: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            if stripped.split("=", 1)[0].strip() in keys:
                continue
        kept.append(raw)
    path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    for key in keys:
        os.environ.pop(key, None)


def _configured_provider(env: dict[str, str]) -> str:
    provider = (env.get(_PROVIDER_KEY) or os.environ.get(_PROVIDER_KEY) or "openai").strip().lower()
    return provider if provider in _PROVIDERS else "openai"


def _plaintext_key_present(env: dict[str, str], provider: str) -> bool:
    return any(bool(env.get(n) or os.environ.get(n)) for n in _LEGACY_ALIASES[provider])


# -- secret injection -------------------------------------------------------

def inject_secrets(vault: SecretVault) -> None:
    """Push the vault's decrypted secrets into os.environ for config to read (requires unlock)."""
    for name in vault.names():
        value = vault.get_secret(name)
        if value is not None:
            os.environ[name] = value


def _persist_setup(project_root: Path, vault: SecretVault, *, name: str, provider: str,
                   api_key: str, password: str) -> None:
    vault.create(password)  # initializes + unlocks
    key_name = SECRET_KEY_NAMES[provider]
    vault.set_secret(key_name, api_key)
    _write_env(project_root, {_NAME_KEY: name, _PROVIDER_KEY: provider})  # non-secrets only
    os.environ[key_name] = api_key
    LOGGER.info("onboarding_saved provider=%s name=%s vault=secured", provider, name)


# -- interactive flows ------------------------------------------------------

def _run_gui_setup(project_root: Path, vault: SecretVault) -> bool:
    import tkinter as tk
    from tkinter import messagebox, ttk

    state = {"done": False}
    root = tk.Tk()
    root.title("WhisperToMe — First-time setup")
    root.geometry("460x440")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Welcome to WhisperToMe", font=("Segoe UI", 13, "bold")).pack(anchor="w")
    ttk.Label(frame, text="Your API key is encrypted on disk and unlocked with a password.",
              foreground="#555", wraplength=410).pack(anchor="w", pady=(0, 12))

    def field(label: str, *, secret: bool = False) -> tk.StringVar:
        ttk.Label(frame, text=label).pack(anchor="w")
        var = tk.StringVar()
        ttk.Entry(frame, textvariable=var, width=48, show="•" if secret else "").pack(
            anchor="w", pady=(2, 10)
        )
        return var

    name_var = field("Your name")

    ttk.Label(frame, text="LLM provider").pack(anchor="w")
    provider_var = tk.StringVar(value="openai")
    row = ttk.Frame(frame)
    row.pack(anchor="w", pady=(2, 10))
    ttk.Radiobutton(row, text="OpenAI", variable=provider_var, value="openai").pack(side="left")
    ttk.Radiobutton(row, text="Cerebras", variable=provider_var, value="cerebras").pack(
        side="left", padx=(12, 0)
    )

    key_var = field("API key", secret=True)
    pw_var = field("Create a password", secret=True)
    pw2_var = field("Confirm password", secret=True)

    def submit() -> None:
        name, api_key = name_var.get().strip(), key_var.get().strip()
        pw, pw2 = pw_var.get(), pw2_var.get()
        if not name or not api_key:
            messagebox.showwarning("WhisperToMe", "Please enter your name and an API key.")
            return
        if not pw:
            messagebox.showwarning("WhisperToMe", "Please choose a password.")
            return
        if pw != pw2:
            messagebox.showwarning("WhisperToMe", "Passwords do not match.")
            return
        _persist_setup(project_root, vault, name=name, provider=provider_var.get(),
                       api_key=api_key, password=pw)
        state["done"] = True
        root.destroy()

    buttons = ttk.Frame(frame)
    buttons.pack(anchor="e", side="bottom")
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right", padx=(8, 0))
    ttk.Button(buttons, text="Save", command=submit).pack(side="right")
    root.bind("<Return>", lambda _e: submit())
    root.mainloop()
    return state["done"]


def _run_console_setup(project_root: Path, vault: SecretVault) -> bool:
    from getpass import getpass

    print("\nWhisperToMe first-time setup\n----------------------------")
    try:
        name = input("Your name: ").strip()
        provider = ""
        while provider not in _PROVIDERS:
            provider = (input("LLM provider [openai/cerebras] (openai): ").strip().lower()
                        or "openai")
        api_key = input(f"{provider} API key: ").strip()
        pw = getpass("Create a password: ")
        pw2 = getpass("Confirm password: ")
    except (EOFError, OSError):
        return False
    if not name or not api_key or not pw:
        print("Name, API key, and password are required. Setup cancelled.")
        return False
    if pw != pw2:
        print("Passwords do not match. Setup cancelled.")
        return False
    _persist_setup(project_root, vault, name=name, provider=provider, api_key=api_key, password=pw)
    print("Saved — your key is encrypted in the vault.\n")
    return True


def run_onboarding(project_root: Path | None, vault: SecretVault, *, gui: bool = True) -> bool:
    root = project_root or Path.cwd()
    root.mkdir(parents=True, exist_ok=True)
    if gui:
        try:
            return _run_gui_setup(root, vault)
        except Exception as exc:  # noqa: BLE001 — no display / Tk missing -> console
            LOGGER.info("onboarding_gui_unavailable error=%s", exc)
    return _run_console_setup(root, vault)


def _prompt_password_gui(*, error: str | None = None) -> str | None:
    import tkinter as tk
    from tkinter import ttk

    state: dict[str, str | None] = {"pw": None}
    root = tk.Tk()
    root.title("WhisperToMe — Unlock")
    root.geometry("360x170")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Enter your password to unlock WhisperToMe.").pack(anchor="w")
    if error:
        ttk.Label(frame, text=error, foreground="#c0392b").pack(anchor="w", pady=(4, 0))
    pw_var = tk.StringVar()
    entry = ttk.Entry(frame, textvariable=pw_var, width=40, show="•")
    entry.pack(anchor="w", pady=(10, 12))

    def submit() -> None:
        state["pw"] = pw_var.get()
        root.destroy()

    buttons = ttk.Frame(frame)
    buttons.pack(anchor="e", side="bottom")
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right", padx=(8, 0))
    ttk.Button(buttons, text="Unlock", command=submit).pack(side="right")
    root.bind("<Return>", lambda _e: submit())
    entry.focus_set()
    root.mainloop()
    return state["pw"]


def unlock_interactive(vault: SecretVault, *, gui: bool = True, attempts: int = 3) -> bool:
    """Prompt for the password (up to `attempts` times); on success inject secrets to os.environ."""
    error: str | None = None
    for _ in range(max(1, attempts)):
        if gui:
            try:
                password = _prompt_password_gui(error=error)
            except Exception as exc:  # noqa: BLE001 — fall back to console
                LOGGER.info("unlock_gui_unavailable error=%s", exc)
                gui = False
                password = None
            if not gui:
                continue
        else:
            from getpass import getpass
            try:
                password = getpass("WhisperToMe password: ")
            except (EOFError, OSError):
                return False
        if password is None:
            return False  # cancelled
        if vault.unlock(password):
            inject_secrets(vault)
            return True
        error = "Incorrect password — try again."
        LOGGER.info("unlock_rejected")
    return False


# -- orchestration ----------------------------------------------------------

def ensure_credentials_ready(project_root: Path | None, *, gui: bool = True) -> bool:
    """Make an LLM key available before config load. Returns False if the user cancels.

    - Vault already set up -> prompt to unlock and inject the decrypted key.
    - No vault but a plaintext key exists in .env (legacy) -> leave it; the app still works.
    - Otherwise -> first-run onboarding (creates the vault).
    """
    root = project_root or Path.cwd()
    env = _read_env_file(root / ".env")
    provider = _configured_provider(env)
    # Already available — legacy plaintext .env, or injected into os.environ by a parent process
    # (the desktop host unlocks once, then spawns the voice-loop child which inherits the env).
    # This short-circuit avoids a second password prompt in the child.
    if _plaintext_key_present(env, provider):
        return True
    vault = SecretVault(default_vault_path(root))
    if vault.is_initialized():
        return unlock_interactive(vault, gui=gui)
    return run_onboarding(root, vault, gui=gui)


def secure_existing_env(project_root: Path | None, *, gui: bool = True) -> bool:
    """Migration: move plaintext API keys from .env into a new password-protected vault."""
    root = project_root or Path.cwd()
    vault = SecretVault(default_vault_path(root))
    if vault.is_initialized():
        print("A vault already exists. Use the app's unlock prompt, or delete artifacts/secrets.sqlite.")
        return False
    env = _read_env_file(root / ".env")
    found = {}
    for provider, aliases in _LEGACY_ALIASES.items():
        for alias in aliases:
            val = env.get(alias) or os.environ.get(alias)
            if val:
                found[SECRET_KEY_NAMES[provider]] = val
                break
    if not found:
        print("No plaintext API keys found in .env to secure.")
        return False
    from getpass import getpass
    try:
        pw = getpass("Create a vault password: ")
        pw2 = getpass("Confirm password: ")
    except (EOFError, OSError):
        return False
    if not pw or pw != pw2:
        print("Passwords missing or mismatched. Aborted.")
        return False
    vault.create(pw)
    for name, value in found.items():
        vault.set_secret(name, value)
    # Scrub plaintext keys (canonical + lowercase aliases) from .env.
    scrub = tuple({a for aliases in _LEGACY_ALIASES.values() for a in aliases})
    _scrub_env(root, scrub)
    print(f"Secured {len(found)} key(s) into the vault and removed them from .env.")
    return True
