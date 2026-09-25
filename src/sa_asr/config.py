"""Config loading and the language registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LANGUAGES_FILE = PROJECT_ROOT / "configs" / "languages.yaml"


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    whisper_token: str
    whisper_native: bool
    mms_adapter: str | None
    # Excluded from hashing so Language can be used as a dict key.
    corpora: dict[str, str] = field(default_factory=dict, hash=False, compare=False)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_languages(path: str | Path = DEFAULT_LANGUAGES_FILE) -> dict[str, Language]:
    raw = load_yaml(path)["languages"]
    langs = {
        code: Language(
            code=code,
            name=spec["name"],
            whisper_token=spec["whisper_token"],
            whisper_native=bool(spec.get("whisper_native", False)),
            mms_adapter=spec.get("mms_adapter"),
            corpora=dict(spec.get("corpora") or {}),
        )
        for code, spec in raw.items()
    }
    validate_languages(langs)
    return langs


def validate_languages(langs: dict[str, Language]) -> None:
    """Proxy tokens must be unique, otherwise the model cannot tell languages apart."""
    seen: dict[str, str] = {}
    for lang in langs.values():
        if lang.whisper_token in seen:
            raise ValueError(
                f"whisper_token '{lang.whisper_token}' is used by both "
                f"'{seen[lang.whisper_token]}' and '{lang.code}'"
            )
        seen[lang.whisper_token] = lang.code


def select_languages(langs: dict[str, Language], codes: str | list[str] | None) -> list[Language]:
    """Resolve codes against the registry. None, "all" or ["all"] = every language."""
    if isinstance(codes, str):
        codes = [codes]
    if not codes or codes == ["all"]:
        return list(langs.values())
    unknown = [c for c in codes if c not in langs]
    if unknown:
        raise ValueError(f"Unknown language codes {unknown}; known: {sorted(langs)}")
    return [langs[c] for c in codes]
