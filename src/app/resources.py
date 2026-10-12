"""Load maintained prompt and SQL assets from the source tree."""

from pathlib import Path
import re


_ASSET_ROOT = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    return (_ASSET_ROOT / "prompts" / name).read_text(encoding="utf-8").strip()


def load_sql(name: str) -> str:
    return (_ASSET_ROOT / "sql" / name).read_text(encoding="utf-8").strip()


def load_sql_section(name: str, section: str) -> str:
    sections = re.split(r"(?m)^-- name: ([\w-]+)\s*$", load_sql(name))
    for index in range(1, len(sections), 2):
        if sections[index] == section:
            return sections[index + 1].strip()
    raise KeyError(f"SQL section {section!r} not found in {name!r}.")
