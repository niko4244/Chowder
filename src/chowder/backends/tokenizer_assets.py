"""Small local tokenizer assets that determine Gen-1 chat rendering."""

from __future__ import annotations

import hashlib
from pathlib import Path


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
)


def tokenizer_asset_digests(directory: Path) -> dict[str, str]:
    """Hash tokenizer configuration/data only; never touch model weight shards."""
    directory = Path(directory)
    if not directory.is_dir() or not (directory / "tokenizer.json").is_file():
        raise ValueError(f"local tokenizer.json is required at {directory}")
    paths = [directory / name for name in _TOKENIZER_FILES]
    paths.extend(directory.glob("tokenization_*.py"))
    templates = directory / "chat_templates"
    if templates.is_dir():
        paths.extend(templates.rglob("*.jinja*"))
    result: dict[str, str] = {}
    for path in sorted(paths):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[path.relative_to(directory).as_posix()] = digest.hexdigest()
    return result
