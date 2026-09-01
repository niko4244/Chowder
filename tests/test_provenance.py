import os

import pytest

from chowder.provenance import sha256_directory


def test_directory_digest_is_path_independent_and_content_sensitive(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "nested").mkdir(parents=True)
        (root / "adapter_config.json").write_text('{"r":16}\n')
        (root / "nested" / "weights.bin").write_bytes(b"weights")

    assert sha256_directory(left) == sha256_directory(right)
    before = sha256_directory(left)
    (left / "nested" / "weights.bin").write_bytes(b"changed")
    assert sha256_directory(left) != before


def test_directory_digest_includes_relative_filenames(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "one.bin").write_bytes(b"same")
    (b / "two.bin").write_bytes(b"same")
    assert sha256_directory(a) != sha256_directory(b)


def test_directory_digest_rejects_symlinks(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    link = root / "linked.bin"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable on this platform")
    with pytest.raises(ValueError, match="symlink"):
        sha256_directory(root)
