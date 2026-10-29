"""Read-only, Windows handle-locked snapshot of original guard.v3 evidence.

Original F: files are never opened for writing. A Windows share-read-only
handle rejects concurrent writers and deletion while a file is copied. All
directory handles are retained during enumeration, and the original inventory
and bytes are checked again after the independent scorer finishes. The scorer
reads only the private temporary copy, never a mutable original path.

This prevents a scorer from hashing one open and parsing a different open.
It does not turn mutable local evidence into a cryptographic execution witness.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import ctypes
from ctypes import wintypes
import hashlib
import os
from pathlib import Path
import stat
import tempfile


_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_TOTAL = 512 * 1024 * 1024
_MAX_SINGLE = 64 * 1024 * 1024


def _tree(root: Path) -> tuple[list[Path], list[Path]]:
    """Enumerate ordinary entries without following any filesystem redirect."""
    directories, files = [], []
    pending = [root]
    while pending:
        path = pending.pop()
        metadata = path.lstat()
        if (stat.S_ISLNK(metadata.st_mode)
                or getattr(metadata, "st_file_attributes", 0) & _REPARSE):
            raise ValueError("original guard.v3 evidence reparse point: " + str(path))
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(path)
            pending.extend(path.iterdir())
        elif stat.S_ISREG(metadata.st_mode):
            files.append(path)
        else:
            raise ValueError("original guard.v3 evidence nonregular entry: " + str(path))
    return sorted(directories), sorted(files)


class _Info(ctypes.Structure):
    _fields_ = [
        ("dwFileAttributes", wintypes.DWORD),
        ("ftCreationTime", wintypes.FILETIME),
        ("ftLastAccessTime", wintypes.FILETIME),
        ("ftLastWriteTime", wintypes.FILETIME),
        ("dwVolumeSerialNumber", wintypes.DWORD),
        ("nFileSizeHigh", wintypes.DWORD),
        ("nFileSizeLow", wintypes.DWORD),
        ("nNumberOfLinks", wintypes.DWORD),
        ("nFileIndexHigh", wintypes.DWORD),
        ("nFileIndexLow", wintypes.DWORD),
    ]


@contextmanager
def _locked_read(path: Path, *, directory: bool):
    if os.name != "nt":
        raise ValueError("original guard.v3 stable snapshot requires Windows file-share locks")
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Info)]
    kernel.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    flags = 0x00200000 | (0x02000000 if directory else 0)  # OPEN_REPARSE_POINT, BACKUP_SEMANTICS
    handle = kernel.CreateFileW(str(path), 0x80000000, 0x00000001, None, 3, flags, None)
    invalid = ctypes.c_void_p(-1).value
    if handle is None or int(handle) == invalid:
        raise ValueError("original guard.v3 source cannot be exclusively read-locked: " + str(path))
    try:
        info = _Info()
        if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise ValueError("original guard.v3 locked handle identity unavailable: " + str(path))
        if info.dwFileAttributes & _REPARSE:
            raise ValueError("original guard.v3 locked source is reparse: " + str(path))
        if info.nNumberOfLinks != 1:
            raise ValueError("original guard.v3 source has hardlink aliases: " + str(path))
        current = path.lstat()
        opened_inode = (info.nFileIndexHigh << 32) | info.nFileIndexLow
        if (stat.S_ISDIR(current.st_mode) != directory
                or getattr(current, "st_ino", opened_inode) != opened_inode):
            raise ValueError("original guard.v3 path changed during handle acquisition: " + str(path))
        if directory:
            yield None
        else:
            fd = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
            handle = None  # The fd now owns the Win32 handle.
            with os.fdopen(fd, "rb") as stream:
                yield stream
    finally:
        if handle is not None:
            kernel.CloseHandle(handle)


def _locked_inventory(root: Path, locks: ExitStack):
    """Open/lock each directory BEFORE listing its children, then lock files."""
    directories = []
    sources = {}
    pending = [root]
    while pending:
        path = pending.pop()
        metadata = path.lstat()
        if (stat.S_ISLNK(metadata.st_mode)
                or getattr(metadata, "st_file_attributes", 0) & _REPARSE):
            raise ValueError("original guard.v3 evidence reparse point: " + str(path))
        if stat.S_ISDIR(metadata.st_mode):
            locks.enter_context(_locked_read(path, directory=True))
            directories.append(path)
            pending.extend(path.iterdir())
        elif stat.S_ISREG(metadata.st_mode):
            sources[path] = locks.enter_context(_locked_read(path, directory=False))
        else:
            raise ValueError("original guard.v3 evidence nonregular entry: " + str(path))
    return sorted(directories), sources


def _digest_tree(root: Path, expected: dict[str, tuple[int, str]],
                 expected_dirs: set[str]) -> None:
    with ExitStack() as locks:
        directories, sources = _locked_inventory(root, locks)
        if ({p.relative_to(root).as_posix() for p in sources} != set(expected)
                or {p.relative_to(root).as_posix() for p in directories} != expected_dirs):
            raise ValueError("original guard.v3 evidence inventory changed during scoring")
        for path, source in sources.items():
            relative = path.relative_to(root).as_posix()
            h = hashlib.sha256()
            length = 0
            for block in iter(lambda: source.read(1024 * 1024), b""):
                length += len(block)
                h.update(block)
            if (length, h.hexdigest()) != expected[relative]:
                raise ValueError("original guard.v3 evidence bytes changed during scoring: " + relative)


@contextmanager
def stable_snapshot(root: Path, workspace: Path):
    """Yield a private immutable-to-the-scorer copy and verify F: afterwards."""
    with tempfile.TemporaryDirectory(prefix=".reap-guardv3-readonly-", dir=workspace) as temp:
        target = Path(temp)
        expected: dict[str, tuple[int, str]] = {}
        total = 0
        with ExitStack() as locks:
            # Each parent is already locked before its children are listed.
            directories, sources = _locked_inventory(root, locks)
            expected_dirs = {p.relative_to(root).as_posix() for p in directories}
            for path, source in sorted(sources.items()):
                relative = path.relative_to(root)
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                length = 0
                with destination.open("xb") as out:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        length += len(block)
                        if length > _MAX_SINGLE:
                            raise ValueError("original guard.v3 individual evidence file exceeds snapshot limit")
                        h.update(block)
                        out.write(block)
                expected[relative.as_posix()] = (length, h.hexdigest())
                total += length
                if total > _MAX_TOTAL:
                    raise ValueError("original guard.v3 evidence exceeds bounded snapshot limit")
            new_directories, new_files = _tree(root)
            if (set(new_directories) != set(directories)
                    or set(new_files) != set(sources)):
                raise ValueError("original guard.v3 evidence inventory changed during snapshot")
        # A temporary directory alone is not immutable to another process
        # running as the same user. Keep *snapshot* file/directory handles
        # share-read-only throughout original WorldState scoring, too.
        with ExitStack() as snapshot_locks:
            snapshot_dirs, snapshot_sources = _locked_inventory(target, snapshot_locks)
            if ({p.relative_to(target).as_posix() for p in snapshot_sources} != set(expected)
                    or {p.relative_to(target).as_posix() for p in snapshot_dirs} != expected_dirs):
                raise ValueError("original guard.v3 private snapshot inventory mismatch")
            _digest_tree(target, expected, expected_dirs)
            try:
                yield target
            finally:
                _digest_tree(target, expected, expected_dirs)
                _digest_tree(root, expected, expected_dirs)
