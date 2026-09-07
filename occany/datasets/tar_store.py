# Random-access reader for the per-scene tar archives.
#
# dataset_setup/build_tar_store.py stores one tar per scene, with
# members named "<scene>/<file>". They are written with tarfile.open(..., "w") -- i.e.
# UNCOMPRESSED despite the .tar.gz name -- so a member can be read by byte offset without
# inflating the stream. That is what makes them usable as a live training source: millions
# of small files collapse to one inode per scene while reads stay O(1).
#
# A gzipped archive would defeat this entirely (no seek without inflating from the start),
# so the tar is opened with mode 'r:' below, which refuses a compressed stream loudly.
import functools
import io
import math
import os
import tarfile
import time

import numpy as np

# Sidecar written next to the tar, e.g. 000000.tar.gz.idx.npz
INDEX_SUFFIX = '.idx.npz'


def scene_tar_path(root, scene_name):
    """Resolve <root>/<scene>.tar[.gz]. The backup set uses .tar.gz for an uncompressed tar."""
    for ext in ('.tar', '.tar.gz'):
        path = os.path.join(root, scene_name + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'No scene tar for {scene_name!r} under {root}')


def build_index(tar_path, index_path=None):
    """Record every member's byte offset into a sidecar index.

    Streaming over the tar reads headers only and seeks past payloads, so this costs one
    seek per member rather than a full pass over the data.
    """
    index_path = index_path or tar_path + INDEX_SUFFIX
    names, offsets, sizes = [], [], []
    # 'r:' pins uncompressed; a gzip stream raises here instead of silently being slow.
    with tarfile.open(tar_path, 'r:') as tar:
        for member in tar:
            if not member.isfile():
                continue
            names.append(member.name)
            offsets.append(member.offset_data)
            sizes.append(member.size)

    # np.savez appends .npz to a name that lacks it, so write to "<index>.tmp.npz" and
    # rename atomically -- a killed job then leaves no half-written index for a later
    # read to trust.
    tmp_path = index_path + '.tmp.npz'
    np.savez(
        tmp_path,
        names=np.array(names),                      # (M,) member names, "<scene>/<file>"
        offsets=np.array(offsets, dtype=np.int64),  # (M,) payload start byte
        sizes=np.array(sizes, dtype=np.int64),      # (M,) payload length
    )
    os.replace(tmp_path, index_path)
    return index_path, len(names)


def _pread_exact(fd, offset, size):
    """pread the full range; a single pread may return short."""
    chunks, got = [], 0
    while got < size:
        chunk = os.pread(fd, size - got, offset + got)
        if not chunk:
            raise EOFError(f'Short read at offset {offset + got} (wanted {size}, got {got})')
        chunks.append(chunk)
        got += len(chunk)
    return b''.join(chunks)


class TarSceneStore:
    """One scene tar plus its offset index, safe across DataLoader fork."""

    def __init__(self, tar_path):
        self.tar_path = tar_path
        # set first: __del__ runs even if __init__ raises below
        self._fd = None
        self._pid = None
        index_path = tar_path + INDEX_SUFFIX
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f'Missing tar index {index_path}. Build it with '
                f'dataset_setup/build_tar_store.py before training with use_tar=True.'
            )
        with np.load(index_path) as idx:
            self._entries = {
                str(n): (int(o), int(s))
                for n, o, s in zip(idx['names'], idx['offsets'], idx['sizes'])
            }

    def _close(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:  # already closed, or interpreter shutdown
                pass
            self._fd = None

    def __del__(self):
        # os.open returns a bare int with no finalizer, so an LRU eviction would otherwise
        # leak one fd per scene -- thousands per worker over an epoch.
        self._close()

    def _fileno(self):
        # Workers fork after the dataset is built. os.pread is positional so a shared fd
        # would not corrupt reads, but reopening per process keeps each worker's fd
        # independent and avoids a parent close invalidating every child.
        pid = os.getpid()
        if self._fd is None or self._pid != pid:
            self._close()  # drop the fd inherited across fork; close is process-local
            self._fd = os.open(self.tar_path, os.O_RDONLY)
            self._pid = pid
        return self._fd

    def __contains__(self, name):
        return name in self._entries

    def read(self, name):
        """Return the raw bytes of one member."""
        try:
            offset, size = self._entries[name]
        except KeyError:
            raise KeyError(f'{name!r} not found in {self.tar_path}') from None
        return _pread_exact(self._fileno(), offset, size)


# Bounded so we don't hold an fd (and index) for every scene in the dataset; the cache is
# module state, so each DataLoader worker keeps its own after fork.
@functools.lru_cache(maxsize=32)
def get_store(tar_path):
    return TarSceneStore(tar_path)


# ---------------------------------------------------------------------------
# Writers (used by preprocessing pipelines, not during training)
# ---------------------------------------------------------------------------

class TarSceneWriter:
    """Write one scene's frames into an uncompressed tar + sidecar index.

    Atomic: writes to .tmp.{pid}, os.replace on close.
    Resume-safe: skips if both tar and index already exist.
    """

    def __init__(self, tar_path):
        self.tar_path = tar_path
        self._index_path = tar_path + INDEX_SUFFIX
        self._tmp_path = tar_path + f'.tmp.{os.getpid()}'
        self._tar = None
        self._skipped = False

        if os.path.isfile(tar_path) and os.path.isfile(self._index_path):
            self._skipped = True
            return

        os.makedirs(os.path.dirname(tar_path) or '.', exist_ok=True)
        self._tar = tarfile.open(self._tmp_path, 'w')

    @property
    def skipped(self):
        return self._skipped

    def add_npz(self, member_name, **arrays):
        """Write np.savez_compressed data as a tar member."""
        assert self._tar is not None, 'writer is closed or was skipped'
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        self._add_buf(member_name, buf)

    def add_bytes(self, member_name, data):
        """Write raw bytes (e.g. PNG) as a tar member."""
        assert self._tar is not None, 'writer is closed or was skipped'
        buf = io.BytesIO(data)
        self._add_buf(member_name, buf)

    def _add_buf(self, member_name, buf):
        buf.seek(0, 2)
        size = buf.tell()
        buf.seek(0)
        info = tarfile.TarInfo(name=member_name)
        info.size = size
        info.mtime = time.time()
        self._tar.addfile(info, buf)

    def close(self):
        if self._skipped or self._tar is None:
            return
        self._tar.close()
        self._tar = None
        os.replace(self._tmp_path, self.tar_path)
        build_index(self.tar_path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is not None and self._tar is not None:
            self._tar.close()
            self._tar = None
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass
            return False
        self.close()
        return False


class TarSceneAppender:
    """Append new members to an existing scene tar, rebuild sidecar on close.

    Idempotent: skips members already in sidecar.
    Crash-safe: truncates tar to last valid member end on open.
    """

    def __init__(self, tar_path):
        self.tar_path = tar_path
        index_path = tar_path + INDEX_SUFFIX
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f'Missing sidecar {index_path}')

        with np.load(index_path) as idx:
            self._existing = set(str(n) for n in idx['names'])
            offsets = idx['offsets'].astype(np.int64)
            sizes = idx['sizes'].astype(np.int64)

        # Truncate to last valid member end (removes torn tail from prior crash),
        # then re-append the two-block EOF marker that tarfile.open('a') expects.
        if len(offsets) > 0:
            member_ends = offsets + np.array(
                [math.ceil(s / 512) * 512 for s in sizes], dtype=np.int64)
            valid_end = int(member_ends.max())
            if os.path.getsize(tar_path) < valid_end:
                raise RuntimeError(f'Tar {tar_path} shorter than sidecar expects')
            os.truncate(tar_path, valid_end)
            # tarfile 'a' mode scans for the EOF marker (two 512-byte zero blocks);
            # after truncation those are gone, so write them back.
            with open(tar_path, 'ab') as f:
                f.write(b'\0' * 1024)

        self._tar = tarfile.open(tar_path, 'a')
        self._added = 0

    def add_bytes(self, member_name, data):
        """Write raw bytes, skip if already present."""
        if member_name in self._existing:
            return
        assert self._tar is not None, 'appender is closed'
        buf = io.BytesIO(data)
        info = tarfile.TarInfo(name=member_name)
        info.size = len(data)
        info.mtime = time.time()
        self._tar.addfile(info, buf)
        self._existing.add(member_name)
        self._added += 1

    def close(self):
        if self._tar is None:
            return
        self._tar.close()
        self._tar = None
        if self._added > 0:
            build_index(self.tar_path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
