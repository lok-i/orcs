"""Bounded, revision-pinned remote ZIP inspection; credentials stay in memory."""

from __future__ import annotations

import zipfile
from pathlib import Path

from .workspace import safe_name


class RemoteFile:
    def __init__(self, url, size, token=None, max_bytes=256 * 1024 * 1024):
        import requests

        self.url = url
        self.size = size
        self.pos = 0
        self.downloaded = 0
        self.max_bytes = max_bytes
        self.session = requests.Session()
        self.headers = {"Accept-Encoding": "identity"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def seek(self, p, whence=0):
        dest = p if whence == 0 else self.pos + p if whence == 1 else self.size + p
        if not 0 <= dest <= self.size:
            raise ValueError("Remote seek outside archive")
        self.pos = dest
        return dest

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n == 0:
            return b""
        if n > 64 * 1024 * 1024 or self.downloaded + n > self.max_bytes:
            raise ValueError("Remote read exceeds byte budget")
        start = self.pos
        with self.session.get(
            self.url,
            headers={**self.headers, "Range": f"bytes={start}-{start + n - 1}"},
            stream=True,
            timeout=60,
        ) as r:
            if r.status_code != 206:
                raise RuntimeError(
                    f"Server refused bounded range (HTTP {r.status_code})"
                )
            if (
                r.headers.get("Content-Range", "")
                != f"bytes {start}-{start + n - 1}/{self.size}"
            ):
                raise RuntimeError("Wrong HTTP Content-Range")
            data = r.raw.read(n + 1)
        if len(data) != n:
            raise RuntimeError("Truncated/oversized remote range")
        self.pos += n
        self.downloaded += n
        return data


def inspect_remote(ws, sequence, revision, extract=(), prefix=()):
    from huggingface_hub import HfApi, get_token, hf_hub_url

    safe_name(sequence)
    api = HfApi()
    info = api.dataset_info("leggedrobotics/egohtr", revision=revision)
    sha = info.sha
    path = f"sequences/{sequence}/full_sequence.npz"
    entry = next(
        x
        for x in api.list_repo_tree(
            "leggedrobotics/egohtr",
            path_in_repo=f"sequences/{sequence}",
            repo_type="dataset",
            revision=sha,
        )
        if x.path == path
    )
    remote = RemoteFile(
        hf_hub_url("leggedrobotics/egohtr", path, repo_type="dataset", revision=sha),
        entry.size,
        get_token(),
    )
    destination = Path(ws.settings()["data_root"]) / "sequences" / sequence
    written = []
    try:
        with zipfile.ZipFile(remote) as archive:
            members = [
                dict(
                    name=m.filename, compressed_bytes=m.compress_size, bytes=m.file_size
                )
                for m in archive.infolist()
            ]
            for name in list(extract) + list(prefix):
                if Path(name).name != name:
                    raise ValueError("Extract only top-level named archive members")
                member = archive.getinfo(name)
                if name in extract and (
                    member.file_size > 512 * 1024 * 1024
                    or member.compress_size > 64 * 1024 * 1024
                ):
                    raise ValueError("Member exceeds extraction budget")
                destination.mkdir(parents=True, exist_ok=True)
                target = destination / (name if name in extract else name + ".prefix")
                if target.exists():
                    raise ValueError(f"Already exists: {target}")
                with archive.open(name) as stream:
                    target.write_bytes(
                        stream.read() if name in extract else stream.read(16384)
                    )
                written.append(str(target))
    finally:
        remote.session.close()
    return dict(
        sequence=sequence,
        revision=sha,
        members=members,
        extracted=written,
        http_bytes=remote.downloaded,
    )
