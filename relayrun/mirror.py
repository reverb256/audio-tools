"""Copy a release directory to a second host with rsync over ssh.

The host and the remote path always come from flags or the environment
(``RELAY_MIRROR_HOST``, ``RELAY_MIRROR_PATH``). Nothing in this module names a
host. ``RELAY_MIRROR_RSH`` overrides the remote shell (default ``ssh``), which
is the rsync ``-e`` command.

After the copy, a ``MANIFEST.md5`` written into the release directory is checked
on the far side with ``md5sum -c``. The destination path must be absolute so the
remote check does not depend on a home directory.
"""

from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
from pathlib import Path

from relayrun.errors import RelayError

_HOST = re.compile(r"[A-Za-z0-9._-]+@[A-Za-z0-9._-]+|[A-Za-z0-9._-]+")


def check_host(host: str) -> str:
    host = (host or "").strip()
    if not host or not _HOST.fullmatch(host):
        raise RelayError(
            "mirror needs a host from --host or RELAY_MIRROR_HOST "
            "(a hostname or user@hostname, never a built-in default)",
            exit_code=2,
        )
    return host


def check_dest(dest: str) -> str:
    dest = (dest or "").strip()
    if not dest.startswith("/"):
        raise RelayError(
            "mirror destination must be an absolute path (--dest or RELAY_MIRROR_PATH)",
            exit_code=2,
        )
    if any(ch in dest for ch in ("\n", "\r", "'", '"', " ", "\t")):
        raise RelayError("mirror destination path contains whitespace or quotes", exit_code=2)
    return dest.rstrip("/")


def write_manifest(release: Path) -> Path:
    """Write MANIFEST.md5 (md5sum format) for every file except the manifest."""
    release = Path(release)
    if not release.is_dir():
        raise RelayError(f"release directory not found: {release}", exit_code=2)
    lines = []
    files = sorted(
        p for p in release.rglob("*")
        if p.is_file() and not p.is_symlink() and p.name != "MANIFEST.md5"
    )
    if not files:
        raise RelayError(f"release directory is empty: {release}", exit_code=2)
    for path in files:
        rel = path.relative_to(release).as_posix()
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    manifest = release / "MANIFEST.md5"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def rsync_command(release: Path, host: str, dest: str, rsh: str) -> list[str]:
    source = str(Path(release).resolve()) + "/"
    remote = f"{host}:{dest}/"
    return ["rsync", "-a", "--checksum", "-e", rsh, source, remote]


def verify_command(host: str, dest: str, rsh: str) -> list[str]:
    remote = f"cd {shlex.quote(dest)} && md5sum -c MANIFEST.md5"
    return [*shlex.split(rsh), host, remote]


def run_checked(cmd: list[str], *, what: str) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RelayError(f"{what} failed: command not found ({cmd[0]})") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        detail = "\n".join(tail) if tail else f"exit {proc.returncode}"
        raise RelayError(f"{what} failed:\n{detail}")
    return proc


def verify_remote(host: str, dest: str, rsh: str) -> None:
    proc = run_checked(verify_command(host, dest, rsh), what="checksum verification")
    bad = [line for line in (proc.stdout or "").splitlines() if line and "OK" not in line]
    if bad:
        raise RelayError("checksum verification failed:\n" + "\n".join(bad[:20]))


def mirror(release: Path, host: str, dest: str, *, rsh: str = "ssh", dry_run: bool = False) -> dict:
    host = check_host(host)
    dest = check_dest(dest)
    rsh = (rsh or "ssh").strip()
    if not rsh:
        raise RelayError("mirror remote shell is empty", exit_code=2)
    release = Path(release)
    if not release.is_dir():
        raise RelayError(f"release directory not found: {release}", exit_code=2)
    copy = rsync_command(release, host, dest, rsh)
    verify = verify_command(host, dest, rsh)
    if dry_run:
        return {"dry_run": True, "rsync": copy, "verify": verify, "files": None}
    manifest = write_manifest(release)
    file_count = len(manifest.read_text(encoding="utf-8").splitlines())
    run_checked(copy, what="rsync")
    verify_remote(host, dest, rsh)
    return {"dry_run": False, "rsync": copy, "verify": verify, "files": file_count, "manifest": str(manifest)}
