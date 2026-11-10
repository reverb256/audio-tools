"""Write RELEASE.tsv for a finished release directory.

Columns, in order::

    track_no  side  title  file  duration  lufs_i  true_peak_dbtp  bpm  key
    camelot  md5  isrc  songwriter  instrumental

``songwriter`` is always blank. Design fields that might carry a credit or a
personal name (``songwriter``, ``composer``, ``legal_name``, ``author``,
``writer``) are ignored. ``isrc`` is blank until a store assigns one.

Side A rows come from ``NN-slug.json`` plus ``A/<stem>.wav`` and its
``.wav.json`` receipt (verdict must be PASS, and the file md5 must match).
Side B rows come from ``B/split-manifest.json``. Piece loudness is copied from
the receipt and is allowed to vary. The unsplit full mix is not a store row.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from relayrun.audio import file_md5
from relayrun.errors import RelayError

HEADER = [
    "track_no", "side", "title", "file", "duration", "lufs_i", "true_peak_dbtp",
    "bpm", "key", "camelot", "md5", "isrc", "songwriter", "instrumental",
]
# Never copied into the sheet. The songwriter column stays blank regardless.
_IGNORED_NAME_FIELDS = ("songwriter", "composer", "legal_name", "author", "writer", "credit")


def format_duration(seconds: float) -> str:
    total = int(round(float(seconds)))
    if total < 0:
        total = 0
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _load_json(path: Path, what: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RelayError(f"{what}: {exc}") from exc
    if not isinstance(data, dict):
        raise RelayError(f"{what}: expected an object")
    return data


def _receipt(wav: Path) -> dict:
    path = Path(str(wav) + ".json")
    if not path.is_file():
        raise RelayError(f"missing measurement receipt {path.name} for {wav.name}")
    data = _load_json(path, path.name)
    digest = file_md5(wav)
    if str(data.get("md5") or "") != digest:
        raise RelayError(f"{wav.name}: md5 {digest} does not match the receipt {data.get('md5')}")
    return data


def _blank(value) -> str:
    if value is None:
        return ""
    return str(value)


def _row(number: int, side: str, title: str, file: str, receipt: dict, design: dict) -> list[str]:
    instrumental = design.get("instrumental", True)
    if isinstance(instrumental, str):
        instrumental = instrumental.strip().lower() not in ("false", "0", "no")
    return [
        str(number),
        side,
        title,
        file,
        format_duration(receipt["duration_s"]),
        f"{float(receipt['lufs_i']):.1f}",
        f"{float(receipt['true_peak_dbtp']):.1f}",
        _blank(design.get("bpm", "")),
        _blank(design.get("key", "")),
        _blank(design.get("camelot", "")),
        str(receipt["md5"]),
        "",   # isrc, filled after delivery
        "",   # songwriter stays blank
        "true" if instrumental else "false",
    ]


def _designs(designs: Path) -> list[tuple[str, dict]]:
    rows = []
    for path in sorted(designs.glob("[0-9][0-9]-*.json")):
        data = _load_json(path, path.name)
        side = str(data.get("side") or "A").upper()
        if side != "A":
            continue
        rows.append((path.stem, data))
    rows.sort(key=lambda item: int(item[1].get("track") or item[0][:2]))
    return rows


def _match_design(designs: list[tuple[str, dict]], source_title: str, slug: str) -> dict:
    for _stem, data in designs:
        if str(data.get("title") or "") == source_title:
            return data
        if str(data.get("slug") or "") == slug:
            return data
    for stem, data in designs:
        if slug and (slug == stem.split("-", 1)[-1] or stem.endswith(slug)):
            return data
    return {}


def build_rows(designs: Path, release: Path) -> tuple[list[list[str]], list[str]]:
    designs, release = Path(designs), Path(release)
    if not designs.is_dir():
        raise RelayError(f"designs directory not found: {designs}", exit_code=2)
    if not release.is_dir():
        raise RelayError(f"release directory not found: {release}", exit_code=2)
    problems: list[str] = []
    rows: list[list[str]] = []
    number = 0
    side_a = _designs(designs)
    if not side_a:
        problems.append(f"no side A designs in {designs}")
    for stem, data in side_a:
        number += 1
        wav = release / "A" / f"{stem}.wav"
        if not wav.is_file():
            matches = sorted((release / "A").glob(f"{stem[:2]}-*.wav")) if (release / "A").is_dir() else []
            wav = matches[0] if len(matches) == 1 else wav
        if not wav.is_file():
            problems.append(f"side A missing {stem}.wav")
            continue
        try:
            receipt = _receipt(wav)
        except RelayError as exc:
            problems.append(str(exc))
            continue
        if receipt.get("verdict") != "PASS":
            problems.append(f"{wav.name}: master verdict {receipt.get('verdict')}")
        title = str(data.get("title") or stem)
        rel = wav.relative_to(release).as_posix()
        rows.append(_row(number, "A", title, rel, receipt, data))

    manifest_path = release / "B" / "split-manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path, "split-manifest.json")
        for piece in manifest.get("pieces") or []:
            number += 1
            rel_file = str(piece.get("file") or "")
            wav = release / "B" / rel_file
            if not wav.is_file():
                problems.append(f"side B missing {rel_file}")
                continue
            try:
                receipt = _receipt(wav)
            except RelayError as exc:
                problems.append(str(exc))
                continue
            design = _match_design(side_a, str(piece.get("source_title") or ""), str(piece.get("slug") or ""))
            title = str(piece.get("title") or "")
            if not title.endswith(" (Mixed)"):
                problems.append(f"{rel_file}: title {title!r} must end with ' (Mixed)'")
            rows.append(_row(number, "B", title, f"B/{rel_file}", receipt, design))

    extras_path = release / "C" / "extras.json"
    if extras_path.is_file():
        try:
            extras = json.loads(extras_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RelayError(f"C/extras.json: {exc}") from exc
        if not isinstance(extras, list):
            raise RelayError("C/extras.json: expected a list")
        for extra in extras:
            number += 1
            rel_file = str(extra.get("file") or "")
            wav = release / rel_file
            if not wav.is_file():
                problems.append(f"side C missing {rel_file}")
                continue
            try:
                receipt = _receipt(wav)
            except RelayError as exc:
                problems.append(str(exc))
                continue
            if receipt.get("verdict") not in ("PASS", "RECORDED"):
                problems.append(f"{wav.name}: verdict {receipt.get('verdict')}")
            rows.append(_row(
                number, str(extra.get("side") or "C"), str(extra.get("title") or wav.stem),
                rel_file, receipt, extra,
            ))
    return rows, problems


def render_tsv(rows: list[list[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(HEADER)
    for row in rows:
        if len(row) != len(HEADER):
            raise RelayError("internal error: release row has the wrong width")
        if row[HEADER.index("songwriter")] != "":
            raise RelayError("internal error: songwriter column was not blank")
        if row[HEADER.index("isrc")] != "":
            raise RelayError("internal error: isrc column was not blank")
        writer.writerow(row)
    return buffer.getvalue()


def name_tokens(*blobs: dict) -> list[str]:
    """String values of credit fields. These must not be copied onto the sheet."""
    tokens = []
    for data in blobs:
        if not isinstance(data, dict):
            continue
        for key in _IGNORED_NAME_FIELDS:
            value = data.get(key)
            if isinstance(value, str) and len(value.strip()) >= 8:
                tokens.append(value.strip())
    return tokens


def assert_names_absent(text: str, tokens: list[str]) -> None:
    for token in tokens:
        if token and token in text:
            raise RelayError("release sheet contains a credit or name that must stay off the file")


def assert_sheet_safe(text: str) -> None:
    """The sheet carries no songwriter credit. Name-like design fields are not copied."""
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    header = next(reader, None)
    if header != HEADER:
        raise RelayError(f"RELEASE.tsv header is {header}, expected the release columns")
    idx = header.index("songwriter")
    for line_no, row in enumerate(reader, start=2):
        if len(row) != len(header):
            raise RelayError(f"RELEASE.tsv line {line_no} has {len(row)} columns")
        if row[idx] != "":
            raise RelayError(f"RELEASE.tsv line {line_no} has a songwriter value")
