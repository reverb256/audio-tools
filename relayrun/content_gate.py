"""Content gate for one album design directory, or a root of album directories.

The checks are the design content gate: schema, copy rules, distinctness, blend
compatibility, and an estimated continuous-mix length. A design directory holds
``NN-slug.json``, a matching ``NN-slug.md``, and ``ALBUM.md``.

An external implementation can replace this one. Set ``RELAY_CONTENT_GATE`` or
pass ``--content-gate`` to a Python file that defines ``main(root, report)`` and
returns 0 on pass and 1 on fail.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from relayrun.errors import RelayError

REQUIRED = [
    "album", "track", "side", "title", "style_prompt", "lyrics", "instrumental",
    "bpm", "key", "camelot", "duration", "seed", "takes",
]
STOP = set(
    "a an the of in on at to and or for from with by into onto over under "
    "is it my your our its no not up out".split()
)
PLACEHOLDERS = [("summary.", False), ("table.", False), ("TBD", True), ("lorem", False)]
BANNED = ["queer"]
MAX_VOCAL = 5
JACCARD_MAX = 0.6
CONCEPT_JACCARD_MAX = 0.5
MIN_MD = 1200
DUR_MIN, DUR_MAX = 180, 300
MIX_MIN, MIX_MAX = 117 * 60, 123 * 60
MIN_TRACKS = 24
_TRACK_JSON = re.compile(r"\d\d-.+\.json$")
_SKIP_SCAN = {"DISTINCTNESS-CHECK.md", "GATE.md", "GATE.json"}


def tags(prompt: str) -> set[str]:
    return {t.strip().lower() for t in prompt.split(",") if t.strip()}


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOP and len(w) > 2}


def camelot_ok(a: str, b: str) -> bool:
    m1 = re.fullmatch(r"(\d{1,2})([AB])", a or "")
    m2 = re.fullmatch(r"(\d{1,2})([AB])", b or "")
    if not (m1 and m2):
        return False
    n1, letter1, n2, letter2 = int(m1[1]), m1[2], int(m2[1]), m2[2]
    if n1 == n2:
        return True
    return letter1 == letter2 and (abs(n1 - n2) == 1 or abs(n1 - n2) == 11)


def section(md: str, name: str) -> str:
    match = re.search(rf"^## {re.escape(name)}\s*\n(.*?)(?=^## |\Z)", md, re.S | re.M)
    return match.group(1).strip() if match else ""


def labeled(md: str, name: str) -> str:
    match = re.search(rf"\*\*{re.escape(name)}:\*\*\s*(.+)", md)
    return match.group(1).strip() if match else ""


def _hms(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def is_album_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(_TRACK_JSON.match(name) for name in os.listdir(path))


def album_dirs(root: Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise RelayError(f"designs directory not found: {root}", exit_code=2)
    if is_album_dir(root):
        return [root]
    found = []
    for name in sorted(os.listdir(root)):
        path = root / name
        if name == "tools" or not path.is_dir():
            continue
        if is_album_dir(path):
            found.append(path)
    return found


@dataclass
class GateResult:
    album_dir: Path
    name: str
    tracks: int = 0
    fails: list[tuple[str, str]] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.fails

    def as_dict(self) -> dict:
        return {
            "album_dir": str(self.album_dir),
            "album": self.name,
            "tracks": self.tracks,
            "fails": [{"check": c, "message": m} for c, m in self.fails],
            "info": list(self.info),
            "verdict": "PASS" if self.ok else "FAIL",
        }


def _check_album(album_dir: Path) -> tuple[GateResult, list[dict]]:
    name = album_dir.name
    result = GateResult(album_dir=album_dir, name=name)
    fails = result.fails
    tracks: list[dict] = []
    json_names = sorted(f for f in os.listdir(album_dir) if _TRACK_JSON.match(f))
    if len(json_names) < MIN_TRACKS:
        fails.append((
            "structure",
            f"{name}: {len(json_names)} track JSON files, expected at least {MIN_TRACKS}",
        ))
    nums = sorted(int(f[:2]) for f in json_names)
    if json_names and nums != list(range(1, len(json_names) + 1)):
        fails.append(("structure", f"{name}: track numbers are not 1..{len(json_names)} without gaps"))
    if not (album_dir / "ALBUM.md").is_file():
        fails.append(("structure", f"{name}: ALBUM.md missing"))
    for filename in json_names:
        path = album_dir / filename
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            fails.append(("json", f"{name}/{filename}: unreadable ({exc})"))
            continue
        if not isinstance(data, dict):
            fails.append(("json", f"{name}/{filename}: expected an object"))
            continue
        missing = [k for k in REQUIRED if k not in data or data[k] in (None, "")]
        if missing:
            fails.append(("json-fields", f"{name}/{filename}: missing {', '.join(missing)}"))
        md_path = album_dir / (filename[:-5] + ".md")
        md = md_path.read_text(encoding="utf-8") if md_path.is_file() else ""
        if not md:
            fails.append(("structure", f"{name}/{filename[:-5]}.md missing"))
        tracks.append({"album": name, "file": filename, "j": data, "md": md})
    result.tracks = len(tracks)

    for item in tracks:
        data, md, where = item["j"], item["md"], f"{item['album']}/{item['file']}"
        title = str(data.get("title", ""))
        if re.search(r"track\s*\d", title, re.I):
            fails.append(("title-placeholder", f"{where}: title '{title}' matches Track N"))
        bpm = data.get("bpm")
        if not isinstance(bpm, int) or isinstance(bpm, bool) or bpm < 30:
            fails.append(("json-fields", f"{where}: bpm must be an int >= 30"))
        camelot = str(data.get("camelot", ""))
        if not re.fullmatch(r"\d{1,2}[AB]", camelot):
            fails.append(("json-fields", f"{where}: camelot '{camelot}' invalid"))
        if re.fullmatch(r"\d{1,2}[AB]", str(data.get("key", ""))):
            fails.append(("json-fields", f"{where}: key holds a Camelot code, needs a musical key"))
        duration = data.get("duration", 0)
        if not isinstance(duration, int) or isinstance(duration, bool) or not DUR_MIN <= duration <= DUR_MAX:
            fails.append(("duration", f"{where}: duration {duration} outside {DUR_MIN}-{DUR_MAX} s"))
        if data.get("takes") != 1:
            fails.append(("json-fields", f"{where}: takes should be 1"))
        if len(str(data.get("style_prompt", ""))) >= 400:
            fails.append(("prompt", f"{where}: style prompt >= 400 chars"))
        item["concept"] = section(md, "Concept")
        item["meaning"] = section(md, "Deep meaning")
        item["lead"] = labeled(md, "Lead instrument")
        if len(item["concept"]) < 120:
            fails.append(("concept", f"{where}: concept missing or under 120 chars"))
        if len(re.findall(r"[.!?](\s|$)", item["meaning"])) < 2:
            fails.append(("meaning", f"{where}: deep meaning missing or under 2 sentences"))
        if not item["lead"]:
            fails.append(("lead", f"{where}: lead instrument missing"))
        arrangement = section(md, "Arrangement")
        rows = re.findall(r"^\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|", arrangement, re.M)
        rows = [row for row in rows if row[0].lower() != "total"]
        if len(rows) < 4:
            fails.append(("arrangement", f"{where}: arrangement needs bar counts for at least 4 sections"))
        elif not (16 <= int(rows[0][1]) <= 32 and 16 <= int(rows[-1][1]) <= 32):
            fails.append((
                "arrangement",
                f"{where}: intro/outro must be 16-32 bars (got {rows[0][1]}/{rows[-1][1]})",
            ))
        if not section(md, "Mix notes") or "Mix in" not in md or "Mix out" not in md:
            fails.append(("mix-notes", f"{where}: mix-in/mix-out notes missing"))
        vocal = data.get("instrumental") is False
        item["vocal"] = vocal
        lyrics = str(data.get("lyrics", ""))
        if vocal:
            content = [line for line in lyrics.splitlines()
                       if line.strip() and not re.fullmatch(r"\[[^\]]+\]", line.strip())]
            if lyrics.strip() in ("", "[instrumental]") or not content:
                fails.append(("vocal-lyrics", f"{where}: vocal track has no lyrics or hook words"))
            if not re.search(r"Hook( words)?:", md):
                fails.append(("vocal-lyrics", f"{where}: vocal plan does not state the hook"))
        elif lyrics.strip() != "[instrumental]":
            fails.append(("vocal-lyrics", f"{where}: instrumental track must have lyrics '[instrumental]'"))

    ordered = sorted(tracks, key=lambda t: t["j"].get("track", 0) if isinstance(t["j"].get("track"), int) else 0)
    total = 0.0
    for index, item in enumerate(ordered):
        duration = item["j"].get("duration") if isinstance(item["j"].get("duration"), int) else 0
        match = re.search(r"\*\*Blend in:\*\*\s*(\d+) bars", item["md"])
        bars = int(match.group(1)) if (match and index) else 0
        bpm = item["j"].get("bpm") if isinstance(item["j"].get("bpm"), int) and item["j"].get("bpm") else 120
        total += duration - bars * 240 / bpm
    result.info.append(f"{name}: {len(ordered)} tracks, estimated continuous mix {_hms(total)}")
    if tracks and not MIX_MIN <= total <= MIX_MAX:
        fails.append(("mix-runtime", f"{name}: estimated mix {_hms(total)} outside {_hms(MIX_MIN)}-{_hms(MIX_MAX)}"))

    vocal_count = sum(1 for item in tracks if item.get("vocal"))
    result.info.append(f"{name}: {vocal_count} vocal tracks")
    if vocal_count > MAX_VOCAL:
        fails.append(("vocal-count", f"{name}: {vocal_count} vocal tracks (max {MAX_VOCAL})"))

    for prev, nxt in zip(ordered, ordered[1:]):
        prev_j, next_j = prev["j"], nxt["j"]
        try:
            delta = abs(int(next_j.get("bpm")) - int(prev_j.get("bpm")))
        except (TypeError, ValueError):
            delta = 999
        ok = camelot_ok(str(prev_j.get("camelot")), str(next_j.get("camelot"))) and delta <= 2
        if ok:
            continue
        if re.search(r"\*\*Cut reason:\*\*\s*.{40,}", nxt["md"]):
            result.info.append(
                f"{name}: {prev_j.get('title')} -> {next_j.get('title')} is an intentional cut "
                f"({prev_j.get('camelot')}/{prev_j.get('bpm')} -> {next_j.get('camelot')}/{next_j.get('bpm')}), reason given"
            )
        else:
            fails.append((
                "blend",
                f"{name}: {prev_j.get('title')} -> {next_j.get('title')} breaks blend rule "
                f"({prev_j.get('camelot')}/{prev_j.get('bpm')} -> {next_j.get('camelot')}/{next_j.get('bpm')}) "
                "with no cut reason",
            ))
    return result, tracks


def check_tree(root: Path) -> list[GateResult]:
    """Run the built-in gate. One result per album directory."""
    root = Path(root)
    albums = album_dirs(root)
    results: list[GateResult] = []
    all_tracks: list[dict] = []
    if not albums:
        empty = GateResult(album_dir=root, name=root.name)
        empty.fails.append(("structure", "no album folders with track JSON found"))
        results.append(empty)
    for album in albums:
        result, tracks = _check_album(album)
        results.append(result)
        all_tracks.extend(tracks)

    def fail_all(check: str, message: str) -> None:
        # Tree-wide failures attach to every album so a single-album run still reports them,
        # and a multi-album run does not hide a cross-album collision inside one folder.
        for result in results:
            result.fails.append((check, message))

    seen_titles: dict[str, str] = {}
    seen_words: dict[str, str] = {}
    for item in all_tracks:
        title = str(item["j"].get("title", ""))
        key = title.lower().strip()
        if key in seen_titles:
            fail_all("title-dup", f"'{title}' in {item['album']} duplicates {seen_titles[key]}")
        else:
            seen_titles[key] = item["album"]
        for word in set(re.findall(r"[a-z0-9']+", key)) - STOP:
            if word in seen_words:
                fail_all("title-word-dup", f"word '{word}' in '{title}' also in '{seen_words[word]}'")
            else:
                seen_words[word] = title

    seeds = [item["j"].get("seed") for item in all_tracks]
    if len(seeds) != len(set(seeds)):
        fail_all("seed-dup", "duplicate seeds")

    combos: dict[tuple, str] = {}
    for item in all_tracks:
        combo = (item["j"].get("camelot"), item["j"].get("bpm"), item.get("lead", "").lower())
        title = str(item["j"].get("title"))
        if combo in combos:
            fail_all("combo-dup", f"{title} repeats key+BPM+lead of {combos[combo]}")
        else:
            combos[combo] = title

    max_j, max_pair = 0.0, None
    for left, right in itertools.combinations(all_tracks, 2):
        score = jaccard(tags(str(left["j"].get("style_prompt", ""))), tags(str(right["j"].get("style_prompt", ""))))
        if score > max_j:
            max_j, max_pair = score, (left["j"].get("title"), right["j"].get("title"))
        if score > JACCARD_MAX:
            fail_all(
                "prompt-jaccard",
                f"{left['j'].get('title')} vs {right['j'].get('title')}: {score:.2f}",
            )
    if results:
        results[0].info.append(f"style-prompt pairs checked: {len(all_tracks) * (len(all_tracks) - 1) // 2}; max Jaccard {max_j:.2f} ({max_pair})")

    max_c, cpair = 0.0, None
    concept_tracks = [item for item in all_tracks if item.get("concept")]
    for left, right in itertools.combinations(concept_tracks, 2):
        if left["concept"].strip().lower() == right["concept"].strip().lower():
            fail_all("concept-dup", f"{left['j'].get('title')} and {right['j'].get('title')} share a concept")
            continue
        score = jaccard(words(left["concept"]), words(right["concept"]))
        if score > max_c:
            max_c, cpair = score, (left["j"].get("title"), right["j"].get("title"))
        if score > CONCEPT_JACCARD_MAX:
            fail_all(
                "concept-dup",
                f"{left['j'].get('title')} vs {right['j'].get('title')}: concept word overlap {score:.2f}",
            )
    if results:
        results[0].info.append(f"concept max word-Jaccard {max_c:.2f} ({cpair})")

    scan_root = root if not is_album_dir(root) else root
    for dirpath, _, filenames in os.walk(scan_root):
        for filename in filenames:
            if not filename.endswith((".md", ".json")) or filename in _SKIP_SCAN:
                continue
            path = Path(dirpath) / filename
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = path.relative_to(scan_root).as_posix()
            for phrase, case_sensitive in PLACEHOLDERS:
                flags = 0 if case_sensitive else re.I
                if re.search(r"(?<![A-Za-z])" + re.escape(phrase), text, flags):
                    fail_all("placeholder", f"{rel}: contains '{phrase}'")
            for banned in BANNED:
                if re.search(rf"\b{banned}\b", text, re.I):
                    fail_all("banned-word", f"{rel}: contains '{banned}'")
            if filename.endswith(".md") and len(text.encode()) < MIN_MD:
                fail_all("md-size", f"{rel}: {len(text.encode())} bytes < {MIN_MD}")
    return results


def render_markdown(root: Path, results: list[GateResult]) -> str:
    fails = [(c, m) for result in results for c, m in result.fails]
    info = [line for result in results for line in result.info]
    names = ", ".join(r.name for r in results) or "none"
    tracks = sum(r.tracks for r in results)
    verdict = "PASS" if not fails else "FAIL"
    lines = [
        f"# Content gate: {verdict}",
        "",
        f"Root: `{root}`",
        f"Albums: {names}; tracks: {tracks}",
        "",
        "Checks: Track-N titles, duplicate titles and title words (stopwords ignored), "
        "style-prompt Jaccard <= 0.6 on comma tags, concept duplicates (exact or word-Jaccard > 0.5), "
        f"<= {MAX_VOCAL} vocal tracks per album, {DUR_MIN}-{DUR_MAX} s per track, "
        "estimated mix 1:57-2:03, vocal tracks carry lyrics or hook words, placeholder phrases, "
        f"md files >= {MIN_MD} bytes, required JSON fields, neighbour blend rule "
        "(+-1 Camelot or relative, <= 2 BPM, else a written cut reason), unique key+BPM+lead, "
        "unique seeds, 16-32 bar intro/outro, banned copy words.",
        "",
        "## Info",
        "",
    ]
    lines += [f"- {item}" for item in info] or ["- none", ""]
    lines += ["", f"## Failures ({len(fails)})", ""]
    grouped: dict[str, list[str]] = {}
    for check, message in fails:
        grouped.setdefault(check, []).append(message)
    for check, messages in sorted(grouped.items()):
        lines.append(f"### {check} ({len(messages)})")
        lines += [f"- {message}" for message in messages[:40]]
        if len(messages) > 40:
            lines.append(f"- ... {len(messages) - 40} more")
        lines.append("")
    if not fails:
        lines.append("None.")
    return "\n".join(lines).rstrip() + "\n"


def write_results(results: list[GateResult], report: Path | None) -> None:
    from mixgen.util import atomic_write_json, atomic_write_text

    for result in results:
        body = render_markdown(result.album_dir, [result])
        atomic_write_text(result.album_dir / "GATE.md", body)
        atomic_write_json(result.album_dir / "GATE.json", result.as_dict())
    if report is not None:
        root = results[0].album_dir.parent if results else Path(".")
        atomic_write_text(Path(report), render_markdown(root, results))


def load_external(script: Path):
    script = Path(script)
    if not script.is_file():
        raise RelayError(f"content gate script not found: {script}", exit_code=2)
    spec = importlib.util.spec_from_file_location("relay_external_content_gate", script)
    if spec is None or spec.loader is None:
        raise RelayError(f"could not import content gate: {script}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RelayError(f"content gate {script.name} failed to import: {exc}") from exc
    if not hasattr(module, "main"):
        raise RelayError(f"content gate {script.name} has no main(root, report)")
    return module


def run_external(script: Path, root: Path, report: Path | None) -> int:
    module = load_external(script)
    try:
        code = module.main(str(root), None if report is None else str(report))
    except SystemExit as exc:
        code = exc.code
    except Exception as exc:
        raise RelayError(f"content gate failed: {exc}") from exc
    if code is None:
        return 0
    try:
        return int(code)
    except (TypeError, ValueError) as exc:
        raise RelayError(f"content gate returned {code!r}, expected 0 or 1") from exc
