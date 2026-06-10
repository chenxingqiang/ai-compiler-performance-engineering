#!/usr/bin/env python3
"""Audit book sample labels against repo implementations.

The current manuscript is text extracted from the laid-out book, not fenced markdown.
This tool therefore extracts explicit sample labels such as ``foo.cu`` or ``bar.py``
directly from the chapter text, then resolves them by:

1. exact chapter-local path match
2. explicit registry entry from ``book/sample_registry.json``
3. wrapper comments that intentionally preserve the book label
4. token-based generic matching within the chapter directory

Only ``hard_mismatch`` findings are treated as failures. Generic pedagogical labels,
wrapper files, and pseudocode/debug-artifact labels are reported but do not fail.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BOOK_DIR = ROOT / "book"
REGISTRY_PATH = Path(__file__).with_name("book_sample_registry.json")

# NOTE: the lookbehind intentionally allows a leading "/" or "." so path- and
# ``./``-prefixed sample references (e.g. ``./add_sequential.py``) are still
# captured by their basename; excluding them silently dropped real labels.
LABEL_RE = re.compile(r"(?<![\w-])([A-Za-z][A-Za-z0-9_./-]*\.(?:cu|py|cuh))(?![\w.-])")
CHAPTER_RE = re.compile(r"ch(\d+)\.md$")
SAMPLE_SUFFIXES = {".cu", ".py", ".cuh"}
GENERIC_TOKEN_SYNONYMS = {
    "before": {"baseline"},
    "after": {"optimized"},
    "naive": {"baseline"},
    "predicated": {"optimized"},
    "sequential": {"baseline"},
    "parallel": {"optimized"},
}


@dataclass(frozen=True)
class RegistryEntry:
    chapter: int
    label: str
    classification: str
    canonical_path: str | None = None
    notes: str | None = None
    required_markers: list[str] = field(default_factory=list)


@dataclass
class SampleResult:
    chapter: int
    label: str
    classification: str
    canonical_path: str | None
    status: str
    resolution: str
    notes: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chapter",
        action="append",
        default=[],
        help="Chapter number or slug (e.g. 10, ch10). Repeatable.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional output path for the structured audit JSON report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every resolved sample, not just chapter summaries and mismatches.",
    )
    parser.add_argument(
        "--book-dir",
        type=Path,
        default=BOOK_DIR,
        help="Book manuscript directory to audit (default: repo_root/book).",
    )
    return parser.parse_args()


def chapter_number(book_path: Path) -> int:
    match = CHAPTER_RE.fullmatch(book_path.name)
    if not match:
        raise ValueError(f"Unexpected chapter filename: {book_path}")
    return int(match.group(1))


def chapter_dir(chapter: int) -> Path | None:
    padded = ROOT / f"ch{chapter:02d}"
    if padded.exists():
        return padded
    plain = ROOT / f"ch{chapter}"
    if plain.exists():
        return plain
    return None


def load_registry(path: Path) -> dict[tuple[int, str], RegistryEntry]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[tuple[int, str], RegistryEntry] = {}
    for raw in payload.get("entries", []):
        entry = RegistryEntry(
            chapter=int(raw["chapter"]),
            label=raw["label"],
            classification=raw["classification"],
            canonical_path=raw.get("canonical_path"),
            notes=raw.get("notes"),
            required_markers=list(raw.get("required_markers", [])),
        )
        entries[(entry.chapter, entry.label)] = entry
    return entries


def extract_labels(book_path: Path) -> list[str]:
    text = book_path.read_text(encoding="utf-8")
    return sorted(set(LABEL_RE.findall(text)))


def candidate_files(chapter_root: Path | None) -> list[Path]:
    if chapter_root is None:
        return []
    return sorted(
        path
        for path in chapter_root.rglob("*")
        if path.is_file() and path.suffix in SAMPLE_SUFFIXES
    )


def camel_to_tokens(text: str) -> list[str]:
    split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return [token for token in re.split(r"[^a-z0-9]+", split.lower()) if token]


def expanded_tokens(label: str) -> set[str]:
    tokens = set(camel_to_tokens(Path(label).stem))
    for token in list(tokens):
        tokens.update(GENERIC_TOKEN_SYNONYMS.get(token, set()))
    return tokens


def validate_markers(path: Path, markers: list[str]) -> list[str]:
    if not markers:
        return []
    content = path.read_text(encoding="utf-8")
    return [marker for marker in markers if marker not in content]


def registry_resolution(entry: RegistryEntry) -> SampleResult:
    if entry.classification == "pseudocode":
        return SampleResult(
            chapter=entry.chapter,
            label=entry.label,
            classification=entry.classification,
            canonical_path=None,
            status="ok",
            resolution="registry",
            notes=entry.notes,
        )

    if entry.canonical_path is None:
        return SampleResult(
            chapter=entry.chapter,
            label=entry.label,
            classification="hard_mismatch",
            canonical_path=None,
            status="hard_mismatch",
            resolution="registry",
            notes="Registry entry is missing canonical_path",
        )

    target = ROOT / entry.canonical_path
    if not target.exists():
        return SampleResult(
            chapter=entry.chapter,
            label=entry.label,
            classification="hard_mismatch",
            canonical_path=entry.canonical_path,
            status="hard_mismatch",
            resolution="registry",
            notes=f"Registry target does not exist: {entry.canonical_path}",
        )

    missing = validate_markers(target, entry.required_markers)
    if missing:
        return SampleResult(
            chapter=entry.chapter,
            label=entry.label,
            classification="hard_mismatch",
            canonical_path=entry.canonical_path,
            status="hard_mismatch",
            resolution="registry",
            notes=f"Required markers missing from {entry.canonical_path}: {', '.join(missing)}",
        )

    return SampleResult(
        chapter=entry.chapter,
        label=entry.label,
        classification=entry.classification,
        canonical_path=entry.canonical_path,
        status="ok",
        resolution="registry",
        notes=entry.notes,
    )


def exact_resolution(chapter: int, label: str, chapter_root: Path | None) -> SampleResult | None:
    repo_relative = ROOT / label
    if repo_relative.exists():
        return SampleResult(
            chapter=chapter,
            label=label,
            classification="exact",
            canonical_path=str(repo_relative.relative_to(ROOT)),
            status="ok",
            resolution="exact_path",
        )

    if chapter_root is None:
        return None

    direct = chapter_root / label
    if direct.exists():
        return SampleResult(
            chapter=chapter,
            label=label,
            classification="exact",
            canonical_path=str(direct.relative_to(ROOT)),
            status="ok",
            resolution="chapter_path",
        )

    basename_hits = [path for path in candidate_files(chapter_root) if path.name == Path(label).name]
    if len(basename_hits) == 1:
        return SampleResult(
            chapter=chapter,
            label=label,
            classification="exact",
            canonical_path=str(basename_hits[0].relative_to(ROOT)),
            status="ok",
            resolution="chapter_basename",
        )
    return None


def wrapper_resolution(chapter: int, label: str, candidates: list[Path]) -> SampleResult | None:
    hits = []
    for candidate in candidates:
        try:
            content = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if label in content:
            hits.append(candidate)

    if len(hits) == 1:
        return SampleResult(
            chapter=chapter,
            label=label,
            classification="wrapper",
            canonical_path=str(hits[0].relative_to(ROOT)),
            status="ok",
            resolution="wrapper_comment",
            notes="Book label is intentionally preserved by a wrapper or helper comment.",
        )
    return None


def generic_resolution(chapter: int, label: str, candidates: list[Path]) -> SampleResult | None:
    label_tokens = expanded_tokens(label)
    label_suffix = Path(label).suffix
    scored: list[tuple[float, Path]] = []

    for candidate in candidates:
        if candidate.suffix not in SAMPLE_SUFFIXES:
            continue
        candidate_tokens = expanded_tokens(candidate.name)
        overlap = len(label_tokens & candidate_tokens)
        if overlap == 0:
            continue
        score = float(overlap)
        if candidate.suffix == label_suffix:
            score += 0.25
        if Path(candidate.name).stem.startswith(Path(label).stem.lower()):
            score += 0.25
        scored.append((score, candidate))

    if not scored:
        return None

    scored.sort(key=lambda item: (-item[0], len(str(item[1]))))
    best_score = scored[0][0]
    best = [candidate for score, candidate in scored if score == best_score]
    if len(best) != 1:
        return SampleResult(
            chapter=chapter,
            label=label,
            classification="hard_mismatch",
            canonical_path=None,
            status="hard_mismatch",
            resolution="generic_search",
            notes=f"Ambiguous generic match candidates: {', '.join(str(path.relative_to(ROOT)) for path in best[:5])}",
        )

    label_token_count = len(label_tokens)
    if best_score < 2.0 and label_token_count > 1:
        return None

    return SampleResult(
        chapter=chapter,
        label=label,
        classification="generic",
        canonical_path=str(best[0].relative_to(ROOT)),
        status="ok",
        resolution="generic_search",
        notes="Resolved via token similarity inside the chapter directory.",
    )


def resolve_label(
    chapter: int,
    label: str,
    chapter_root: Path | None,
    registry: dict[tuple[int, str], RegistryEntry],
) -> SampleResult:
    registry_entry = registry.get((chapter, label))
    if registry_entry is not None:
        return registry_resolution(registry_entry)

    exact = exact_resolution(chapter, label, chapter_root)
    if exact is not None:
        return exact

    candidates = candidate_files(chapter_root)

    wrapper = wrapper_resolution(chapter, label, candidates)
    if wrapper is not None:
        return wrapper

    generic = generic_resolution(chapter, label, candidates)
    if generic is not None:
        return generic

    return SampleResult(
        chapter=chapter,
        label=label,
        classification="hard_mismatch",
        canonical_path=None,
        status="hard_mismatch",
        resolution="unresolved",
        notes="No exact, registry, wrapper, or generic chapter-local match found.",
    )


def selected_chapters(args: argparse.Namespace) -> list[Path]:
    book_dir = Path(args.book_dir).resolve()
    all_books = sorted(path for path in book_dir.glob("ch*.md") if CHAPTER_RE.fullmatch(path.name))
    if not args.chapter:
        return all_books

    wanted: set[int] = set()
    for item in args.chapter:
        if item.lower().startswith("ch"):
            wanted.add(int(item[2:]))
        else:
            wanted.add(int(item))
    return [path for path in all_books if chapter_number(path) in wanted]


def orphaned_registry_entries(
    registry: dict[tuple[int, str], RegistryEntry],
    chapter_results: dict[int, list[SampleResult]],
) -> list[RegistryEntry]:
    """Registry entries whose label never appears in the current book chapter text.

    These are stale: the manuscript no longer prints the sample label they map, so
    the audit never exercises them. Reported (non-fatal) so the registry stays
    honest against the extracted ``book/chNN.md`` source.
    """
    present = {(ch, res.label) for ch, results in chapter_results.items() for res in results}
    return [entry for key, entry in sorted(registry.items()) if key not in present]


def chapter_summary(results: list[SampleResult]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.classification] = counts.get(result.classification, 0) + 1
    return {
        "samples": len(results),
        "hard_mismatches": sum(1 for result in results if result.status == "hard_mismatch"),
        "classification_counts": counts,
    }


def print_report(chapter_results: dict[int, list[SampleResult]], verbose: bool) -> None:
    total_samples = sum(len(results) for results in chapter_results.values())
    total_hard_mismatches = sum(
        1 for results in chapter_results.values() for result in results if result.status == "hard_mismatch"
    )

    print("=" * 72)
    print("BOOK SAMPLE ALIGNMENT REPORT")
    print("=" * 72)
    print(f"Samples checked: {total_samples}")
    print(f"Hard mismatches: {total_hard_mismatches}")
    print()

    for chapter in sorted(chapter_results):
        results = chapter_results[chapter]
        summary = chapter_summary(results)
        counts = ", ".join(
            f"{name}={count}" for name, count in sorted(summary["classification_counts"].items())
        )
        print(
            f"Chapter {chapter}: {summary['samples']} samples, "
            f"{summary['hard_mismatches']} hard mismatches ({counts})"
        )
        for result in results:
            if not verbose and result.status != "hard_mismatch":
                continue
            path = result.canonical_path or "-"
            print(
                f"  [{result.status}] {result.label} -> {path} "
                f"({result.classification}, {result.resolution})"
            )
            if result.notes:
                print(f"    {result.notes}")
        if not verbose and summary["hard_mismatches"] == 0:
            print("  No hard mismatches.")
        print()


def main() -> int:
    args = parse_args()
    registry = load_registry(REGISTRY_PATH)
    books = selected_chapters(args)

    chapter_results: dict[int, list[SampleResult]] = {}
    for book_path in books:
        chapter = chapter_number(book_path)
        labels = extract_labels(book_path)
        results = [
            resolve_label(chapter, label, chapter_dir(chapter), registry)
            for label in labels
        ]
        chapter_results[chapter] = results

    print_report(chapter_results, verbose=args.verbose)

    orphans = orphaned_registry_entries(registry, chapter_results)
    if orphans:
        print("=" * 72)
        print(f"ORPHANED REGISTRY ENTRIES (label not present in book/chNN.md): {len(orphans)}")
        print("=" * 72)
        for entry in orphans:
            print(f"  ch{entry.chapter}: {entry.label}  ({entry.classification})")
        print("  These map sample labels the manuscript no longer prints; prune or")
        print("  re-add the labels to the book. Non-fatal (audit exit code unchanged).")
        print()

    if args.json is not None:
        payload = {
            "chapters": {
                str(chapter): {
                    "summary": chapter_summary(results),
                    "samples": [asdict(result) for result in results],
                }
                for chapter, results in sorted(chapter_results.items())
            },
            "orphaned_registry": [asdict(entry) for entry in orphans],
            "summary": {
                "chapters": len(chapter_results),
                "samples": sum(len(results) for results in chapter_results.values()),
                "hard_mismatches": sum(
                    1
                    for results in chapter_results.values()
                    for result in results
                    if result.status == "hard_mismatch"
                ),
                "orphaned_registry_entries": len(orphans),
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return 1 if any(
        result.status == "hard_mismatch"
        for results in chapter_results.values()
        for result in results
    ) else 0


if __name__ == "__main__":
    sys.exit(main())
