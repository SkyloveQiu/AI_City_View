"""Batch runner for the 7-stage pipeline.

Runs the existing pipeline over a folder (or a single image), writing each image's
outputs into its own subfolder under the output root.

Example:
  python batch_run.py input/ output_batch/ --limit 500 --skip-existing

Notes:
- Keeps a single Python process so model caches (Depth Anything / LangSAM) are reused.
- This script is intentionally minimal: no multiprocessing by default.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from concurrent.futures import ThreadPoolExecutor, as_completed, Future

from main import (
    get_default_config,
    process_image_pipeline,
    process_image_pipeline_stage1_to_5,
    process_image_pipeline_stage6_to_7,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _expand_path(p: str) -> Path:
    return Path(p).expanduser()


def _hint_if_common_path_mistake(p: Path) -> str | None:
    # Common mistake when running from repo root: using /input instead of ./input
    if p.is_absolute() and not p.exists():
        candidate = Path.cwd() / p.name
        if candidate.exists():
            return f"Did you mean: {candidate} ?"
    return None


@dataclass
class ItemResult:
    image_path: str
    ok: bool
    output_dir: str | None
    duration_seconds: float | None
    error: str | None


def _bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + ("#" * width) + "]"
    done = max(0, min(done, total))
    filled = int(round(width * (done / total)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _iter_images(input_path: Path, recursive: bool) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pattern = "**/*" if recursive else "*"
    for p in sorted(input_path.glob(pattern)):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _should_skip(out_dir: Path) -> bool:
    return (out_dir / "metadata.json").exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input image file or directory")
    parser.add_argument("output", type=str, help="Output root directory")
    parser.add_argument("--limit", type=int, default=0, help="Max number of images to process (0 = no limit)")
    parser.add_argument("--recursive", action="store_true", help="Recursively search input directory")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images that already have metadata.json")
    parser.add_argument("--profile", action="store_true", help="Enable per-stage timing logs (verbose)")
    parser.add_argument("--workers", type=int, default=1, help="Thread workers (1 = sequential). On single-GPU workloads this may not speed up.")
    parser.add_argument(
        "--post-workers",
        type=int,
        default=2,
        help="Threads for stage6+7 (generate+save). Overlaps CPU I/O with GPU inference.",
    )
    parser.add_argument(
        "--max-inflight-post",
        type=int,
        default=2,
        help="Max number of images concurrently in stage6+7. Limits RAM usage.",
    )
    args = parser.parse_args()

    input_path = _expand_path(args.input)
    output_root = _expand_path(args.output)

    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print(f"ERROR: Cannot create output directory: {output_root}")
        print(f"  Reason: {e}")
        print("  Tip: Use a writable path, e.g. ./output_batch_input_150 or ~/output_batch_input_150")
        return 3

    config = get_default_config()
    if args.profile:
        config["profile"] = True

    try:
        images = list(_iter_images(input_path, args.recursive))
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        hint = _hint_if_common_path_mistake(input_path)
        if hint:
            print(f"  Tip: {hint}")
        return 2
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    if not images:
        print("No images found.", flush=True)
        return 2

    print(f"Found {len(images)} image(s). Output root: {output_root}", flush=True)

    t_batch0 = time.perf_counter()
    results: List[ItemResult] = []
    total = len(images)
    def _print_progress(done: int, extra: str = "") -> None:
        elapsed = time.perf_counter() - t_batch0
        ok_so_far = sum(1 for r in results if r.ok)
        mean_ok = (sum((r.duration_seconds or 0.0) for r in results if r.ok) / ok_so_far) if ok_so_far else None
        remaining = total - done
        eta = (mean_ok * remaining) if mean_ok is not None else None
        eta_s = f" ETA~{eta/60:.1f}m" if eta is not None else ""
        msg = f"PROGRESS {_bar(done, total)} {done}/{total} elapsed={elapsed/60:.1f}m{eta_s}"
        if extra:
            msg += f" | {extra}"
        print(msg, flush=True)

    _print_progress(0, "starting")

    workers = max(1, int(args.workers))
    if workers > 1:
        # Warm up model caches once in the main thread to avoid repeated cold imports.
        try:
            from pipeline.stage2_ai_inference import get_depth_model, get_semantic_model

            print(f"Warming up models (workers={workers})...", flush=True)
            _ = get_semantic_model(config)
            _ = get_depth_model(config)
            print("Warmup done.", flush=True)
        except Exception as e:
            print(f"Warmup skipped/failed: {e}", flush=True)


    def _run_one(idx: int, img_path: Path) -> ItemResult:
        """Legacy run: full pipeline in one call."""
        out_dir = output_root / img_path.stem
        if args.skip_existing and _should_skip(out_dir):
            return ItemResult(
                image_path=str(img_path),
                ok=True,
                output_dir=str(out_dir),
                duration_seconds=0.0,
                error=None,
            )

        t0 = time.perf_counter()
        r = process_image_pipeline(str(img_path), str(out_dir), config)
        dt = time.perf_counter() - t0
        return ItemResult(
            image_path=str(img_path),
            ok=bool(r.get("success", False)),
            output_dir=r.get("output_dir"),
            duration_seconds=float(r.get("duration_seconds") or dt),
            error=r.get("error"),
        )

    def _run_stage1_to_5(img_path: Path) -> dict:
        out_dir = output_root / img_path.stem
        return process_image_pipeline_stage1_to_5(str(img_path), str(out_dir), config)

    def _run_stage6_to_7(payload: dict) -> dict:
        return process_image_pipeline_stage6_to_7(payload, config)

    post_workers = max(0, int(args.post_workers))
    max_inflight_post = max(1, int(args.max_inflight_post))

    if workers == 1 and post_workers > 0:
        # Single-GPU friendly: keep stage2 (GPU) in main thread, offload stage6+7 (CPU/I/O)
        # to a small thread pool to overlap work.
        inflight: list[tuple[int, Path, float, Future]] = []
        post_ex = ThreadPoolExecutor(max_workers=post_workers)

        def _drain_one(block: bool) -> None:
            nonlocal inflight
            if not inflight:
                return
            if not block:
                # check if any completed
                done_idx = None
                for i, (_idx, _p, _t0, fut) in enumerate(inflight):
                    if fut.done():
                        done_idx = i
                        break
                if done_idx is None:
                    return
                idx0, p0, t0, fut0 = inflight.pop(done_idx)
            else:
                idx0, p0, t0, fut0 = inflight.pop(0)

            try:
                r = fut0.result()
                ok = bool(r.get('success', False))
                err = None
                if not ok:
                    errs = r.get('errors')
                    err = str(errs[0]) if isinstance(errs, list) and errs else 'postprocess failed'
                results.append(
                    ItemResult(
                        image_path=str(p0),
                        ok=ok,
                        output_dir=str(output_root / p0.stem),
                        duration_seconds=(time.perf_counter() - t0),
                        error=err,
                    )
                )
            except Exception as e:
                results.append(
                    ItemResult(
                        image_path=str(p0),
                        ok=False,
                        output_dir=str(output_root / p0.stem),
                        duration_seconds=(time.perf_counter() - t0),
                        error=str(e),
                    )
                )

        submitted = 0
        completed = 0
        for idx, img_path in enumerate(images, start=1):
            out_dir = output_root / img_path.stem
            if args.skip_existing and _should_skip(out_dir):
                print(f"[{idx}/{len(images)}] SKIP {img_path} (already processed)", flush=True)
                results.append(
                    ItemResult(
                        image_path=str(img_path),
                        ok=True,
                        output_dir=str(out_dir),
                        duration_seconds=0.0,
                        error=None,
                    )
                )
                _print_progress(idx, "skip")
                continue

            # Bound inflight post tasks to control RAM.
            while len(inflight) >= max_inflight_post:
                # Drain one completed (blocking)
                _drain_one(block=True)
                completed = len([r for r in results if r.duration_seconds is not None])
                _print_progress(completed, "post")

            print(f"[{idx}/{len(images)}] RUN  {img_path}", flush=True)
            t0 = time.perf_counter()
            payload = _run_stage1_to_5(img_path)
            fut = post_ex.submit(_run_stage6_to_7, payload)
            inflight.append((idx, img_path, t0, fut))
            submitted += 1

            # Opportunistically drain any finished post tasks.
            _drain_one(block=False)
            completed = len(results)
            _print_progress(completed, f"submitted={submitted}")

        # Drain remaining
        while inflight:
            _drain_one(block=True)
            completed = len(results)
            _print_progress(completed, "post")
        post_ex.shutdown(wait=True)

    elif workers == 1:
        for idx, img_path in enumerate(images, start=1):
            out_dir = output_root / img_path.stem
            if args.skip_existing and _should_skip(out_dir):
                print(f"[{idx}/{len(images)}] SKIP {img_path} (already processed)", flush=True)
                results.append(
                    ItemResult(
                        image_path=str(img_path),
                        ok=True,
                        output_dir=str(out_dir),
                        duration_seconds=0.0,
                        error=None,
                    )
                )
                _print_progress(idx, "skip")
                continue

            print(f"[{idx}/{len(images)}] RUN  {img_path}", flush=True)
            results.append(_run_one(idx, img_path))
            _print_progress(idx, "ok" if results[-1].ok else "failed")
    else:
        # Threaded: output from process_image_pipeline will interleave.
        # Note: model inference is mostly GPU-bound; threads may not speed up.
        print("Running with threads; per-image logs may interleave.", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            in_flight = {}
            next_idx = 1

            # Prime the executor with up to `workers` tasks.
            while next_idx <= total and len(in_flight) < workers:
                img_path = images[next_idx - 1]
                print(f"[{next_idx}/{total}] QUEUE {img_path}", flush=True)
                fut = ex.submit(_run_one, next_idx, img_path)
                in_flight[fut] = next_idx
                next_idx += 1

            done_count = 0
            while in_flight:
                for fut in as_completed(list(in_flight.keys()), timeout=None):
                    idx = in_flight.pop(fut)
                    try:
                        item = fut.result()
                    except Exception as e:
                        img_path = images[idx - 1]
                        item = ItemResult(
                            image_path=str(img_path),
                            ok=False,
                            output_dir=str(output_root / img_path.stem),
                            duration_seconds=None,
                            error=str(e),
                        )
                    results.append(item)
                    done_count += 1
                    _print_progress(done_count, "ok" if item.ok else "failed")

                    # Refill one task.
                    if next_idx <= total:
                        img_path = images[next_idx - 1]
                        print(f"[{next_idx}/{total}] QUEUE {img_path}", flush=True)
                        nf = ex.submit(_run_one, next_idx, img_path)
                        in_flight[nf] = next_idx
                        next_idx += 1
                    break

    total_s = time.perf_counter() - t_batch0
    ok_count = sum(1 for r in results if r.ok)
    fail_count = len(results) - ok_count
    mean_s = (
        sum((r.duration_seconds or 0.0) for r in results if r.ok) / ok_count
        if ok_count
        else None
    )

    summary = {
        "input": str(input_path),
        "output_root": str(output_root),
        "count": len(results),
        "ok": ok_count,
        "failed": fail_count,
        "total_seconds": total_s,
        "mean_seconds_ok": mean_s,
        "items": [r.__dict__ for r in results],
    }

    summary_path = output_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nBatch done.", flush=True)
    print(f"  ok={ok_count} failed={fail_count} total={total_s:.2f}s", flush=True)
    if mean_s is not None:
        print(f"  mean(ok)={mean_s:.2f}s", flush=True)
    print(f"  summary={summary_path}", flush=True)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
