#!/usr/bin/env python3
"""Batch JPEG/PNG → JPEG XL (lossless) compressor

Recursively walks a source directory and encodes qualifying raster images
(JPEG / JPG / PNG by default) into lossless .jxl files, preserving the
relative directory structure under an output root.

Design goals:
  - Pure stdlib (subprocess call out to `cjxl` / optional `djxl` for verification)
  - Idempotent (skips existing outputs unless --overwrite)
  - Safe by default (never deletes originals unless --delete-original + confirmation)
  - Deterministic & resumable (can re-run; only processes missing / stale outputs)
  - Parallel (thread or process pool; mostly I/O + external CPU bound encoder)

Core encoding rules:
  - JPEG sources: use `cjxl --lossless_jpeg=1` for reversible transcoding.
  - PNG sources: use `cjxl -d 0` (distance 0 => mathematically lossless).
  - Other formats (if added later) default to -d 0 unless overridden.

Output path logic:
  - Default outdir: <source_root>/compressed_jxl
  - Relative structure preserved:  src/2025/Trip/img001.jpg → out/2025/Trip/img001.jxl
  - Name collision handling (e.g., img001.jpg & img001.png): configurable via --collision-mode

Verification (optional):
  - PNG: decode and byte-compare with original (sha256)
  - JPEG: if transcoded with --lossless_jpeg=1, reconstruct JPEG via djxl & byte-compare

Exit codes:
  0 success (all processed or skipped successfully)
  1 general error
  2 partial failures (some files failed) – still produces a summary

Planned (scaffold) – implementation filled in after CLI agreement.
"""
from __future__ import annotations

import argparse
import concurrent.futures as _futures
import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------- Data Structures ----------------------------- #

@dataclass
class EncodeTask:
    src: Path
    dst: Path
    src_type: str  # 'jpeg' or 'png' (future: 'other')
    reason: str    # why it's being processed (missing, stale, overwrite, etc.)

@dataclass
class EncodeResult:
  task: EncodeTask
  ok: bool
  skipped: bool
  message: str = ""
  bytes_in: int = 0
  bytes_out: int = 0
  fallback_used: bool = False  # whether we used a non-reconstructible fallback path

# ------------------------------- CLI Parsing ------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="batch_jxl_compress.py",
        description="Recursively convert JPEG/PNG images to lossless JPEG XL (.jxl).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dir", type=Path, default=Path.cwd(), help="Source root directory to scan.")
    p.add_argument("--outdir", type=Path, default=None, help="Output root (default: <source>/compressed_jxl)")
    p.add_argument("--extensions", default="jpg,jpeg,png", help="Comma-separated list (case-insensitive) of source extensions to include.")
    p.add_argument("--exclude-glob", action="append", default=[], help="Glob (relative) patterns to exclude (can repeat).")
    p.add_argument("--include-glob", action="append", default=[], help="Glob (relative) patterns to force-include (applied after exclude).")
    p.add_argument("--max-workers", type=int, default=os.cpu_count() or 4, help="Maximum parallel encodes (pool size).")
    p.add_argument("--threads-per-encode", type=int, default=0, help="Pass --num_threads to cjxl (0 = let cjxl decide).")
    p.add_argument("--min-bytes", type=int, default=0, help="Skip files smaller than this many bytes (savings negligible).")
    p.add_argument("--skip-if-larger", action="store_true", help="If output is larger than input, delete output and mark skipped.")
    p.add_argument("--overwrite", action="store_true", help="Force re-encode even if output exists and is newer.")
    p.add_argument("--collision-mode", choices=["suffix", "skip", "overwrite"], default="suffix", help="Behavior when name collisions (same stem different ext) map to same dst.")
    p.add_argument("--collision-suffix-format", default="{stem}.{ext}.jxl", help="Format used when collision-mode=suffix (available tokens: {stem}, {ext}).")
    p.add_argument("--verify", action="store_true", help="After encode, verify losslessness (PNG hash compare / JPEG round-trip).")
    p.add_argument("--strip", action="store_true", help="Pass --strip to cjxl (remove metadata).")
    p.add_argument("--preserve-times", action="store_true", help="Set output mtime = input mtime.")
    p.add_argument("--dry-run", action="store_true", help="Show what would be done without encoding.")
    p.add_argument("--resume", action="store_true", help="Skip outputs that already exist (same as default, explicit for clarity).")
    p.add_argument("--delete-original", action="store_true", help="Delete original after successful verified encode (DANGEROUS: requires --yes).")
    p.add_argument("--yes", action="store_true", help="Assume yes for prompts (needed for --delete-original).")
    p.add_argument("--progress", action="store_true", help="Show live progress bar (auto-enabled if stdout is TTY).")
    p.add_argument("--quiet", action="store_true", help="Reduce log output (errors only).")
    p.add_argument("--verbose", action="store_true", help="Verbose logging (debug).")
    p.add_argument("--log-file", type=Path, default=None, help="Optional log file to append structured events.")
    p.add_argument("--cjxl", type=Path, default=Path("cjxl"), help="Path to cjxl binary (or in $PATH).")
    p.add_argument("--djxl", type=Path, default=Path("djxl"), help="Path to djxl binary (for verification).")
    p.add_argument("--fail-fast", action="store_true", help="Stop processing on first encode failure.")
    # Fallback / safety options
    p.add_argument("--strict-jpeg", action="store_true", help="Do NOT fallback; fail if reversible JPEG transcode impossible.")
    p.add_argument("--force-delete-fallback", action="store_true", help="Allow deleting originals even when fallback (non-reconstructible) path used.")
    p.add_argument("--version", action="store_true", help="Print tool version and exit.")
    p.add_argument("--terminate-grace-seconds", type=float, default=8.0, help="Seconds to wait after Ctrl-C before force kill of encoders.")
    p.add_argument("--terminate-kill-seconds", type=float, default=2.0, help="Additional seconds after SIGTERM before SIGKILL.")
    p.add_argument("--jpeg-decode-fallback", choices=["skip", "reencode"], default="skip", help="On 'Getting pixel data failed' errors, try lossy-to-lossless pixel reencode using -d 0 (one extra generation) instead of failing.")
    p.add_argument("--quarantine-dir", type=Path, default=None, help="Copy originals of files that still fail after all fallbacks to this directory (preserves relative path).")
    return p

# ------------------------------ Version String ----------------------------- #
__version__ = "0.1.0-dev"  # updated when stabilized

# ------------------------------ Core Functions ----------------------------- #

def _stderr(msg: str):
  sys.stderr.write(msg + "\n")


def _is_tty() -> bool:
  return sys.stdout.isatty()


def _rel_match(rel_posix: str, patterns: List[str]) -> bool:
  if not patterns:
    return False
  # Use Path.match semantics but supply pattern relative (no leading ./)
  p = Path(rel_posix)
  return any(p.match(glob) for glob in patterns)


def _human_size(n: int) -> str:
  for unit in ["B", "KB", "MB", "GB", "TB"]:
    if n < 1024:
      return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
    n /= 1024
  return f"{n:.1f}PB"


def _hash_file(path: Path, algo: str = "sha256") -> str:
  h = hashlib.new(algo)
  with path.open('rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


def _quarantine_on_failure(task: EncodeTask, args) -> str:
  if not args.quarantine_dir:
    return ''
  try:
    src_root = args.dir.resolve()
    rel = task.src.relative_to(src_root)
  except Exception:
    # Fallback: just use filename
    rel = Path(task.src.name)
  qdst = args.quarantine_dir / rel
  try:
    qdst.parent.mkdir(parents=True, exist_ok=True)
    if not qdst.exists():
      shutil.copy2(task.src, qdst)
    return ' [quarantined]'
  except Exception:
    return ' [quarantine-failed]'


class ProgressPrinter(threading.Thread):
  def __init__(self, total: int, enabled: bool, initial_skipped: int = 0):
    super().__init__(daemon=True)
    self.total = total
    self.enabled = enabled and total > 0
    self.start_time = time.time()
    self.lock = threading.Lock()
    self.processed = 0
    self.success = 0
    self.skipped = 0  # runtime skipped (e.g. larger-no-gain, aborted, etc.)
    self.failed = 0
    self.last_line_len = 0
    self._stop_evt = threading.Event()
    self.initial_skipped = initial_skipped  # pre-scan skipped (up-to-date, too-small, collision, etc.)

  def update(self, processed_delta=0, success_delta=0, skipped_delta=0, failed_delta=0):
    if not self.enabled:
      return
    with self.lock:
      self.processed += processed_delta
      self.success += success_delta
      self.skipped += skipped_delta
      self.failed += failed_delta

  def stop(self):
    self._stop_evt.set()

  def run(self):
    if not self.enabled:
      return
    while not self._stop_evt.wait(0.2):
      self._print_status()
    # final
    self._print_status(final=True)

  def _print_status(self, final: bool = False):
    with self.lock:
      elapsed = time.time() - self.start_time
      rate = self.processed / elapsed if elapsed > 0 else 0
      total_skipped = self.skipped + self.initial_skipped
      if self.initial_skipped:
        skipped_repr = f"skipped={total_skipped} (pre={self.initial_skipped})"
      else:
        skipped_repr = f"skipped={total_skipped}"
      msg = (f"Processed {self.processed}/{self.total} | ok={self.success} "
             f"{skipped_repr} failed={self.failed} | {rate:.2f}/s")
      if final:
        msg += f" | elapsed {elapsed:.1f}s"
      # carriage return style
      line = msg + (" " * max(0, self.last_line_len - len(msg)))
      self.last_line_len = len(msg)
      print("\r" + line, end="" if not final else "\n", file=sys.stderr)


class ProcessManager:
  """Tracks external encoder processes so we can terminate them on interrupt."""
  def __init__(self):
    self.procs: List[subprocess.Popen] = []
    self.lock = threading.Lock()
    self.stop_flag = threading.Event()

  def register(self, p: subprocess.Popen):
    with self.lock:
      self.procs.append(p)

  def unregister(self, p: subprocess.Popen):
    with self.lock:
      try:
        self.procs.remove(p)
      except ValueError:
        pass

  def request_stop(self):
    self.stop_flag.set()

  def terminating(self) -> bool:
    return self.stop_flag.is_set()

  def terminate_all(self, grace: float, kill_extra: float):
    with self.lock:
      procs = list(self.procs)
    if not procs:
      return
    for p in procs:
      if p.poll() is None:
        try:
          p.terminate()
        except Exception:
          pass
    end = time.time() + grace
    while time.time() < end:
      if all(p.poll() is not None for p in procs):
        return
      time.sleep(0.1)
    # escalate
    for p in procs:
      if p.poll() is None:
        try:
          p.kill()
        except Exception:
          pass
    end2 = time.time() + kill_extra
    while time.time() < end2:
      if all(p.poll() is not None for p in procs):
        return
      time.sleep(0.05)


def _run_external(cmd: List[str], manager: ProcessManager, verbose: bool) -> Tuple[int, bytes, bytes, float]:
  """Run external command with cooperative cancellation.

  Returns (returncode, stdout, stderr, elapsed_seconds).
  If manager.stop_flag is set, sends terminate signals.
  """
  start = time.time()
  try:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  except FileNotFoundError as e:
    return (127, b'', str(e).encode(), 0.0)
  manager.register(p)
  try:
    # Poll loop so we can react to stop_flag
    while True:
      if manager.terminating():
        if p.poll() is None:
          try:
            p.terminate()
          except Exception:
            pass
      if p.poll() is not None:
        break
      time.sleep(0.05)
    out, err = p.communicate()
    return (p.returncode, out, err, time.time() - start)
  finally:
    manager.unregister(p)


def build_tasks(args) -> Tuple[List[EncodeTask], List[EncodeResult]]:
  root: Path = args.dir.resolve()
  if not root.is_dir():
    raise SystemExit(f"Source directory not found: {root}")
  outdir: Path = (args.outdir or (root / "compressed_jxl")).resolve()
  extensions = {e.strip().lower().lstrip('.') for e in args.extensions.split(',') if e.strip()}
  exclude_patterns = args.exclude_glob
  include_patterns = args.include_glob

  existing_collision: Dict[Path, EncodeTask] = {}
  tasks: List[EncodeTask] = []
  pre_results: List[EncodeResult] = []

  # Walk
  for dirpath, _dirnames, filenames in os.walk(root):
    # Prune hidden directories and files
    _dirnames[:] = [d for d in _dirnames if not d.startswith('.')]

    dirpath_p = Path(dirpath)
    for name in (f for f in filenames if not f.startswith('.')):
      src = dirpath_p / name
      if src.is_symlink():
        continue  # skip symlinks
      ext = src.suffix.lower().lstrip('.')
      if ext not in extensions:
        continue
      rel = src.relative_to(root)
      rel_posix = rel.as_posix()
      if _rel_match(rel_posix, exclude_patterns) and not _rel_match(rel_posix, include_patterns):
        continue
      if src.stat().st_size < args.min_bytes:
        pre_results.append(EncodeResult(
          task=EncodeTask(src=src, dst=Path("/dev/null"), src_type=ext if ext in ("jpg", "jpeg") else "png", reason="too-small"),
          ok=True, skipped=True, message="below-min-bytes"))
        continue

      # Determine dest path
      rel_parent = rel.parent
      dst_dir = outdir / rel_parent
      stem = src.stem
      # Normalize src type
      src_type = 'jpeg' if ext in ("jpg", "jpeg") else 'png'
      candidate = dst_dir / f"{stem}.jxl"
      original_candidate = candidate
      if candidate in existing_collision:
        # collision
        if args.collision_mode == 'skip':
          pre_results.append(EncodeResult(
            task=EncodeTask(src=src, dst=candidate, src_type=src_type, reason='collision-skip'),
            ok=True, skipped=True, message='collision-skip'))
          continue
        elif args.collision_mode == 'overwrite':
          # mark previous as skipped-overwritten
          prior = existing_collision[candidate]
          pre_results.append(EncodeResult(
            task=prior, ok=True, skipped=True, message='collision-overwritten'))
          # replace with new one (fall through)
        elif args.collision_mode == 'suffix':
          fmt = args.collision_suffix_format
          # ensure tokens
          collision_name = fmt.format(stem=stem, ext=ext)
          candidate = dst_dir / collision_name
          # ensure .jxl extension present
          if candidate.suffix.lower() != '.jxl':
            candidate = candidate.with_suffix('.jxl')
      # Decide if needs processing
      if candidate.exists() and not args.overwrite:
        src_m = src.stat().st_mtime
        dst_m = candidate.stat().st_mtime
        if src_m <= dst_m:
          pre_results.append(EncodeResult(
            task=EncodeTask(src=src, dst=candidate, src_type=src_type, reason='up-to-date'),
            ok=True, skipped=True, message='up-to-date'))
          continue
        reason = 'stale'
      else:
        reason = 'overwrite' if (candidate.exists() and args.overwrite) else 'new'

      task = EncodeTask(src=src, dst=candidate, src_type=src_type, reason=reason)
      existing_collision[original_candidate] = task
      tasks.append(task)

  return tasks, pre_results


def encode_one(task: EncodeTask, args, manager: ProcessManager) -> EncodeResult:
  try:
    task.dst.parent.mkdir(parents=True, exist_ok=True)
    bytes_in = task.src.stat().st_size
    fallback_used = False

    if manager.terminating():
      return EncodeResult(task=task, ok=True, skipped=True, message='aborted-before-start')

    final_dst = task.dst
    temp_dst = final_dst.with_name(final_dst.name + '.part')
    if temp_dst.exists():
      try:
        temp_dst.unlink()
      except Exception:
        return EncodeResult(task=task, ok=False, skipped=False, message=f'stale-temp-cannot-remove:{temp_dst}')

    def build_base_cmd(out_path: Path) -> List[str]:
      base = [str(args.cjxl), str(task.src), str(out_path)]
      if args.strip:
        base.append('--strip')
      if args.threads_per_encode > 0:
        base.extend(['--num_threads', str(args.threads_per_encode)])
      if not args.verbose:
        base.append('--quiet')
      return base

    # Encoding path selection
    if task.src_type == 'jpeg':
      # Primary attempt: lossless JPEG transcode (reconstructible)
      cmd = build_base_cmd(temp_dst) + ['--lossless_jpeg=1']
      rc, out, err, elapsed = _run_external(cmd, manager, args.verbose)
      primary_err_msg = err # Store error message in case fallback also fails

      # If primary fails, and fallback is enabled, try full re-encode.
      # This handles corrupt JPEGs (e.g. from WhatsApp) that cjxl can't
      # losslessly transcode but can decode pixels from.
      if rc != 0 and args.jpeg_decode_fallback == 'reencode' and not manager.terminating():
        if temp_dst.exists():
          try: temp_dst.unlink()
          except Exception: pass

        # Fallback to pixel-by-pixel re-encode using --lossless_jpeg=0
        cmd = build_base_cmd(temp_dst) + ['--lossless_jpeg=0']
        rc, out, err, elapsed = _run_external(cmd, manager, args.verbose)
        fallback_used = True
    else: # PNG
      cmd = build_base_cmd(temp_dst) + ['-d', '0']
      rc, out, err, elapsed = _run_external(cmd, manager, args.verbose)

    # Check for termination signal during encode
    if manager.terminating():
      if temp_dst.exists():
        try: temp_dst.unlink()
        except Exception: pass
      return EncodeResult(task=task, ok=True, skipped=True, message='aborted')

    # Check final result code
    if rc != 0:
      quarantine_note = _quarantine_on_failure(task, args)
      if fallback_used:
        # Fallback was used and failed. Report both errors for better diagnostics.
        err_report = f"primary_error='{primary_err_msg.decode(errors='ignore').strip()}' fallback_error='{err.decode(errors='ignore').strip()}'"
        return EncodeResult(task=task, ok=False, skipped=False, message=f"reencode-fallback-failed{quarantine_note}: {err_report}")
      else:
        # Primary attempt failed, no fallback attempted or it was disabled.
        return EncodeResult(task=task, ok=False, skipped=False, message=f"cjxl-failed{quarantine_note}: {err.decode(errors='ignore').strip()}")

    if not temp_dst.exists():
      return EncodeResult(task=task, ok=False, skipped=False, message='temp-destination-missing-after-encode', fallback_used=fallback_used)

    bytes_out = temp_dst.stat().st_size
    if args.skip_if_larger and bytes_out >= bytes_in:
      try:
        temp_dst.unlink()
      except Exception as e:
        return EncodeResult(task=task, ok=False, skipped=False, message=f'skip-if-larger-temp-cleanup-failed: {e}', fallback_used=fallback_used)
      return EncodeResult(task=task, ok=True, skipped=True, message='larger-no-gain', bytes_in=bytes_in, bytes_out=bytes_out, fallback_used=fallback_used)

    # Verification on temp
    if args.verify and not manager.terminating():
      if fallback_used and task.src_type == 'jpeg':
        verify_note = 'verify-skipped-fallback'
      else:
        if not args.djxl:
          return EncodeResult(task=task, ok=False, skipped=False, message='verify-requested-no-djxl', fallback_used=fallback_used)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.recon', delete=False) as tmp:
          tmp_path = Path(tmp.name)
        try:
          dec_cmd = [str(args.djxl), str(temp_dst), str(tmp_path)]
          if not args.verbose:
            dec_cmd.append('--quiet')
          rc_d, out_d, err_d, _ = _run_external(dec_cmd, manager, args.verbose)
          if manager.terminating():
            return EncodeResult(task=task, ok=True, skipped=True, message='aborted', fallback_used=fallback_used)
          if rc_d != 0:
            return EncodeResult(task=task, ok=False, skipped=False, message=f"djxl-failed: {err_d.decode(errors='ignore').strip()}", fallback_used=fallback_used)
          orig_hash = _hash_file(task.src)
          recon_hash = _hash_file(tmp_path)
          if orig_hash != recon_hash:
            return EncodeResult(task=task, ok=False, skipped=False, message='verify-mismatch', fallback_used=fallback_used)
          verify_note = 'verified'
        finally:
          try:
            if tmp_path.exists():
              tmp_path.unlink()
          except Exception:
            pass
    else:
      verify_note = 'no-verify'

    # Atomic promote
    try:
      os.replace(temp_dst, final_dst)
    except Exception as e:
      return EncodeResult(task=task, ok=False, skipped=False, message=f'atomic-rename-failed: {e}', fallback_used=fallback_used)

    if args.preserve_times:
      st = task.src.stat()
      os.utime(final_dst, (st.st_atime, st.st_mtime))

    deletion_note = ''
    if args.delete_original:
      if fallback_used and task.src_type == 'jpeg' and not args.force_delete_fallback:
        deletion_note = ' original-retained-fallback'
      else:
        if not args.yes:
          return EncodeResult(task=task, ok=False, skipped=False, message='delete-original-requires-yes', fallback_used=fallback_used)
        try:
          task.src.unlink()
          deletion_note = ' original-deleted'
        except Exception as e:
          return EncodeResult(task=task, ok=False, skipped=False, message=f'delete-original-failed: {e}', fallback_used=fallback_used)

    fb_note = ''
    if fallback_used:
      fb_note = ' fallback=reencode'
    msg = f"encoded {bytes_in}->{bytes_out} ratio={bytes_out/bytes_in:.3f} time={elapsed:.2f}s {verify_note}{fb_note}{deletion_note}".strip()
    return EncodeResult(task=task, ok=True, skipped=False, message=msg, bytes_in=bytes_in, bytes_out=bytes_out, fallback_used=fallback_used)
  except Exception as e:
    try:
      if 'temp_dst' in locals() and temp_dst.exists():
        temp_dst.unlink()
    except Exception:
      pass
    return EncodeResult(task=task, ok=False, skipped=False, message=f'exception: {e}')


def run_batch(args) -> int:
  tasks, pre_results = build_tasks(args)
  total_tasks = len(tasks)
  if args.dry_run:
    print(f"[DRY-RUN] Source: {args.dir}")
    outdir = args.outdir or (Path(args.dir) / 'compressed_jxl')
    print(f"[DRY-RUN] Outdir: {outdir}")
    for r in pre_results:
      if r.skipped:
        print(f"SKIP  {r.task.src} -> (n/a) reason={r.message}")
    for t in tasks:
      print(f"DO    {t.src} -> {t.dst} reason={t.reason} type={t.src_type}")
    print(f"[DRY-RUN] {len(tasks)} tasks, {sum(1 for r in pre_results if r.skipped)} pre-skipped")
    return 0

  if args.delete_original and not args.yes:
    _stderr("Refusing to run with --delete-original without --yes (safety). Abort.")
    return 1

  if not tasks and not pre_results:
    print("No matching files found.")
    return 0

  pre_skipped_count = sum(1 for r in pre_results if r.skipped)
  progress = ProgressPrinter(total=total_tasks, enabled=(args.progress or _is_tty()), initial_skipped=pre_skipped_count)
  progress.start()

  log_file_handle = None
  if args.log_file:
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file_handle = args.log_file.open('a', encoding='utf-8')

  all_results: List[EncodeResult] = []
  all_results.extend(pre_results)
  failures = 0

  # Handler for ctrl-c to attempt graceful shutdown
  stop_flag = threading.Event()

  manager = ProcessManager()

  def handle_sigint(signum, frame):  # noqa: ARG001
    stop_flag.set()
    manager.request_stop()
    _stderr("Received interrupt – cancelling pending tasks and stopping encoders...")

  orig_int = signal.getsignal(signal.SIGINT)
  signal.signal(signal.SIGINT, handle_sigint)
  try:
    with _futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
      future_map = {ex.submit(encode_one, t, args, manager): t for t in tasks}
      for fut in _futures.as_completed(future_map):
        if stop_flag.is_set():
          # Cancel remaining futures that haven't started
          for f2, tk in list(future_map.items()):
            if not f2.done():
              f2.cancel()
          break
        res = fut.result()
        all_results.append(res)
        if res.skipped:
          progress.update(processed_delta=1, skipped_delta=1)
        elif res.ok:
          progress.update(processed_delta=1, success_delta=1)
        else:
          progress.update(processed_delta=1, failed_delta=1)
          failures += 1
        # logging
        if log_file_handle:
          log_entry = {
            'event': 'result',
            'src': str(res.task.src),
            'dst': str(res.task.dst),
            'ok': res.ok,
            'skipped': res.skipped,
            'message': res.message,
            'bytes_in': res.bytes_in,
            'bytes_out': res.bytes_out,
            'ratio': (res.bytes_out / res.bytes_in) if res.bytes_in else None,
            'fallback_used': res.fallback_used,
          }
          log_file_handle.write(json.dumps(log_entry) + "\n")
        if not res.ok and args.fail_fast:
          _stderr("Fail-fast: aborting further tasks")
          stop_flag.set()
          break
  finally:
    progress.stop()
    progress.join(timeout=2)
    signal.signal(signal.SIGINT, orig_int)
    if stop_flag.is_set():
      manager.terminate_all(args.terminate_grace_seconds, args.terminate_kill_seconds)
    if log_file_handle:
      log_file_handle.flush()
      log_file_handle.close()

  # Print final summary (detailed prints only if not quiet)
  if not args.quiet:
    for r in all_results:
      if r.skipped:
        status = 'SKIP'
      elif r.ok:
        status = 'OK  '
      else:
        status = 'FAIL'
      if args.verbose or not r.ok:
        print(f"{status} {r.task.src} -> {r.task.dst} : {r.message}")
    total_in = sum(r.bytes_in for r in all_results if r.ok and not r.skipped)
    total_out = sum(r.bytes_out for r in all_results if r.ok and not r.skipped)
    ratio = (total_out / total_in) if total_in else 0
    print(f"Summary: files={len(all_results)} encoded={sum(1 for r in all_results if r.ok and not r.skipped)} skipped={sum(1 for r in all_results if r.skipped)} failures={failures}")
    if total_in:
      print(f"Bytes: in={_human_size(total_in)} out={_human_size(total_out)} ratio={ratio:.3f}")

  if stop_flag.is_set():
    _stderr("Interrupted – partial results saved.")
    return 130
  if failures:
    return 2
  return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.version:
    print(f"batch_jxl_compress {__version__}")
    return 0

  # Rename argparse dest for consistency with our attribute names used in functions
  # (Argparse already set attributes; nothing else to do.)
  return run_batch(args)

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
