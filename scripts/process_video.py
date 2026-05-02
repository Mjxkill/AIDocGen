#!/usr/bin/env python3
"""Video → Markdown via ffmpeg + Ollama omnimodal model (default nemotron3:33b).

Modes:
  vision   – sample N frames, ask the model to describe each (no audio)
  audio    – chunk audio, ask the model to transcribe each chunk (no frames)
  full     – chunk both: send audio + 1 frame per chunk and ask for transcript+scene

Resume:
  Per-chunk results are stored in --state-dir (default <output_dir>/chunks).
  Re-running the same command skips chunks already on disk; lets you continue
  after a crash or cancel without redoing the whole video.

Overlap:
  --overlap N adds N seconds of audio context before each chunk to avoid losing
  words spoken across chunk boundaries (default 1.5s). Frame timing unchanged.
"""
import argparse
import base64
import concurrent.futures as cf
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PROMPTS = {
    "vision": (
        "You are watching a single frame from a video. "
        "Describe in 1–3 sentences what is visible (people, slides, code, UI, scene). "
        "Output plain text only, no preamble, no markdown headers."
    ),
    "audio": (
        "You are listening to one chunk of a video's audio. "
        "Transcribe the speech verbatim in the original language. "
        "If there is only music or silence write '[music]' or '[silence]'. "
        "Output plain text only, no preamble."
    ),
    "full": (
        "You are given one chunk of a video: a frame and the matching audio. "
        "1. Transcribe the speech verbatim (original language).\n"
        "2. On a new line starting with '> Scene:', describe the visible scene in one sentence.\n"
        "If audio is silent write '[silence]' instead of a transcript. "
        "Output plain text only, no preamble."
    ),
}


def ffprobe_duration(path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ], text=True)
    return float(out.strip())


def ffmpeg(*args: str):
    return subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args],
        check=True, capture_output=True, text=True,
    )


def hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def extract_frame(video: str, t_sec: float, out_path: Path) -> bool:
    try:
        ffmpeg("-ss", f"{t_sec}", "-i", video, "-frames:v", "1", "-q:v", "3", str(out_path))
        return out_path.exists() and out_path.stat().st_size > 0
    except subprocess.CalledProcessError:
        return False


def extract_audio_chunk(video: str, t_sec: float, dur_sec: float, out_path: Path) -> bool:
    try:
        ffmpeg("-ss", f"{t_sec}", "-t", f"{dur_sec}", "-i", video,
               "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out_path))
        return out_path.exists() and out_path.stat().st_size > 1024
    except subprocess.CalledProcessError:
        return False


def call_ollama(model: str, prompt: str, images: list[str] | None = None,
                audios: list[str] | None = None, host: str = "http://localhost:11434") -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1},
    }
    if images:
        payload["images"] = images
    if audios:
        payload["audios"] = audios
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())["response"].strip()


def b64_file(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def chunk_path(state_dir: Path, idx: int) -> Path:
    return state_dir / f"{idx:06d}.txt"


def process_chunk(idx: int, total: int, t_sec: float, mode: str, video: str, model: str,
                  chunk_dur: float, overlap: float, tmp: Path, state_dir: Path,
                  host: str) -> tuple[int, float, str]:
    out_file = chunk_path(state_dir, idx)
    if out_file.exists() and out_file.stat().st_size > 0:
        # Already done — skip.
        return idx, t_sec, out_file.read_text(encoding="utf-8")

    images = audios = None
    try:
        if mode in ("vision", "full"):
            mid = t_sec + chunk_dur / 2
            frame_path = tmp / f"frame_{idx:06d}.jpg"
            if extract_frame(video, mid, frame_path):
                images = [b64_file(frame_path)]
        if mode in ("audio", "full"):
            audio_start = max(0.0, t_sec - overlap)
            audio_dur = (t_sec - audio_start) + chunk_dur  # chunk + overlap before
            wav_path = tmp / f"audio_{idx:06d}.wav"
            if extract_audio_chunk(video, audio_start, audio_dur, wav_path):
                audios = [b64_file(wav_path)]
        if not images and not audios:
            text = "[error: nothing extracted]"
        else:
            text = call_ollama(model, PROMPTS[mode], images=images, audios=audios, host=host)
        # Persist immediately so we can resume.
        out_file.write_text(text, encoding="utf-8")
        print(f"  chunk {idx+1:4d}/{total} @ {hms(t_sec)}: {len(text)} chars", flush=True)
        return idx, t_sec, text
    except Exception as e:
        msg = f"[error: {e}]"
        out_file.write_text(msg, encoding="utf-8")
        print(f"  chunk {idx+1:4d}/{total} @ {hms(t_sec)}: ERROR {e}", flush=True)
        return idx, t_sec, msg


def update_progress(state_dir: Path, total: int, started_at: float, mode: str, model: str):
    done = sum(1 for p in state_dir.glob("[0-9]*.txt") if p.stat().st_size > 0)
    progress = {
        "done": done,
        "total": total,
        "pct": round(done / total * 100, 1) if total else 0,
        "started_at": started_at,
        "updated_at": time.time(),
        "mode": mode,
        "model": model,
    }
    (state_dir.parent / "progress.json").write_text(json.dumps(progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--mode", choices=["vision", "audio", "full"], default="full")
    ap.add_argument("--model", default="nemotron3:33b")
    ap.add_argument("--chunk", type=float, default=30.0,
                    help="chunk duration in seconds (default 30)")
    ap.add_argument("--overlap", type=float, default=1.5,
                    help="audio overlap in seconds added before each chunk (default 1.5)")
    ap.add_argument("--parallel", type=int, default=2,
                    help="parallel API calls (default 2)")
    ap.add_argument("--state-dir", default=None,
                    help="dir for per-chunk results (default <output>/chunks). "
                         "Re-running with the same dir resumes — already-done chunks are skipped.")
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg/ffprobe required")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_dir = Path(args.state_dir) if args.state_dir else out_path.parent / "chunks"
    state_dir.mkdir(parents=True, exist_ok=True)

    duration = ffprobe_duration(args.input)
    n_chunks = max(1, int(duration / args.chunk) + (1 if duration % args.chunk > 1 else 0))
    started_at = time.time()
    print(f"input: {args.input}  duration: {hms(duration)}  chunks: {n_chunks}  "
          f"mode: {args.mode}  model: {args.model}  overlap: {args.overlap}s", flush=True)

    already_done = sum(1 for i in range(n_chunks) if chunk_path(state_dir, i).exists()
                       and chunk_path(state_dir, i).stat().st_size > 0)
    if already_done:
        print(f"resume: {already_done}/{n_chunks} chunks already done — skipping those", flush=True)

    update_progress(state_dir, n_chunks, started_at, args.mode, args.model)

    with tempfile.TemporaryDirectory(prefix="vid_") as tmp_str:
        tmp = Path(tmp_str)
        with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = []
            for i in range(n_chunks):
                t = i * args.chunk
                futures.append(ex.submit(process_chunk, i, n_chunks, t, args.mode,
                                         args.input, args.model, args.chunk, args.overlap,
                                         tmp, state_dir, args.host))
            for fut in cf.as_completed(futures):
                fut.result()
                update_progress(state_dir, n_chunks, started_at, args.mode, args.model)

    # Final assembly: read every chunk file in order.
    title = Path(args.input).stem
    parts = [
        f"# {title}\n",
        f"_Source: {args.input} · {hms(duration)} · mode={args.mode} · "
        f"model={args.model} · chunk={args.chunk}s · overlap={args.overlap}s_\n",
    ]
    for i in range(n_chunks):
        p = chunk_path(state_dir, i)
        text = p.read_text(encoding="utf-8").strip() if p.exists() else "[missing]"
        t = i * args.chunk
        parts.append(f"\n## [{hms(t)}]\n\n{text}\n")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nDONE → {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
