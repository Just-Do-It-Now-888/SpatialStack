#!/usr/bin/env python3
"""Generate benchmark answers through an OpenAI-compatible vLLM server.

Port of ``Vision-OPD-main/eval/infer.py`` with three changes this project needs:

* multi-image messages, so the same script can serve a frame-budget benchmark
  later instead of only single-image CV-Bench;
* ``finish_reason`` and completion-token counts are persisted. MV-OPSD v0 was
  scored for a day on responses silently truncated at 16 tokens (ISSUE-003);
  a run that hits the cap must be visible in the artifact, not inferred later;
* ``reasoning_content`` is folded back in, because thinking-mode servers put
  the visible answer there when ``content`` comes back empty.

Resume is by ``sample_uid``: rerunning tops up missing or errored rows only.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import hashlib
import json
import mimetypes
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def make_sample_uid(item: dict, benchmark: str) -> str:
    for key in ("sample_uid", "uid", "index", "question_id", "id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return f"{benchmark}:{key}:{value}"
    stable = {
        "benchmark": benchmark,
        "images": item.get("images") or [],
        "query": item.get("query", ""),
    }
    raw = json.dumps(stable, ensure_ascii=False, sort_keys=True)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def should_retry(item: dict) -> bool:
    answer = item.get("model_answer")
    if not isinstance(answer, str) or not answer.strip():
        return True
    return answer.startswith("[API_ERROR]") or answer.startswith("[FUTURE_ERROR]")


def compact_existing(path: Path, benchmark: str):
    """Collapse duplicate uids, preferring successful rows. Returns (order, map, dirty)."""
    if not path.exists():
        return [], {}, False
    order, best, dirty = [], {}, False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                dirty = True
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dirty = True
                continue
            uid = record.get("sample_uid") or make_sample_uid(record, benchmark)
            record["sample_uid"] = uid
            if uid not in best:
                order.append(uid)
                best[uid] = record
                continue
            dirty = True
            if should_retry(best[uid]) or not should_retry(record):
                best[uid] = record
    return order, best, dirty


def rewrite(path: Path, order, records) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for uid in order:
            if uid in records:
                f.write(json.dumps(records[uid], ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def image_to_data_uri(path_str: str) -> str:
    path = Path(path_str)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64,{b64}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark-json", required=True)
    parser.add_argument("--out-dir", default="logs/eval_visionopd/model_answer")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--parallel-workers", type=int, default=64)
    parser.add_argument("--enable-thinking", choices=["True", "False"], default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.benchmark_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit:
        stride = max(1, len(data) // args.limit)
        data = data[::stride][: args.limit]

    out_dir = Path(args.out_dir) / args.benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_name}_answer.jsonl"

    # Resume works by reading what is already on disk, so a second process
    # writing the same file silently duplicates every request: both make
    # progress, dedup hides it at the end, and the only symptom is halved
    # throughput. That happened once (ISSUE-007), so refuse to start instead.
    lock = out_dir / f"{args.model_name}_answer.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = lock.read_text(encoding="utf-8").strip() or "unknown"
        pid = holder.split()[0]
        alive = pid.isdigit() and Path(f"/proc/{pid}").exists()
        print(f"another run holds {lock} ({holder}); alive={alive}", file=sys.stderr)
        if alive:
            print("refusing to write the same answer file twice", file=sys.stderr)
            sys.exit(1)
        print("holder is gone, taking over the lock", file=sys.stderr)
        lock.unlink()
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock_fd, f"{os.getpid()} {time.strftime('%F %T')}\n".encode())
    os.close(lock_fd)
    atexit.register(lambda: lock.unlink(missing_ok=True))

    done = set()
    if out_path.exists():
        order, existing, dirty = compact_existing(out_path, args.benchmark)
        if dirty:
            rewrite(out_path, order, existing)
        retry = {uid for uid in order if should_retry(existing[uid])}
        done = {uid for uid in order if uid not in retry}
        print(f"resume: {len(done)} done, {len(retry)} to retry")

    todo = []
    for item in data:
        uid = make_sample_uid(item, args.benchmark)
        if uid in done:
            continue
        record = dict(item)
        record["sample_uid"] = uid
        todo.append(record)
    print(f"remaining: {len(todo)}")
    if not todo:
        print("nothing to do")
        return

    thread_local = threading.local()

    def get_client() -> OpenAI:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=3600)
            thread_local.client = client
        return client

    def run_one(item: dict) -> dict:
        content = [
            {"type": "image_url", "image_url": {"url": image_to_data_uri(path)}}
            for path in (item.get("images") or [])
        ]
        content.append({"type": "text", "text": item.get("query", "").replace("<image>", "").strip()})
        messages = [{"role": "user", "content": content}]

        extra = {}
        if args.enable_thinking is not None:
            extra["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": args.enable_thinking == "True"}
            }

        record = dict(item)
        for attempt in range(1, args.max_retries + 1):
            try:
                resp = get_client().chat.completions.create(
                    model=args.model_id,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                    **extra,
                )
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                reasoning = (getattr(choice.message, "reasoning_content", None) or "").strip()
                if not text and reasoning:
                    text = reasoning
                record["model_answer"] = text
                record["finish_reason"] = choice.finish_reason
                record["reasoning_content"] = reasoning
                usage = getattr(resp, "usage", None)
                record["completion_tokens"] = getattr(usage, "completion_tokens", None)
                return record
            except Exception as exc:  # noqa: BLE001 - server-side errors vary
                if attempt == args.max_retries:
                    record["model_answer"] = f"[API_ERROR] {exc}"
                    record["finish_reason"] = "error"
                else:
                    time.sleep(1.0)
        return record

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool, open(
        out_path, "a", encoding="utf-8"
    ) as f_out:
        futures = {pool.submit(run_one, item): item for item in todo}
        with tqdm(total=len(todo), desc="inference", unit="case", dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001
                    record = dict(futures[future])
                    record["model_answer"] = f"[FUTURE_ERROR] {exc}"
                    record["finish_reason"] = "error"
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                bar.update(1)

    order, records, dirty = compact_existing(out_path, args.benchmark)
    if dirty:
        rewrite(out_path, order, records)

    rows = list(records.values())
    capped = sum(1 for r in rows if r.get("finish_reason") == "length")
    errored = sum(1 for r in rows if should_retry(r))
    lengths = [r["completion_tokens"] for r in rows if isinstance(r.get("completion_tokens"), int)]
    print(f"\ndone in {time.time() - start:.0f}s -> {out_path} ({len(rows)} rows)")
    print(f"  hit max_tokens: {capped} ({100 * capped / max(1, len(rows)):.1f}%)")
    print(f"  errored:        {errored}")
    if lengths:
        lengths.sort()
        print(
            f"  completion tokens: mean {sum(lengths) / len(lengths):.0f} "
            f"p50 {lengths[len(lengths) // 2]} p99 {lengths[int(0.99 * (len(lengths) - 1))]} "
            f"max {lengths[-1]}"
        )
    if capped:
        print(
            "  WARNING: responses hit the cap. Raise --max-tokens; a truncated "
            "response is graded on whatever prefix survived (ISSUE-003)."
        )


if __name__ == "__main__":
    main()
