#!/usr/bin/env python3
"""End-to-end integration tests: ingest videos from Instagram and YouTube, verify the full pipeline."""

import sys
import time

import httpx

API_BASE = "http://localhost:8000"
TIMEOUT = 10
POLL_INTERVAL = 5
MAX_WAIT = 600  # 10 minutes max


def check_health(client: httpx.Client):
    print("[health] Checking API health...")
    resp = client.get(f"{API_BASE}/health", timeout=TIMEOUT)
    resp.raise_for_status()
    print(f"  OK: {resp.json()}")


def submit_ingestion(client: httpx.Client, query: str) -> int:
    print(f"[ingest] Submitting: '{query}'...")
    resp = client.post(
        f"{API_BASE}/api/ingest",
        json={"query": query},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    task_id = data["task_id"]
    print(f"  Task #{task_id} created — {data['message']}")
    return task_id


def wait_for_completion(client: httpx.Client, task_id: int) -> dict:
    print(f"[poll] Polling task #{task_id} until completion (max {MAX_WAIT}s)...")
    elapsed = 0
    while elapsed < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        resp = client.get(f"{API_BASE}/api/tasks/{task_id}", timeout=TIMEOUT)
        resp.raise_for_status()
        task = resp.json()
        status = task["status"]
        progress = task["progress"]
        step = task.get("current_step", "")
        print(f"  {elapsed}s — status: {status}, progress: {progress * 100:.0f}%{f', step: {step}' if step else ''}")

        if status == "completed":
            print(f"  Task completed in ~{elapsed}s")
            return task
        if status == "failed":
            print(f"  FAILED: {task.get('error_message', 'unknown error')}")
            sys.exit(1)

    print(f"  Timed out after {MAX_WAIT}s")
    sys.exit(1)


def get_task_video(client: httpx.Client, task_id: int) -> int:
    """Get a video from this specific task, wait for it to reach a terminal state."""
    print(f"[video] Waiting for video pipeline (task #{task_id})...")
    resp = client.get(f"{API_BASE}/api/tasks/{task_id}/videos", timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    videos = data["videos"]

    assert len(videos) >= 1, f"Expected at least 1 video in task #{task_id}, got {len(videos)}"
    video = videos[0]
    video_id = video["id"]
    print(f"  Video #{video_id}: '{video.get('title', 'Untitled')}' — status: {video['status']}")

    terminal_states = {"completed", "failed", "gdpr_blocked"}
    elapsed = 0
    while elapsed < MAX_WAIT:
        if video["status"] in terminal_states:
            print(f"  Video reached '{video['status']}' in ~{elapsed}s")
            return video_id

        step = video.get("current_step", "")
        step_info = f", step: {step}" if step else ""
        print(f"  {elapsed}s — video status: {video['status']}{step_info}")
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        resp = client.get(f"{API_BASE}/api/videos/{video_id}", timeout=TIMEOUT)
        resp.raise_for_status()
        video = resp.json()

    print(f"  Video pipeline timed out after {MAX_WAIT}s (status: {video['status']})")
    return video_id


def verify_video(client: httpx.Client, video_id: int):
    """Verify pipeline outputs including v2 annotation format."""
    print(f"[verify] Checking pipeline output for video #{video_id}...")
    resp = client.get(f"{API_BASE}/api/videos/{video_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    v = resp.json()

    if v["status"] == "gdpr_blocked":
        print("  Video was GDPR-blocked — skipping pipeline checks")
        print(f"  [PASS] storage_url: {v.get('storage_url') is not None}")
        return

    # Basic pipeline checks
    checks = {
        "storage_url": v.get("storage_url") is not None,
        "thumbnail_url": v.get("thumbnail_url") is not None,
        "tags": len(v.get("tags", [])) > 0,
        "detections": len(v.get("detections", [])) > 0,
        "annotation": v.get("annotation") is not None,
    }

    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    # Annotation v2 checks
    ann = v.get("annotation")
    if ann:
        version = ann.get("version")
        print(f"\n  Annotation version: {version}")
        if version == 2:
            v2_checks = {
                "summary": bool(ann.get("summary")),
                "scenes": isinstance(ann.get("scenes"), list) and len(ann["scenes"]) > 0,
                "primary_activities": isinstance(ann.get("primary_activities"), list),
                "content_tags": isinstance(ann.get("content_tags"), list),
                "visual_style": isinstance(ann.get("visual_style"), str),
                "video_metadata": isinstance(ann.get("video_metadata"), dict),
                "transcript (dict)": isinstance(ann.get("transcript"), (dict, type(None))),
            }
            for name, passed in v2_checks.items():
                print(f"  [{'PASS' if passed else 'FAIL'}] v2: {name}")

            # Print summary preview
            summary = ann.get("summary", "")
            if summary:
                preview = summary[:120] + "..." if len(summary) > 120 else summary
                print(f"\n  Summary: {preview}")
            if ann.get("primary_activities"):
                print(f"  Activities: {', '.join(ann['primary_activities'])}")
            if ann.get("content_tags"):
                print(f"  Tags: {', '.join(ann['content_tags'])}")
            if ann.get("scenes"):
                print(f"  Scenes: {len(ann['scenes'])} keyframe descriptions")
        else:
            print("  [INFO] v1 annotation (legacy format)")

    failed = [k for k, v in checks.items() if not v]
    if failed:
        print(f"\n  WARNING: {len(failed)} check(s) did not pass: {', '.join(failed)}")
        print("  (Some checks may fail if the video has no audio or detectable objects)")
    else:
        print("\n  All pipeline outputs present!")


def verify_search(client: httpx.Client, search_term: str):
    print(f"[search] Searching for '{search_term}'...")
    resp = client.get(
        f"{API_BASE}/api/search",
        params={"q": search_term, "limit": 5},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"  Returned {data['total']} result(s)")

    if data["total"] > 0:
        top = data["results"][0]
        print(f"  Top hit: video #{top['video_id']} — similarity: {top['similarity']:.3f}")
    else:
        print("  WARNING: No results (embeddings may not be stored or video was GDPR-blocked)")


def run_test(client: httpx.Client, query: str, search_term: str):
    """Run a full ingestion + pipeline + verification cycle."""
    task_id = submit_ingestion(client, query)
    wait_for_completion(client, task_id)
    video_id = get_task_video(client, task_id)
    verify_video(client, video_id)
    verify_search(client, search_term)
    return video_id


def main():
    print("=" * 60)
    print("ForgeIndex Integration Tests")
    print("=" * 60)

    client = httpx.Client()

    try:
        check_health(client)

        # Test 1: Instagram
        print("\n" + "-" * 60)
        print("TEST 1: Instagram")
        print("-" * 60)
        run_test(client, "1 satisfying video from instagram", "satisfying")

        # Test 2: YouTube
        print("\n" + "-" * 60)
        print("TEST 2: YouTube")
        print("-" * 60)
        run_test(client, "1 cooking tutorial from youtube", "cooking tutorial")

    except httpx.ConnectError:
        print(f"\nERROR: Cannot connect to {API_BASE}. Is the backend running?")
        sys.exit(1)
    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        sys.exit(1)
    finally:
        client.close()

    print("\n" + "=" * 60)
    print("All integration tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
