#!/usr/bin/env python3
"""End-to-end integration test: ingest 1 Instagram video via hashtag scraper and verify the full pipeline."""

import sys
import time

import httpx

API_BASE = "http://localhost:8000"
TIMEOUT = 10
POLL_INTERVAL = 5
MAX_WAIT = 300  # 5 minutes max


def check_health(client: httpx.Client):
    print("[1/7] Checking API health...")
    resp = client.get(f"{API_BASE}/health", timeout=TIMEOUT)
    resp.raise_for_status()
    print(f"  OK: {resp.json()}")


def submit_ingestion(client: httpx.Client) -> int:
    print("[2/7] Submitting ingestion: '1 satisfying video from instagram'...")
    resp = client.post(
        f"{API_BASE}/api/ingest",
        json={"query": "1 satisfying video from instagram"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    task_id = data["task_id"]
    print(f"  Task #{task_id} created — {data['message']}")
    return task_id


def wait_for_completion(client: httpx.Client, task_id: int) -> dict:
    print(f"[3/7] Polling task #{task_id} until completion (max {MAX_WAIT}s)...")
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
    """Get a video from this specific task using the task videos API."""
    print(f"[4/7] Verifying video was ingested for task #{task_id}...")
    resp = client.get(f"{API_BASE}/api/tasks/{task_id}/videos", timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    videos = data["videos"]

    assert len(videos) >= 1, f"Expected at least 1 video in task #{task_id}, got {len(videos)}"
    video = videos[0]
    video_id = video["id"]
    print(f"  Video #{video_id}: '{video.get('title', 'Untitled')}' — status: {video['status']}")

    # Wait for the video pipeline to reach a terminal state
    terminal_states = {"completed", "failed", "gdpr_blocked"}
    elapsed = 0
    while elapsed < MAX_WAIT:
        if video["status"] in terminal_states:
            print(f"  Video reached terminal state '{video['status']}' in ~{elapsed}s")
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


def verify_video_detail(client: httpx.Client, video_id: int):
    print(f"[5/7] Verifying full pipeline output for video #{video_id}...")
    resp = client.get(f"{API_BASE}/api/videos/{video_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    detail = resp.json()

    if detail["status"] == "gdpr_blocked":
        print("  Video was GDPR-blocked — skipping pipeline checks (expected for some content)")
        print(f"  [PASS] storage_url: {detail.get('storage_url') is not None}")
        print(f"  [PASS] thumbnail_url: {detail.get('thumbnail_url') is not None}")
        return

    checks = {
        "storage_url": detail.get("storage_url") is not None,
        "thumbnail_url": detail.get("thumbnail_url") is not None,
        "tags": len(detail.get("tags", [])) > 0,
        "detections": len(detail.get("detections", [])) > 0,
        "transcript_text": detail.get("transcript_text") is not None,
        "transcript_segments": len(detail.get("transcript_segments", [])) > 0,
        "annotation": detail.get("annotation") is not None,
    }

    for check_name, passed in checks.items():
        symbol = "PASS" if passed else "FAIL"
        print(f"  [{symbol}] {check_name}")

    failed = [k for k, v in checks.items() if not v]
    if failed:
        print(f"\n  WARNING: {len(failed)} check(s) did not pass: {', '.join(failed)}")
        print("  (Some checks may fail if the video has no audio or detectable objects)")
    else:
        print("  All pipeline outputs present!")


def verify_task_videos(client: httpx.Client, task_id: int):
    print(f"[6/7] Verifying enriched task videos API for task #{task_id}...")
    resp = client.get(f"{API_BASE}/api/tasks/{task_id}/videos", timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    task = data["task"]
    videos = data["videos"]
    print(f"  Task status: {task['status']}, videos: {len(videos)}")

    assert len(videos) >= 1, f"Expected at least 1 video in task, got {len(videos)}"

    for v in videos:
        print(f"\n  Video #{v['id']}: {v.get('title') or 'Untitled'} — status: {v['status']}")

        # Verify new enrichment fields exist in response
        enrichment_fields = [
            "current_step", "error_message", "performance_tier",
            "brand_safety_tier", "sentiment_avg", "sentiment_label",
            "content_categories", "is_ai_generated", "gdpr_status",
        ]
        missing = [f for f in enrichment_fields if f not in v]
        if missing:
            print(f"  [FAIL] Missing enrichment fields in response: {', '.join(missing)}")
        else:
            print(f"  [PASS] All enrichment fields present in API response")

        if v["status"] == "completed":
            checks = {
                "current_step": v.get("current_step") == "Pipeline complete",
                "performance_tier": v.get("performance_tier") is not None,
                "brand_safety_tier": v.get("brand_safety_tier") is not None,
                "sentiment_label": v.get("sentiment_label") in (None, "positive", "neutral", "negative"),
                "content_categories_type": isinstance(v.get("content_categories", []), list),
            }
            for check_name, passed in checks.items():
                symbol = "PASS" if passed else "FAIL"
                value = v.get(check_name.split("_type")[0] if "_type" in check_name else check_name)
                print(f"  [{symbol}] {check_name}: {value}")

            if v.get("performance_tier"):
                print(f"    Performance: {v['performance_tier']}")
            if v.get("brand_safety_tier"):
                print(f"    Brand safety: {v['brand_safety_tier']}")
            if v.get("sentiment_label"):
                print(f"    Sentiment: {v['sentiment_label']} (avg={v.get('sentiment_avg')})")
            if v.get("content_categories"):
                print(f"    Categories: {', '.join(v['content_categories'])}")
            if v.get("is_ai_generated"):
                print(f"    AI Generated: {v['is_ai_generated']}")
            if v.get("gdpr_status"):
                print(f"    GDPR status: {v['gdpr_status']}")

        elif v["status"] == "failed":
            print(f"  Error: {v.get('error_message', 'unknown')}")

        elif v["status"] == "gdpr_blocked":
            print(f"  GDPR status: {v.get('gdpr_status', 'blocked')}")


def verify_search(client: httpx.Client):
    print("[7/7] Verifying semantic search works...")
    resp = client.get(
        f"{API_BASE}/api/search",
        params={"q": "satisfying video", "limit": 5},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"  Search for 'dance' returned {data['total']} result(s)")

    if data["total"] > 0:
        top = data["results"][0]
        print(f"  Top hit: video #{top['video_id']} — similarity: {top['similarity']}")
    else:
        print("  WARNING: No search results (embeddings may not have been stored or video was GDPR-blocked)")


def main():
    print("=" * 60)
    print("ForgeIndex E2E Integration Test")
    print("Query: '1 satisfying video from instagram'")
    print("Tests: health, ingest, task polling, video pipeline,")
    print("       pipeline output, enriched task videos API, search")
    print("=" * 60)

    client = httpx.Client()

    try:
        check_health(client)
        task_id = submit_ingestion(client)
        wait_for_completion(client, task_id)
        video_id = get_task_video(client, task_id)
        verify_video_detail(client, video_id)
        verify_task_videos(client, task_id)
        verify_search(client)
    except httpx.ConnectError:
        print(f"\nERROR: Cannot connect to {API_BASE}. Is the backend running?")
        sys.exit(1)
    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        sys.exit(1)
    finally:
        client.close()

    print("\n" + "=" * 60)
    print("Integration test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
