Build me an end-to-end video dataset pipeline called "Forgeindex" — an automated system where a user submits a search query (e.g. "500 videos of CNC machining"), the system fetches videos from Instagram/YouTube/TikTok, processes them through ML models, labels everything, and serves the results asynchronously.
Architecture
Python backend, FastAPI, PostgreSQL + pgvector for semantic search, MinIO for S3-compatible local object storage. React frontend. Docker Compose for the full stack.
The Flow

User sends POST /api/ingest with a query like "CNC machining" and max_videos=500
System uses Apify API (with yt-dlp as fallback) to discover and download videos from the requested platform
Each video goes through: download → extract keyframes → CLIP embed → YOLO detect → activity segment → Whisper transcribe → store annotations
User polls task status or gets notified when processing is complete
User can search semantically, browse videos with annotations, and export filtered datasets as ZIP

Project Structure
forgeindex/
├── docker-compose.yml          # postgres, minio, backend, frontend
├── Dockerfile.backend          # Python 3.11 + ffmpeg + ML libs
├── requirements.txt
├── migrations/init.sql          # PostgreSQL schema with pgvector
├── config/
│   ├── __init__.py             # Settings via pydantic-settings (env vars)
│   ├── database.py             # SQLAlchemy async engine + ORM models
│   ├── storage.py              # MinIO helpers (upload, download, get_url)
│   └── task_queue.py           # Simple threaded task queue (no Celery)
├── ingestion/
│   ├── discovery.py            # Apify API search (YouTube/TikTok/Instagram actors) + yt-dlp fallback
│   ├── downloader.py           # yt-dlp downloads video+thumbnail → uploads to MinIO
│   └── orchestrator.py         # Coordinates discovery → download → trigger pipeline
├── pipeline/
│   ├── frames.py               # ffmpeg keyframe extraction (1fps + scene change detection threshold=0.3)
│   ├── embeddings.py           # OpenCLIP ViT-B/32 embeddings (512-dim), lazy-loaded singleton
│   ├── dedup.py                # Cosine similarity >0.95 via pgvector = duplicate
│   ├── transcription.py        # ffmpeg extract audio to WAV → Whisper base model → timestamped segments
│   └── orchestrator.py         # Runs full pipeline: frames→embed→dedup→transcribe→label→annotate
├── labeling/
│   ├── detector.py             # YOLOv8n on keyframes, COCO→manufacturing label mapping
│   ├── activity.py             # Rule-based temporal activity recognition from YOLO detections
│   └── annotator.py            # Builds final annotation JSON, auto-generates tags
├── api/
│   ├── main.py                 # FastAPI app, CORS, lifespan, /health, /stats
│   └── routes/
│       ├── ingest.py           # POST /api/ingest → starts ingestion task
│       ├── search.py           # GET /api/search → CLIP text embedding → pgvector similarity
│       ├── videos.py           # GET /api/videos (paginated), GET /api/videos/{id} (full detail)
│       ├── datasets.py         # POST /api/dataset (create), GET /api/dataset/{id}/export (ZIP)
│       └── tasks.py            # GET /api/tasks/{id} (poll progress), GET /api/tasks (list)
├── frontend/                   # React + Vite + Tailwind
│   └── src/
│       ├── App.jsx             # Router: /, /search, /videos/:id
│       ├── api.js              # Fetch wrapper for all API calls
│       └── pages/
│           ├── IndexPage.jsx   # Dashboard: total videos, category breakdown, recent ingestions
│           ├── SearchPage.jsx  # Search bar + video grid with thumbnails, tags, scores
│           └── VideoDetailPage.jsx  # Video player + timeline annotations + metadata
└── scripts/
    └── seed.py                 # Ingests 20 manufacturing videos across 5 categories as demo
Database Schema (PostgreSQL + pgvector)
Tables needed:

videos — source_url (unique), platform, title, description, duration, resolution, storage_key, thumbnail_key, status enum (pending→downloading→processing→labeling→completed→failed), created_at
video_embeddings — video_id FK, embedding vector(512), model_name, created_at. Add ivfflat index for ANN search.
keyframes — video_id FK, timestamp float, frame_index int, storage_key, is_scene_change bool
detections — keyframe_id FK, label, confidence float, bbox_x/y/w/h (normalized 0-1), model_name
activity_segments — video_id FK, label, start_time, end_time, confidence, metadata JSONB
transcripts — video_id FK, full_text, language, model_name
transcript_segments — transcript_id FK, start_time, end_time, text
annotations — video_id FK (unique), data JSONB, created_at. GIN indexes on data->'objects_detected' and data->'tags'
video_tags — video_id FK, tag, source (auto/manual)
ingestion_tasks — query, platform, status, max_videos, videos_found, videos_processed, error_message, timestamps
datasets — name, description, filters JSONB, video_count, created_at
dataset_videos — dataset_id FK, video_id FK
duplicate_pairs — video_id_a FK, video_id_b FK, similarity float

Key Implementation Details
Ingestion

Apify: Use YouTube search actor apify/youtube-scraper and Instagram actor apify/instagram-scraper. Fall back to yt-dlp search if no APIFY_API_TOKEN.
yt-dlp: Download best quality ≤720p, extract thumbnail, upload both to MinIO.
All ingestion runs in background threads via the task queue.

Pipeline

ffmpeg keyframes: select='not(mod(n,30))+gt(scene,0.3)' for 1fps + scene changes
CLIP: Use open_clip library, ViT-B/32 pretrained on openai. Lazy-load as singleton. For video embedding, average-pool all keyframe embeddings.
Dedup: After embedding a video, query pgvector for cosine similarity >0.95. Flag duplicates but don't delete.
Whisper: Extract audio with ffmpeg (16kHz mono WAV), run openai-whisper base model locally. Store full text + timed segments.

Labeling

YOLOv8n: Run on keyframes in batches of 16, confidence threshold 0.3. Map COCO labels to manufacturing domain:

laptop → control_panel, scissors → cutting_tool, cell phone → handheld_device
person stays person, bicycle → conveyor (rough mapping)


Activity recognition: Rule-based on frame-level object sets:

person + hand_tool → "tool_usage"
person + control_panel → "machine_operation"
person + safety_equipment → "safety_check"
no person + machine → "automated_process"
Merge consecutive same-label frames, filter by min_duration (2 seconds)


Annotation builder: Compile objects_detected, actions, transcript, tags into one JSONB blob. Auto-tag from keywords (cnc, welding, lathe, 3d printing, etc.) + activity labels + detected objects.

API

Search: Text query → CLIP text embedding → pgvector cosine distance query → return ranked results with scores
Dataset export: Filter videos by activities, objects, confidence, duration → create ZIP containing annotation JSONs + video metadata CSV + taxonomy YAML + README

Frontend

Dark industrial theme (grays, amber accents)
Index page: Stats cards, recent ingestion tasks with status badges, quick search
Search page: Search bar, filter sidebar, video grid with thumbnails/tags/duration
Video detail: Video player, clickable timeline showing activity segments color-coded, object detection summary, transcript with timestamps, raw annotation JSON viewer

Tech Stack

Python 3.11, FastAPI, SQLAlchemy (async), asyncpg
PostgreSQL 16 + pgvector
MinIO
ffmpeg, yt-dlp, apify-client
open-clip-torch, openai-whisper, ultralytics (YOLOv8)
React 18, Vite, Tailwind CSS, react-router-dom, lucide-react
Docker Compose

Constraints

Everything runs locally first, no cloud dependencies except optional Apify API token
Background processing via Python threading (not Celery) — use a simple TaskQueue class with ThreadPoolExecutor
Modular design: each pipeline step is a separate class that can be swapped out
Video status progresses through: pending → downloading → processing → labeling → completed (or failed)
Include a seed script that ingests 20 videos across 5 manufacturing categories (CNC, welding, assembly, 3D printing, lathe)
Vite dev server proxies /api to backend:8000

Environment Variables

DATABASE_URL, DATABASE_URL_SYNC
MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE, MINIO_PUBLIC_URL
APIFY_API_TOKEN (optional)
WHISPER_MODEL=base, CLIP_MODEL=ViT-B-32, CLIP_PRETRAINED=openai, YOLO_MODEL=yolov8n.pt
MAX_VIDEO_DURATION=600, LOG_LEVEL=info

Build the backend first (config, ingestion, pipeline, labeling, API), then the frontend. Make sure docker-compose up starts everything.