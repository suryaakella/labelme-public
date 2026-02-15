# ForgeIndex

Video annotation platform that discovers, downloads, and analyzes video content using ML and LLMs. It extracts keyframes, generates CLIP embeddings, runs object detection, transcribes audio, and produces research-grade structured annotations powered by multimodal Gemini — all searchable via semantic search.

## Getting Started

### Prerequisites

- Docker & Docker Compose
- API keys:
  - `APIFY_API_TOKEN` — **required** for video discovery from Instagram, YouTube, TikTok
  - `GOOGLE_API_KEY` — **recommended** for Gemini LLM (annotations, GDPR screening, query parsing, smart hashtag discovery)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/suryaakella/Shofomvp1.git
cd Shofomvp1

# 2. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 3. Start all services
docker compose up --build

# 4. Verify everything is running
curl http://localhost:8000/health
# {"status": "ok"}
```

Open http://localhost:3000 to use the UI.

### Services

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | React UI |
| Backend API | http://localhost:8000 | FastAPI |
| MinIO Console | http://localhost:9001 | Object storage UI (minioadmin/minioadmin) |
| PostgreSQL | localhost:5433 | Database |

## How It Works

### 1. Query Parsing (LangChain + Gemini)

The user types a natural language query. LangChain + Gemini parses it into structured intent:

```
"5 cooking videos from yt"  →  platform=youtube,  max_results=5,  topic=cooking
"dancing on insta"          →  platform=instagram, max_results=10, topic=dancing
"robot walking"             →  platform=instagram, max_results=10, topic=robot walking
```

- Default platform is **Instagram** — say "from youtube" or "yt" to switch
- Understands slang: `insta`, `ig`, `yt`, `tt`, `reels`
- Falls back to regex parsing if no Google API key

### 2. GDPR Query Screening

Before any video is discovered, the query is screened by Gemini for GDPR compliance:

- **Blocked**: Queries targeting specific private individuals (e.g., "John Smith's home videos")
- **Allowed**: Topics, brands, categories, public figures in educational context (e.g., "cooking tutorials", "Nike commercials")
- Fail-closed: if screening can't run, ingestion is still allowed but videos are quarantined later

### 3. Smart Video Discovery (Apify + LLM Retry)

**Instagram** uses an LLM-powered smart retry loop:

```
Round 1: "robot walking" → search hashtags [#robotwalking, #robot, #walking]
         → 0 results?

Round 2: LLM suggests related hashtags → [#robotics, #mechanicalart, #robotdog, ...]
         → search those → found videos?

Round 3: LLM suggests more alternatives (excluding already-tried)
         → keeps going until enough videos found or 3 rounds exhausted
```

**YouTube / TikTok** use direct keyword search via Apify.

URL deduplication runs across all rounds — same video never appears twice.

### 4. Download & Storage

- Videos downloaded via yt-dlp (handles Instagram CDN, YouTube, TikTok)
- Video files + thumbnails uploaded to MinIO (S3-compatible)
- Rich metadata extracted: creator info, hashtags, mentions, engagement metrics, comments

### 5. ML Pipeline (9 Steps)

Each video runs through a 9-step pipeline:

| Step | What it does |
|------|-------------|
| 1. **Keyframe Extraction** | FFmpeg extracts frames at 1fps + scene-change detection (capped at 100 max) |
| 2. **CLIP Embeddings** | ViT-B/32 generates 512-dim embeddings per keyframe + video-level avg pool |
| 3. **Duplicate Check** | Cosine similarity against existing videos (threshold: 0.95) |
| 4. **Whisper Transcription** | Audio → text with timed segments and language detection |
| 5. **Enrichment** | Language detection (langdetect), OCR on keyframes (EasyOCR), AI-generated content heuristic |
| 6. **Object Detection** | YOLOv8 runs on keyframes, stores bounding boxes + labels |
| 7. **LLM Annotation** | Gemini multimodal analyzes keyframe images + transcript (see below) |
| 8. **GDPR PII Scan** | Scans all text fields for personal data — quarantines if found |
| 9. **Analytics** | Engagement tier, brand safety score, sentiment analysis |

### 6. LLM-Powered Multimodal Annotation (Step 7)

This is the core differentiator. Instead of hardcoded rules, we send 4-6 representative keyframe images + transcript to Gemini and get back structured, research-grade annotations:

```json
{
  "version": 2,
  "summary": "A street food vendor in Bangkok prepares pad thai...",
  "scenes": [
    {
      "timestamp_sec": 0.0,
      "description": "Close-up of a wok over high flame with oil sizzling",
      "objects_in_context": ["wok", "gas burner", "cooking oil"],
      "activities": ["heating wok", "preparing cooking station"]
    }
  ],
  "primary_activities": ["cooking", "street vending"],
  "content_tags": ["thai-food", "street-food", "cooking"],
  "visual_style": "Close-up shots, warm lighting, handheld camera",
  "transcript_context": "Narrator explains each cooking step",
  "video_metadata": { "title": "...", "platform": "instagram", "duration_sec": 45.0 },
  "platform_context": { "creator": {...}, "engagement": {...}, "hashtags": [...] },
  "transcript": { "full_text": "...", "language": "en", "segments": [...] }
}
```

**Tags** are computed from 4 sources combined:
- LLM content tags (e.g., `thai-food`, `street-food`)
- LLM activities with prefix (e.g., `activity:cooking`, `activity:street-vending`)
- CLIP zero-shot classification against 20 tag candidates
- Platform tag (e.g., `platform:instagram`)

Falls back to rule-based annotation if no Google API key.

### 7. GDPR PII Pipeline Scan (Step 8)

After annotation, all text fields are scanned for personally identifiable information:

**Fields scanned**: title, description, transcript, annotation summary, comments, sticker texts, creator username, mentions

**Outcomes**:
- **Clean** → video proceeds to analytics and completion
- **PII detected** → video quarantined (`gdpr_blocked`), never served to users. PII types logged (PERSON_NAME, EMAIL, PHONE, etc.)
- **Unverified** → no API key or no annotation available → quarantined (fail-closed)
- **Error** → scan failed → quarantined (fail-closed)

The pipeline stops after GDPR blocking — analytics never runs on quarantined data.

### 8. Semantic Search

Text queries are embedded with CLIP and matched against video embeddings using pgvector's IVFFlat index (cosine distance). Returns ranked results with similarity scores.

```
GET /api/search?q=street food cooking&limit=10
```

## Frontend

### Dashboard (`/`)
- Stats overview (videos in library, processing, failed, GDPR blocked)
- Ingestion form with real-time progress (animated step indicators)
- Task history table with status badges:
  - Hover over any status to see details (error message, GDPR reason, etc.)
  - Orange "GDPR blocked" badge when videos were quarantined
  - Error messages shown inline for failed/partially-completed tasks

### Video Detail (`/videos/:id`)
- Video player + basic metadata
- **Summary card** — LLM-generated description
- **Scene timeline** — per-keyframe descriptions with timestamps, objects, activities
- **Activity tags** — pills from primary_activities
- **Content tags** — auto-generated tags
- **Visual style** — production style description
- **Transcript** — collapsible full text with segments
- **Platform context** — collapsible engagement/creator/hashtags
- **GDPR banner** — green (compliant), red (blocked with PII types), yellow (unverified)
- **Analytics** — engagement tier, brand safety, sentiment
- **Download annotation JSON** button
- Backward compatible: v1 annotations show raw JSON with "Legacy format" badge

### Task Detail (`/tasks/:id`)
- Real-time progress tracking with step indicators
- Video list with status badges and GDPR indicators

## API Endpoints

### Ingestion
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/ingest` | Submit discovery task (`{"query": "..."}`) |

### Videos
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/videos` | List videos (paginated, filterable by platform/status/tag) |
| GET | `/api/videos/:id` | Full video detail with annotations |
| GET | `/api/videos/:id/thumbnail` | Stream thumbnail image |
| GET | `/api/videos/:id/file` | Stream video file |

### Search
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/search?q=...` | Semantic search (`limit`, `platform`, `min_similarity`) |

### Tasks
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/tasks` | List ingestion tasks |
| GET | `/api/tasks/:id` | Task detail with progress |
| GET | `/api/tasks/:id/videos` | Videos from a task |

### Datasets
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/datasets` | Create filtered dataset |
| GET | `/api/datasets` | List datasets |
| GET | `/api/datasets/:id/export` | Export dataset as ZIP |

### System
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/stats` | System statistics |

## SOTA Model Upgrades

ForgeIndex ships with lightweight, CPU-friendly defaults. If you have GPU resources, each component can be swapped for a state-of-the-art alternative:

### Embeddings — CLIP ViT-B/32 (512-dim)

| Alternative | What's better | Link |
|-------------|--------------|------|
| **SigLIP 2** (Google) | Sigmoid loss, better zero-shot accuracy, multilingual (109 languages), Apache 2.0 | [google-research/big_vision](https://github.com/google-research/big_vision) |
| **DFN-CLIP ViT-H** (Apple) | 84.4% zero-shot ImageNet via aggressive data filtering; highest accuracy open-source CLIP | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) |
| **MetaCLIP** (Meta) | Fully transparent data curation from CommonCrawl; reproducible training pipeline | [facebookresearch/MetaCLIP](https://github.com/facebookresearch/MetaCLIP) |

### Object Detection — YOLOv8n (bounding boxes, 80 COCO classes)

| Alternative | What's better | Link |
|-------------|--------------|------|
| **Grounding DINO** (IDEA Research) | Open-vocabulary detection — detect *any* object by text prompt, no retraining needed | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |
| **SAM 2** (Meta) | Promptable segmentation in images *and* video with cross-frame tracking; pair with Grounding DINO for detect+segment | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| **Florence-2** (Microsoft) | Single lightweight model (0.23B params) for detection + captioning + OCR via text prompts; MIT license | [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) |

### Transcription — Whisper base

| Alternative | What's better | Link |
|-------------|--------------|------|
| **Faster Whisper** (SYSTRAN) | CTranslate2-based, 4x faster than OpenAI Whisper with same accuracy, int8 quantization | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| **Whisper large-v3-turbo** (OpenAI) | Same encoder as large-v3 with only 4 decoder layers — dramatically faster, near drop-in | [openai/whisper](https://github.com/openai/whisper) |
| **NVIDIA Canary-Qwen-2.5B** | Tops HuggingFace Open ASR Leaderboard (5.63% WER), supports 25 languages | [NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo) |

### OCR — EasyOCR

| Alternative | What's better | Link |
|-------------|--------------|------|
| **PaddleOCR v5** (Baidu) | Faster, more accurate, 100+ languages, actively maintained with frequent releases | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| **GOT-OCR 2.0** | Unified end-to-end generative model (580M params) for plain text, formatted text, and region OCR | [Ucas-HaoranWei/GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0) |
| **DocTR** (Mindee) | Clean Python API, PyTorch + TensorFlow backends, MIT license | [mindee/doctr](https://github.com/mindee/doctr) |

### Multimodal LLM — Gemini 2.0 Flash (API)

| Alternative | What's better | Link |
|-------------|--------------|------|
| **Qwen2.5-VL-7B** (Alibaba) | Open-weight, runs on single GPU, native video understanding + bounding box output, Apache 2.0 | [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) |
| **GPT-4o** (OpenAI) | Strong multimodal reasoning, extensive structured output support | [platform.openai.com](https://platform.openai.com) |
| **Claude Sonnet/Opus** (Anthropic) | Excellent structured output + instruction following, 200K context window | [docs.anthropic.com](https://docs.anthropic.com) |

### Recommended upgrade path

| Current | Upgrade to | Key benefit |
|---------|-----------|-------------|
| CLIP ViT-B/32 | SigLIP 2 | Better embeddings, multilingual, open-weight |
| YOLOv8n | Grounding DINO + SAM 2 | Open-vocabulary detection + pixel-level segmentation |
| Whisper base | Faster Whisper + large-v3-turbo | 4x speed + much better accuracy, drop-in swap |
| EasyOCR | PaddleOCR v5 | Faster, more accurate, 100+ languages |
| Gemini 2.0 Flash | Qwen2.5-VL-7B (self-hosted) | No API cost, open-weight, native video support |

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌───────────┐
│   Frontend   │────>│   Backend    │────>│   PostgreSQL     │     │   MinIO   │
│  React/Vite  │     │   FastAPI    │     │   + pgvector     │     │ (S3 store)│
│  Tailwind    │     │              │────>│                  │     │           │
│  port :3000  │     │  port :8000  │     │   port :5433     │     │ port :9000│
└──────────────┘     └──────┬───────┘     └──────────────────┘     └─────┬─────┘
                            │                                            │
                            │         ┌──────────────────┐               │
                            └────────>│   ML Pipeline    │───────────────┘
                                      │ CLIP · YOLO      │
                                      │ Whisper · EasyOCR │
                                      │ Gemini (LLM)     │
                                      └──────────────────┘
```

| Layer | Tech |
|-------|------|
| Frontend | React 18, Vite, Tailwind CSS, React Router v6 |
| Backend | FastAPI, SQLAlchemy (async), Pydantic |
| Database | PostgreSQL 16 + pgvector (512-dim CLIP vectors) |
| Storage | MinIO (S3-compatible) — videos, keyframes, thumbnails |
| ML | CLIP ViT-B/32, YOLOv8n, Whisper base, EasyOCR, langdetect |
| LLM | Gemini 2.0 Flash (annotations, GDPR, query parsing, hashtag discovery) via LangChain |
| Discovery | Apify (Instagram/YouTube/TikTok), yt-dlp fallback |

## Database Schema

14 tables covering the full data model:

- **videos** — Core entity with rich metadata (creator info, hashtags, engagement, GDPR flags, analytics)
- **video_embeddings** — CLIP vectors (512-dim, IVFFlat indexed)
- **keyframes** — Extracted frames with individual embeddings
- **detections** — YOLO bounding boxes per keyframe
- **activity_segments** — Temporal activity recognition
- **transcripts** / **transcript_segments** — Whisper output with timed segments
- **annotations** — Structured JSONB annotation per video (v1 rule-based, v2 LLM-powered)
- **video_tags** — Auto-generated tags from LLM + CLIP + platform
- **ingestion_tasks** — Background task tracking with progress
- **datasets** / **dataset_videos** — Curated collections with filtered export
- **duplicate_pairs** — Dedup results with similarity scores
- **comments** — Platform comments (username, text, likes, replies)

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `APIFY_API_TOKEN` | — | Required for video discovery |
| `GOOGLE_API_KEY` | — | Gemini LLM for annotations, GDPR, parsing |
| `ANNOTATION_LLM_MODEL` | `gemini-2.0-flash` | Vision-capable model for multimodal annotation |
| `GDPR_LLM_MODEL` | `gemini-2.0-flash-lite` | Lightweight model for text-only GDPR/parsing |
| `CLIP_MODEL` | `ViT-B-32` | CLIP model variant |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model (nano) |
| `WHISPER_MODEL` | `base` | Whisper model size |
| `KEYFRAME_INTERVAL` | `30` | Frames between keyframe samples |
| `MAX_KEYFRAMES` | `100` | Max keyframes per video (subsampled if exceeded) |
| `SCENE_THRESHOLD` | `0.3` | Scene change sensitivity |
| `DEDUP_THRESHOLD` | `0.95` | Cosine similarity for duplicate detection |
| `YOLO_CONFIDENCE` | `0.3` | Min confidence for detections |
| `VIDEO_MAX_RESOLUTION` | `720` | Max video resolution for processing |
| `MAX_WORKERS` | `4` | Pipeline thread pool size |
| `GDPR_QUERY_CHECK_ENABLED` | `true` | Enable pre-ingestion GDPR screening |
| `GDPR_PIPELINE_CHECK_ENABLED` | `true` | Enable post-annotation PII scanning |

## Project Structure

```
ForgeIndex/
├── api/
│   ├── main.py              # FastAPI app, lifespan, CORS
│   └── routes/
│       ├── ingest.py         # POST /api/ingest (query parse + GDPR screen)
│       ├── search.py         # GET /api/search (semantic search)
│       ├── videos.py         # GET /api/videos, /api/videos/:id
│       ├── datasets.py       # Dataset CRUD + ZIP export
│       └── tasks.py          # Task listing + detail + videos
├── config/
│   ├── __init__.py           # Pydantic settings (all env vars)
│   ├── database.py           # ORM models, async engine, session mgmt
│   ├── storage.py            # MinIO client
│   └── task_queue.py         # Background thread pool
├── ingestion/
│   ├── discovery.py          # Apify discovery + LLM smart hashtag retry
│   ├── downloader.py         # yt-dlp download + MinIO upload
│   ├── orchestrator.py       # Discovery → download → pipeline flow
│   └── query_parser.py       # LangChain query parsing (NL → structured intent)
├── pipeline/
│   ├── orchestrator.py       # 9-step ML pipeline orchestrator
│   ├── frames.py             # Keyframe extraction (ffmpeg, max 100)
│   ├── embeddings.py         # CLIP embedding generation
│   ├── dedup.py              # Duplicate detection via cosine similarity
│   └── transcription.py      # Whisper transcription with segments
├── labeling/
│   ├── llm_annotator.py      # Gemini multimodal annotation (v2)
│   ├── annotator.py          # Rule-based annotation fallback (v1)
│   ├── detector.py           # YOLOv8 object detection
│   ├── enrichment.py         # Language detection, OCR, AI-gen flag
│   ├── gdpr.py               # GDPR query screening + PII pipeline scan
│   └── analytics.py          # Engagement, sentiment, brand safety
├── frontend/
│   └── src/
│       ├── App.jsx           # Router + navigation
│       ├── api.js            # API client
│       └── pages/
│           ├── IndexPage.jsx       # Dashboard + ingestion + task history
│           ├── VideoDetailPage.jsx  # Full video annotation view
│           ├── TaskDetailPage.jsx   # Task progress + video list
│           ├── VideosPage.jsx       # Video library browser
│           └── SearchPage.jsx       # Semantic search
├── migrations/
│   └── init.sql              # Full database schema (14 tables)
├── scripts/
│   └── seed.py               # Integration tests (Instagram + YouTube)
├── tests/                    # Pytest unit tests
├── docker-compose.yml        # Full stack orchestration
├── Dockerfile.backend        # Python 3.11 + ffmpeg + ML deps
├── Dockerfile.frontend       # Node build + Nginx serve
├── Dockerfile.postgres       # pgvector + init.sql
├── nginx.conf                # Frontend reverse proxy config
└── .env.example              # Environment variable template
```

## Testing

```bash
# Unit tests
pytest tests/

# Integration tests (requires running stack)
python scripts/seed.py
```

The integration test submits Instagram and YouTube ingestion tasks, waits for pipeline completion, and verifies:
- Video discovery and download
- v2 annotation format (summary, scenes, activities, tags)
- Semantic search results
- GDPR compliance handling

## Deployment

For production deployment with API keys:

1. Use your hosting platform's secrets/environment management (Render, Railway, AWS ECS, etc.)
2. Set `APIFY_API_TOKEN` and `GOOGLE_API_KEY` as secrets
3. Never commit `.env` files — use `.env.example` as a template

## License

MIT
