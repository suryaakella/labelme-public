-- ForgeIndex Schema
-- Requires: pgvector extension

CREATE EXTENSION IF NOT EXISTS vector;

-- Videos: core entity
CREATE TABLE videos (
    id              SERIAL PRIMARY KEY,
    url             TEXT NOT NULL UNIQUE,
    platform        VARCHAR(32) NOT NULL,  -- youtube, instagram, tiktok
    title           TEXT,
    description     TEXT,
    duration_sec    FLOAT,
    width           INT,
    height          INT,
    fps             FLOAT,
    file_size_bytes BIGINT,
    storage_key     TEXT,                  -- MinIO object key
    thumbnail_key   TEXT,
    status          VARCHAR(32) NOT NULL DEFAULT 'pending',
        -- pending → downloading → processing → labeling → completed | failed
    error_message   TEXT,
    metadata        JSONB DEFAULT '{}',
    -- Rich Instagram metadata
    creator_username VARCHAR(256),
    creator_id       VARCHAR(256),
    creator_followers INT,
    creator_avatar_url TEXT,
    hashtags         JSONB DEFAULT '[]',
    mentions         JSONB DEFAULT '[]',
    music_info       JSONB,                -- {artist, title, album, url}
    language         JSONB,                -- {caption: "en", transcript: "en", sticker: "es"}
    sticker_texts    JSONB DEFAULT '[]',
    is_ad            BOOLEAN DEFAULT FALSE,
    is_ai_generated  BOOLEAN DEFAULT FALSE,
    posted_at        TIMESTAMPTZ,          -- original post date
    engagement       JSONB DEFAULT '{}',   -- {plays, likes, comments, shares, saves}
    analytics        JSONB DEFAULT '{}',   -- computed intelligence layer
    gdpr_flags       JSONB,
    current_step     TEXT,                 -- pipeline step in progress (e.g. "[3/9] Checking duplicates")
    task_id         INT,                  -- FK added after ingestion_tasks created
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_videos_status ON videos (status);
CREATE INDEX idx_videos_task_id ON videos (task_id);
CREATE INDEX idx_videos_analytics ON videos USING GIN (analytics);
CREATE INDEX idx_videos_platform ON videos (platform);
CREATE INDEX idx_videos_created ON videos (created_at DESC);
CREATE INDEX idx_videos_gdpr_flags ON videos USING GIN (gdpr_flags);

-- Video embeddings (CLIP ViT-B/32 → 512-dim)
CREATE TABLE video_embeddings (
    id          SERIAL PRIMARY KEY,
    video_id    INT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    embedding   vector(512) NOT NULL,
    model_name  VARCHAR(64) NOT NULL DEFAULT 'ViT-B-32',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(video_id, model_name)
);

CREATE INDEX idx_video_embeddings_ivfflat
    ON video_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Keyframes extracted from videos
CREATE TABLE keyframes (
    id          SERIAL PRIMARY KEY,
    video_id    INT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    frame_num   INT NOT NULL,
    timestamp   FLOAT NOT NULL,
    storage_key TEXT NOT NULL,
    embedding   vector(512),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_keyframes_video ON keyframes (video_id);

-- Object detections on keyframes
CREATE TABLE detections (
    id          SERIAL PRIMARY KEY,
    keyframe_id INT NOT NULL REFERENCES keyframes(id) ON DELETE CASCADE,
    video_id    INT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    label       VARCHAR(128) NOT NULL,
    confidence  FLOAT NOT NULL,
    bbox_x      FLOAT NOT NULL,
    bbox_y      FLOAT NOT NULL,
    bbox_w      FLOAT NOT NULL,
    bbox_h      FLOAT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_detections_video ON detections (video_id);
CREATE INDEX idx_detections_label ON detections (label);

-- Activity segments (derived from detection patterns)
CREATE TABLE activity_segments (
    id              SERIAL PRIMARY KEY,
    video_id        INT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    activity_type   VARCHAR(128) NOT NULL,
    start_time      FLOAT NOT NULL,
    end_time        FLOAT NOT NULL,
    confidence      FLOAT NOT NULL,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_activity_segments_video ON activity_segments (video_id);
CREATE INDEX idx_activity_segments_type ON activity_segments (activity_type);

-- Full transcripts
CREATE TABLE transcripts (
    id          SERIAL PRIMARY KEY,
    video_id    INT NOT NULL REFERENCES videos(id) ON DELETE CASCADE UNIQUE,
    full_text   TEXT NOT NULL,
    language    VARCHAR(16),
    model_name  VARCHAR(64) NOT NULL DEFAULT 'base',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_transcripts_video ON transcripts (video_id);

-- Timed transcript segments
CREATE TABLE transcript_segments (
    id              SERIAL PRIMARY KEY,
    transcript_id   INT NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    video_id        INT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start_time      FLOAT NOT NULL,
    end_time        FLOAT NOT NULL,
    text            TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_transcript_segments_video ON transcript_segments (video_id);

-- Annotations (combined JSONB summary per video)
CREATE TABLE annotations (
    id          SERIAL PRIMARY KEY,
    video_id    INT NOT NULL REFERENCES videos(id) ON DELETE CASCADE UNIQUE,
    data        JSONB NOT NULL DEFAULT '{}',
    version     INT NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_annotations_video ON annotations (video_id);
CREATE INDEX idx_annotations_data ON annotations USING GIN (data);

-- Tags for videos
CREATE TABLE video_tags (
    id          SERIAL PRIMARY KEY,
    video_id    INT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    tag         VARCHAR(128) NOT NULL,
    source      VARCHAR(32) NOT NULL DEFAULT 'auto',  -- auto | manual
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(video_id, tag)
);

CREATE INDEX idx_video_tags_video ON video_tags (video_id);
CREATE INDEX idx_video_tags_tag ON video_tags (tag);

-- Ingestion tasks
CREATE TABLE ingestion_tasks (
    id              SERIAL PRIMARY KEY,
    query           TEXT NOT NULL,
    platform        VARCHAR(32) NOT NULL DEFAULT 'youtube',
    max_results     INT NOT NULL DEFAULT 10,
    status          VARCHAR(32) NOT NULL DEFAULT 'pending',
        -- pending → running → completed | failed
    progress        FLOAT NOT NULL DEFAULT 0.0,
    total_videos    INT DEFAULT 0,
    processed_videos INT DEFAULT 0,
    error_message   TEXT,
    current_step    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ingestion_tasks_status ON ingestion_tasks (status);

-- Add FK from videos.task_id → ingestion_tasks (deferred because videos is created first)
ALTER TABLE videos ADD CONSTRAINT fk_videos_task_id FOREIGN KEY (task_id)
    REFERENCES ingestion_tasks(id) ON DELETE SET NULL;

-- Datasets (curated collections)
CREATE TABLE datasets (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(256) NOT NULL,
    description TEXT,
    filters     JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Dataset ↔ Video join table
CREATE TABLE dataset_videos (
    id          SERIAL PRIMARY KEY,
    dataset_id  INT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    video_id    INT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(dataset_id, video_id)
);

CREATE INDEX idx_dataset_videos_dataset ON dataset_videos (dataset_id);
CREATE INDEX idx_dataset_videos_video ON dataset_videos (video_id);

-- Duplicate pairs
CREATE TABLE duplicate_pairs (
    id          SERIAL PRIMARY KEY,
    video_id_a  INT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    video_id_b  INT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    similarity  FLOAT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(video_id_a, video_id_b),
    CHECK (video_id_a < video_id_b)
);

CREATE INDEX idx_duplicate_pairs_a ON duplicate_pairs (video_id_a);
CREATE INDEX idx_duplicate_pairs_b ON duplicate_pairs (video_id_b);

-- Platform comments (e.g. Instagram comments)
CREATE TABLE comments (
    id              SERIAL PRIMARY KEY,
    video_id        INT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    comment_id      VARCHAR(256),          -- platform comment ID
    username        VARCHAR(256),
    text            TEXT NOT NULL,
    like_count      INT DEFAULT 0,
    reply_count     INT DEFAULT 0,
    user_region     VARCHAR(16),
    language        VARCHAR(16),
    sentiment       JSONB,                 -- {compound, positive, negative, neutral, label}
    posted_at       TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_comments_video ON comments (video_id);
