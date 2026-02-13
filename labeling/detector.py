import logging
import os
from typing import List

from sqlalchemy import select

from config import settings
from config.database import get_session, Keyframe, Detection
from config.storage import storage

logger = logging.getLogger(__name__)

_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model: {settings.yolo_model}")
        _yolo_model = YOLO(settings.yolo_model)
    return _yolo_model


class ObjectDetector:
    def __init__(self):
        self.confidence = settings.yolo_confidence
        self.batch_size = settings.yolo_batch_size

    async def detect_for_video(self, video_id: int):
        async with get_session() as session:
            result = await session.execute(
                select(Keyframe).where(Keyframe.video_id == video_id).order_by(Keyframe.frame_num)
            )
            keyframes = result.scalars().all()

        if not keyframes:
            logger.warning(f"No keyframes for video {video_id}")
            return

        # Download keyframe images to temp and run detection
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="forgeindex_detect_")
        try:
            paths = []
            kf_map = {}
            for kf in keyframes:
                local_path = os.path.join(tmpdir, f"kf_{kf.id}.jpg")
                try:
                    storage.download_file(kf.storage_key, local_path)
                    paths.append(local_path)
                    kf_map[local_path] = kf
                except Exception as e:
                    logger.warning(f"Failed to download keyframe {kf.id}: {e}")

            if not paths:
                return

            # Run YOLO in batches
            all_detections = []
            model = _get_yolo()

            for i in range(0, len(paths), self.batch_size):
                batch = paths[i:i + self.batch_size]
                results = model(batch, conf=self.confidence, verbose=False)

                for path, det_result in zip(batch, results):
                    kf = kf_map[path]
                    boxes = det_result.boxes
                    if boxes is None:
                        continue
                    for j in range(len(boxes)):
                        cls_id = int(boxes.cls[j])
                        conf = float(boxes.conf[j])
                        x1, y1, x2, y2 = boxes.xyxyn[j].tolist()

                        label = det_result.names.get(cls_id, f"class_{cls_id}")

                        all_detections.append(Detection(
                            keyframe_id=kf.id,
                            video_id=video_id,
                            label=label,
                            confidence=conf,
                            bbox_x=x1,
                            bbox_y=y1,
                            bbox_w=x2 - x1,
                            bbox_h=y2 - y1,
                        ))

            # Store detections
            async with get_session() as session:
                for det in all_detections:
                    session.add(det)

            logger.info(f"Video {video_id}: {len(all_detections)} detections across {len(keyframes)} keyframes")

        finally:
            # Cleanup
            for f in os.listdir(tmpdir):
                try:
                    os.unlink(os.path.join(tmpdir, f))
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass
