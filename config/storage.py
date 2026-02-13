import io
import logging
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from config import settings

logger = logging.getLogger(__name__)


class StorageClient:
    def __init__(self):
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.bucket = settings.minio_bucket

    def ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
            logger.info(f"Created bucket: {self.bucket}")

    def upload_file(self, object_name: str, file_path: str, content_type: str = "application/octet-stream") -> str:
        self.client.fput_object(
            self.bucket, object_name, file_path, content_type=content_type,
        )
        logger.info(f"Uploaded {file_path} → {object_name}")
        return object_name

    def upload_bytes(self, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(
            self.bucket, object_name, io.BytesIO(data), length=len(data), content_type=content_type,
        )
        return object_name

    def download_file(self, object_name: str, file_path: str) -> str:
        self.client.fget_object(self.bucket, object_name, file_path)
        return file_path

    def get_url(self, object_name: str) -> str:
        scheme = "https" if settings.minio_secure else "http"
        return f"{scheme}://{settings.minio_endpoint}/{self.bucket}/{object_name}"

    def object_exists(self, object_name: str) -> bool:
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error:
            return False


storage = StorageClient()
