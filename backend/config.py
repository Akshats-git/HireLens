"""
API settings sourced from configs/config.yaml.

Read once at import so request handlers do not touch the filesystem.
"""

from src.config import get_section

_api = get_section("api")

MAX_UPLOAD_BYTES: int = _api["max_upload_size_mb"] * 1024 * 1024
MAX_UPLOAD_SIZE_MB: int = _api["max_upload_size_mb"]
MAX_BULK_RESUMES: int = _api["max_bulk_resumes"]
ANALYSIS_CACHE_TTL_SECONDS: int = _api["analysis_cache_ttl_seconds"]
BULK_WORKER_THREADS: int = _api["bulk_worker_threads"]
RESULT_STORE_TTL_SECONDS: int = _api["result_store_ttl_seconds"]

# Shortest resume text worth scoring. Below this the PDF is almost certainly
# scanned images rather than a text layer.
MIN_RESUME_CHARS: int = 100

# Bounds on the pasted job description.
MIN_JD_CHARS: int = 50
MAX_JD_CHARS: int = 20_000
