from __future__ import annotations
from functools import lru_cache
from ..agents.service import LegalAgentService
from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_agent_service() -> LegalAgentService:
    settings = get_settings()
    logger.info("Khởi tạo LegalAgentService (profile=%s)", settings.app_profile)
    service = LegalAgentService(settings)
    service.bootstrap()
    return service
