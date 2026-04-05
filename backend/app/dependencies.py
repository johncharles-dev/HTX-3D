"""FastAPI dependency injection."""

from .services.task_manager import TaskManager
from .services.sam3_segmentation import SAM3Service

# Singleton instances
_task_manager: TaskManager | None = None
_sam3_service: SAM3Service | None = None


def get_task_manager() -> TaskManager:
    if _task_manager is None:
        raise RuntimeError("TaskManager not initialized")
    return _task_manager


def set_task_manager(tm: TaskManager):
    global _task_manager
    _task_manager = tm


def get_sam3_service() -> SAM3Service:
    if _sam3_service is None:
        raise RuntimeError("SAM3Service not initialized")
    return _sam3_service


def set_sam3_service(svc: SAM3Service):
    global _sam3_service
    _sam3_service = svc
