"""FastAPI dependency injection."""

from .services.task_manager import TaskManager

# Singleton instance
_task_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    if _task_manager is None:
        raise RuntimeError("TaskManager not initialized")
    return _task_manager


def set_task_manager(tm: TaskManager):
    global _task_manager
    _task_manager = tm
