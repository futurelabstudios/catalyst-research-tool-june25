from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Literal, Optional
from pydantic import BaseModel


class Activity(BaseModel):
    id: str
    phase: str
    title: str
    details: Optional[str] = None
    status: Literal['in_progress', 'completed', 'error', 'skipped']
    timestamp: datetime
    progress: Optional[Dict[str, int]] = None  # {"current": 5, "total": 10}
    estimated_duration: Optional[int] = None  # seconds
    icon: Optional[str] = None
    retryable: Optional[bool] = None
    importance: Optional[Literal['critical', 'normal', 'optional']] = None

def create_activity(id, phase, title, details, status, icon=None, progress=None):
    return {
        "id": id, "phase": phase, "title": title, "details": details,
        "status": status, "timestamp": datetime.now(timezone.utc).isoformat(),
        "icon": icon, "progress": progress
    }

@dataclass
class RetrievalProgress:
    """A data structure to hold a single update from the retrieval generator."""
    activity: Activity
    content_chunk: Optional[str] = None