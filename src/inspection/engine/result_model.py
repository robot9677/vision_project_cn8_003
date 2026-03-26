from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ROIResult:
    roi_id: Any
    ok: bool
    reason: str
    metrics: Dict[str, Any]