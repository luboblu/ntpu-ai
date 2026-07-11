"""API 請求／回應的 Pydantic 資料模型。"""
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=100_000)
    file_gcs_path:  Optional[str] = None
    file_mime_type: Optional[str] = None
    file_name:      Optional[str] = None
    search_enabled: bool = False
    ntpu_search_enabled: bool = False
    # 使用者自選地端模型（選填）；後端只接受 local-* alias，避免一般使用者
    # 透過此欄位強制指定昂貴的雲端大模型，繞過難度分級與成本控管。
    local_model: Optional[str] = None


class RoutingConfig(BaseModel):
    threshold_tiny: Optional[float] = None   # None = 不啟用開源小模型層
    threshold_medium: float = 4.0
    threshold_large: float = 7.0
    force_model: Optional[str] = None        # None/"small"/"medium"/"large"/"tiny"/單一模型 alias
    prefer_local: bool = False               # True = 有地端（local-*）候選時優先走地端


class UserProfileRequest(BaseModel):
    system_prompt: Optional[str] = Field(default=None, max_length=10_000)
