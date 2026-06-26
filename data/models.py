"""
数据模型模块
-----------
定义校准记录等核心数据结构。
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class CalibrationRecord:
    """
    一次校准的完整记录。

    包含设备信息、校准数据、处理元数据。
    """
    # === 设备信息（用户可选输入） ===
    device_name: str = ''
    device_model: str = ''
    device_serial: str = ''
    manufacturer: str = ''
    send_unit: str = ''

    # === 校准核心数据 ===
    calibrated_time: str = ''       # 被校时钟时间 (ISO 8601)
    standard_time: str = ''         # 标准时钟时间 (ISO 8601)
    time_deviation: float = 0.0     # 时间偏差（秒）

    # === 处理元数据 ===
    video_path: str = ''            # 原始视频路径
    video_filename: str = ''        # 视频文件名
    jump_frame_idx: int = 0         # 跳变帧索引
    clarity_frame_idx: int = 0      # 最终使用的清晰帧索引
    roi_calibrated: str = ''        # 被校时钟ROI JSON
    roi_standard: str = ''          # 标准时钟ROI JSON

    # === OCR元数据 ===
    ocr_confidence: float = 0.0     # OCR平均置信度
    ocr_engine_used: str = ''       # 使用的OCR引擎

    # === 记录元数据 ===
    calibration_date: str = ''      # 校准日期
    calibrator: str = ''            # 校准人员
    notes: str = ''                 # 备注

    # === 数据库字段 ===
    id: Optional[int] = None        # 数据库主键（新建时为None）
    created_at: str = ''
    updated_at: str = ''

    def to_dict(self) -> dict:
        """转为字典，用于数据库操作"""
        d = asdict(self)
        # 不导出id（自增）
        if 'id' in d and d['id'] is None:
            del d['id']
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationRecord':
        """从字典创建实例"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CalibrationSession:
    """
    一次校准会话的状态容器。

    贯穿整个GUI流程的5个步骤，各步骤的结果存储在此。
    """
    # === Step 0: 视频信息 ===
    video_path: str = ''
    video_filename: str = ''
    video_fps: float = 0.0
    video_frame_count: int = 0
    video_duration: float = 0.0

    # === Step 1: ROI ===
    roi_calibrated: Optional[tuple] = None   # (x, y, w, h)
    roi_standard: Optional[tuple] = None     # (x, y, w, h)

    # === Step 2: 跳变检测 ===
    jump_frame_idx: int = -1
    jump_candidates: list = field(default_factory=list)

    # === Step 3: 清晰帧 ===
    clarity_frame_idx: int = -1
    clarity_browser_frames: list = field(default_factory=list)  # [(idx, score, image), ...]

    # === Step 4: OCR结果 ===
    ocr_calibrated_raw: str = ''     # OCR原始输出
    ocr_standard_raw: str = ''
    ocr_calibrated_time: Optional[datetime] = None
    ocr_standard_time: Optional[datetime] = None
    ocr_confidence: float = 0.0

    # === Step 5: 偏差 ===
    time_deviation: float = 0.0

    def is_step_complete(self, step: int) -> bool:
        """检查指定步骤是否已完成"""
        checks = {
            0: bool(self.video_path),
            1: self.roi_calibrated is not None and self.roi_standard is not None,
            2: self.jump_frame_idx >= 0,
            3: self.clarity_frame_idx >= 0,
            4: self.ocr_calibrated_time is not None and self.ocr_standard_time is not None,
        }
        return checks.get(step, False)


@dataclass
class ClarityScore:
    """一帧的清晰度评分"""
    frame_idx: int
    laplacian_score: float = 0.0      # Laplacian方差分数
    transition_score: float = 0.0     # 过渡态分数
    consistency_score: float = 0.0    # SSIM一致性分数
    combined_score: float = 0.0       # 综合分数


@dataclass
class JumpCandidate:
    """跳变候选帧"""
    frame_idx: int
    diff_score: float = 0.0           # 帧差分数
    ssim_score: float = 0.0           # SSIM分数
    timestamp: float = 0.0            # 视频时间戳（秒）
