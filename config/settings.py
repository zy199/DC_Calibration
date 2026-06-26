"""
全局配置模块
-----------
集中管理所有可调参数。修改阈值无需深入代码，改这里即可。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class VideoSettings:
    """视频处理相关配置"""
    frame_cache_size: int = 200
    supported_formats: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
    default_fps: float = 30.0  # 无法读取fps时的默认值


@dataclass
class JumpDetectionSettings:
    """跳变帧检测相关配置"""
    # 末位数字区域宽度占比（相对被校时钟ROI宽度）
    last_digit_width_ratio: float = 0.30
    # 帧差二值化阈值（0-255）
    diff_threshold: int = 30
    # 峰值检测窗口大小（固定帧数，避免高帧率时窗口过大吞噬多跳变）
    peak_window_frames: int = 5
    # 自适应阈值系数：threshold = median + multiplier * std
    peak_sigma_multiplier: float = 2.0
    # 粗筛阈值：帧差变化率低于此值不计算SSIM
    coarse_diff_threshold: float = 0.05
    # 高斯模糊核大小（去噪）
    gaussian_kernel: Tuple[int, int] = (5, 5)


@dataclass
class ClaritySettings:
    """清晰度评估相关配置"""
    # 三指标权重（总和=1.0）
    # 跳变帧一致性低是正常现象（数字确实变了），降低一致性权重
    laplacian_weight: float = 0.50
    transition_weight: float = 0.35
    consistency_weight: float = 0.15
    # 过渡态检测：自适应二值化参数
    adaptive_thresh_block_size: int = 11
    adaptive_thresh_constant: int = 2
    # 最低清晰度警告阈值（相对值，如果最高分/最低分 < 此值则警告）
    clarity_warn_ratio: float = 1.2
    # 两时钟Laplacian分数比值的警告阈值
    clock_lap_diff_warn_ratio: float = 3.0
    # 搜索窗口大小（跳变帧前后各取N帧）
    search_window: int = 2


@dataclass
class OCRSettings:
    """OCR识别相关配置"""
    # 主引擎：'paddleocr' | 'easyocr'
    engine: str = 'paddleocr'
    # 语言
    lang: str = 'en'
    # 识别置信度阈值（低于此值提示用户确认）
    conf_threshold: float = 0.6
    # PaddleOCR 模型目录（None=自动下载）
    paddle_model_dir: str = ''
    # 七段数码管模板匹配阈值
    seven_seg_match_threshold: float = 0.7
    # 字符白名单
    char_whitelist: str = '0123456789:-. /年 月 日'


@dataclass
class DatabaseSettings:
    """数据库相关配置"""
    db_filename: str = 'calibration_records.db'
    # 数据库文件路径（None=使用当前目录）
    db_dir: str = ''

    @property
    def db_path(self) -> str:
        if self.db_dir:
            return str(Path(self.db_dir) / self.db_filename)
        return self.db_filename


@dataclass
class UISettings:
    """界面相关配置"""
    window_title: str = '数字时钟自动校准系统'
    window_default_width: int = 1280
    window_default_height: int = 800
    language: str = 'zh_CN'
    # 视频播放器默认缩放
    default_zoom: float = 1.0
    # 帧浏览器缩略图尺寸
    thumbnail_width: int = 240
    thumbnail_height: int = 135


@dataclass
class AppSettings:
    """应用总配置"""
    video: VideoSettings = field(default_factory=VideoSettings)
    jump_detection: JumpDetectionSettings = field(default_factory=JumpDetectionSettings)
    clarity: ClaritySettings = field(default_factory=ClaritySettings)
    ocr: OCRSettings = field(default_factory=OCRSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    ui: UISettings = field(default_factory=UISettings)

    # 调试模式
    debug: bool = False

    def save(self, filepath: str):
        """保存配置到JSON文件（未来扩展）"""
        import json
        # 简化版：仅保存非默认值
        pass


# 全局单例
settings = AppSettings()
