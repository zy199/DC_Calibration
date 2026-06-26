"""
视频IO模块
---------
封装OpenCV视频读取，提供帧缓存和随机访问。
"""

import cv2
import numpy as np
from collections import OrderedDict
from typing import Tuple, Optional, Dict

from utils.logger import logger


class FrameReadError(Exception):
    """帧读取异常"""
    pass


class VideoIO:
    """
    视频文件读取器。

    支持：
    - 随机帧访问（cv2.CAP_PROP_POS_FRAMES）
    - LRU帧缓存（避免重复解码）
    - ROI区域提取
    - 帧时间戳计算
    """

    def __init__(self, video_path: str, cache_size: int = 200):
        """
        Args:
            video_path: 视频文件路径
            cache_size: 帧缓存最大数量
        """
        self.video_path = video_path
        self.cache_size = cache_size

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FrameReadError(f"无法打开视频文件: {video_path}")

        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._duration = self._frame_count / self._fps if self._fps > 0 else 0.0

        # 高帧率自适应缓存：至少存0.5秒的帧，上限2000
        if self._fps > 100:
            adaptive_cache = min(int(self._fps * 0.5), 2000)
            self.cache_size = max(cache_size, adaptive_cache)
            logger.debug(f"高帧率({self._fps:.0f}fps)，缓存扩展至{self.cache_size}帧")

        # LRU缓存：OrderedDict, 访问/插入时移到末尾, 超限时pop首部
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

        logger.info(f"视频已打开: {self._width}x{self._height}, "
                     f"{self._fps:.2f}fps, {self._frame_count}帧, "
                     f"{self._duration:.1f}秒")

    # ---- 属性 ----

    @property
    def frame_count(self) -> int:
        """总帧数"""
        return self._frame_count

    @property
    def fps(self) -> float:
        """帧率"""
        return self._fps

    @property
    def width(self) -> int:
        """视频宽度"""
        return self._width

    @property
    def height(self) -> int:
        """视频高度"""
        return self._height

    @property
    def resolution(self) -> Tuple[int, int]:
        """分辨率 (宽, 高)"""
        return (self._width, self._height)

    @property
    def duration(self) -> float:
        """视频时长（秒）"""
        return self._duration

    # ---- 帧访问 ----

    def get_frame(self, index: int) -> np.ndarray:
        """
        获取指定索引的帧（BGR格式）。

        自动使用缓存，命中缓存时O(1)，未命中时seek+read。

        Args:
            index: 帧索引 (0-based)

        Returns:
            BGR格式的numpy数组

        Raises:
            FrameReadError: 帧索引越界或读取失败
        """
        if index < 0 or index >= self._frame_count:
            raise FrameReadError(
                f"帧索引越界: {index} (有效范围 0~{self._frame_count - 1})")

        # 缓存命中 → 移到末尾（最近使用）
        if index in self._cache:
            frame = self._cache.pop(index)
            self._cache[index] = frame
            return frame.copy()  # 返回副本，防止外部修改污染缓存

        # 缓存未命中 → 读取
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            raise FrameReadError(f"无法读取第 {index} 帧")

        # 存入缓存
        self._cache[index] = frame
        self._evict_cache()

        return frame.copy()

    def get_frame_timestamp(self, index: int) -> float:
        """
        帧索引转视频时间戳（秒）。

        Args:
            index: 帧索引

        Returns:
            该帧在视频中的时间（秒）
        """
        return index / self._fps if self._fps > 0 else 0.0

    def extract_roi(self, frame_index: int,
                    roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        提取指定帧的ROI区域。

        Args:
            frame_index: 帧索引
            roi: (x, y, w, h) 区域坐标

        Returns:
            ROI区域的BGR图像
        """
        frame = self.get_frame(frame_index)
        x, y, w, h = roi

        # 边界裁剪
        x = max(0, x)
        y = max(0, y)
        w = min(w, self._width - x)
        h = min(h, self._height - y)

        if w <= 0 or h <= 0:
            raise FrameReadError(f"ROI区域无效: {roi}")

        return frame[y:y + h, x:x + w]

    def get_multiple_frames(self, indices: list) -> Dict[int, np.ndarray]:
        """
        批量获取多帧（比单独调用get_frame更高效）。

        Args:
            indices: 帧索引列表

        Returns:
            {帧索引: BGR图像} 字典
        """
        result = {}
        for idx in indices:
            try:
                result[idx] = self.get_frame(idx)
            except FrameReadError as e:
                logger.warning(f"批量读取跳过第{idx}帧: {e}")
        return result

    # ---- 缓存管理 ----

    def _evict_cache(self):
        """LRU淘汰：缓存超过上限时移除最旧的条目"""
        while len(self._cache) > self.cache_size:
            oldest_key, _ = self._cache.popitem(last=False)

    def clear_cache(self):
        """清空帧缓存"""
        self._cache.clear()

    # ---- 资源管理 ----

    def close(self):
        """释放视频资源"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.clear_cache()
        logger.info("视频资源已释放")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return (f"VideoIO('{self.video_path}', "
                f"{self._width}x{self._height}, "
                f"{self._fps:.1f}fps, {self._frame_count}frames)")
