"""
清晰度评估模块
-------------
对帧进行三指标综合评分，判断时钟数字是否清晰可读。

三指标：
  1. Laplacian方差 — 整体锐度（权重50%）
  2. 过渡态检测 — 数字是否处于切换中间态（权重35%）
  3. SSIM一致性 — 与前后帧的结构一致性（权重15%）

关键设计：
  跳变帧的一致性天然偏低（数字确实变了），不应因此被惩罚。
  跳变帧及跳变后帧展示的是新时间值，应优先选择。
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

from config.settings import settings, ClaritySettings


@dataclass
class ClarityScore:
    """单帧清晰度评分"""
    frame_idx: int
    laplacian: float = 0.0       # Laplacian方差（越高越清晰）
    transition: float = 0.0      # 过渡态分数（越高=越稳定）
    consistency: float = 0.0     # SSIM一致性（越高=越接近前后帧）
    combined: float = 0.0        # 综合分数

    def __repr__(self) -> str:
        return (f"ClarityScore(f={self.frame_idx}, "
                f"lap={self.laplacian:.1f}, trans={self.transition:.2f}, "
                f"cons={self.consistency:.3f}, total={self.combined:.2f})")


@dataclass
class ClarityResult:
    """一组帧的清晰度评估结果"""
    scores: List[ClarityScore]        # 所有评估帧的分数
    best_idx: int = -1                # 最佳帧索引
    best_score: float = 0.0           # 最佳帧综合分数
    is_clear: bool = False            # 整体是否足够清晰
    warning: str = ''                 # 警告信息


class ClarityEvaluator:
    """
    清晰度评估器。

    对跳变帧前后各N帧进行评分，选出最清晰的一帧。
    """

    def __init__(self, config: Optional[ClaritySettings] = None):
        self.config = config or settings.clarity

    def evaluate_window(self,
                        frame_getter,
                        center_idx: int,
                        roi_cal: Tuple[int, int, int, int],
                        roi_std: Tuple[int, int, int, int],
                        window: int = 2,
                        jump_idx: int | None = None,
                        ) -> ClarityResult:
        """
        评估跳变帧前后 window 帧（共 2*window+1 帧）的清晰度。

        Args:
            frame_getter: 帧获取函数(idx) → BGR图像
            center_idx: 跳变帧索引（窗口中心）
            roi_cal: 被校时钟ROI
            roi_std: 标准时钟ROI
            window: 前后各取N帧
            jump_idx: 实际跳变帧索引（用于标记，一致性评估时特殊处理）

        Returns:
            ClarityResult
        """
        scores = []
        start = max(0, center_idx - window)
        end = center_idx + window + 1

        for idx in range(start, end):
            try:
                frame = frame_getter(idx)
                score = self._evaluate_single(frame, roi_cal, roi_std, idx)
                scores.append(score)
            except Exception as e:
                # 帧读取失败 → 给最低分
                scores.append(ClarityScore(frame_idx=idx))

        if not scores:
            return ClarityResult(scores=[], warning="无有效帧可评估")

        # 综合评分归一化后加权
        self._normalize_scores(scores)

        # 计算综合分
        w_lap = self.config.laplacian_weight
        w_trans = self.config.transition_weight
        w_cons = self.config.consistency_weight

        actual_jump = jump_idx if jump_idx is not None else center_idx

        for s in scores:
            # 跳变帧的一致性天然偏低（数字确实变了），给跳变帧一致性满分
            if s.frame_idx == actual_jump:
                s.consistency = 1.0

            s.combined = (w_lap * s.laplacian +
                          w_trans * s.transition +
                          w_cons * s.consistency)

            # 跳变帧本身大幅加分（进位瞬间，需优先选择）
            if s.frame_idx == actual_jump:
                s.combined += 0.25
            elif s.frame_idx > actual_jump:
                s.combined += 0.03

        # 跳变帧强制规则：只要跳变帧分数在最佳帧70%以内，选跳变帧
        best = max(scores, key=lambda s: s.combined)
        jump_score = next((s for s in scores if s.frame_idx == actual_jump), None)
        if jump_score and best.frame_idx != actual_jump:
            if jump_score.combined >= best.combined * 0.70:
                best = jump_score

        # 整体清晰度检查
        is_clear = True
        warning = ''

        # 检查：最高分是否足够高（阈值>0.3）
        if best.combined < 0.3:
            is_clear = False
            warning = "画面整体清晰度不足，建议检查视频质量"

        # 检查：两个时钟的Laplacian分数是否差异过大
        # （如果有一台特别模糊需要警告）

        return ClarityResult(
            scores=scores,
            best_idx=best.frame_idx,
            best_score=best.combined,
            is_clear=is_clear,
            warning=warning,
        )

    def _evaluate_single(self,
                         frame: np.ndarray,
                         roi_cal: Tuple[int, int, int, int],
                         roi_std: Tuple[int, int, int, int],
                         frame_idx: int) -> ClarityScore:
        """评估单帧清晰度"""
        x1, y1, w1, h1 = roi_cal
        x2, y2, w2, h2 = roi_std

        cal_roi = frame[y1:y1 + h1, x1:x1 + w1]
        std_roi = frame[y2:y2 + h2, x2:x2 + w2]

        # 指标1: Laplacian方差（对被校和标准分别计算，取平均）
        lap_cal = self._laplacian_sharpness(cal_roi)
        lap_std = self._laplacian_sharpness(std_roi)
        laplacian = (lap_cal + lap_std) / 2

        # 指标2: 过渡态检测（主要检查被校时钟）
        transition = self._transition_score(cal_roi)

        # 指标3: SSIM一致性暂不在此处计算（需要前后帧）
        # 由调用方在normalize时处理

        return ClarityScore(
            frame_idx=frame_idx,
            laplacian=laplacian,
            transition=transition,
        )

    def evaluate_consistency(self,
                             scores: List[ClarityScore],
                             frame_getter,
                             roi: Tuple[int, int, int, int]):
        """
        为已经评估的帧补充SSIM一致性分数。

        比较每帧与其前后帧的结构相似度。
        """
        for i, s in enumerate(scores):
            if i == 0 or i == len(scores) - 1:
                s.consistency = 0.5  # 边界帧给中等分
                continue

            try:
                from skimage.metrics import structural_similarity as ssim_fn

                prev = frame_getter(scores[i - 1].frame_idx)
                curr = frame_getter(s.frame_idx)
                next_f = frame_getter(scores[i + 1].frame_idx)

                x, y, w, h = roi
                prev_g = cv2.cvtColor(prev[y:y + h, x:x + w],
                                      cv2.COLOR_BGR2GRAY)
                curr_g = cv2.cvtColor(curr[y:y + h, x:x + w],
                                      cv2.COLOR_BGR2GRAY)
                next_g = cv2.cvtColor(next_f[y:y + h, x:x + w],
                                      cv2.COLOR_BGR2GRAY)

                ssim_prev = ssim_fn(prev_g, curr_g, data_range=255)
                ssim_next = ssim_fn(curr_g, next_g, data_range=255)
                s.consistency = (ssim_prev + ssim_next) / 2

            except Exception:
                s.consistency = 0.5

    # ---- 底层指标计算 ----

    def _laplacian_sharpness(self, roi: np.ndarray) -> float:
        """Laplacian方差 — 衡量图像锐度"""
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    def _transition_score(self, roi: np.ndarray) -> float:
        """
        过渡态检测 — 检测七段数码管是否处于切换中间态。

        原理：正常状态下，每个段要么亮要么灭（双峰分布）；
        切换中间态时像素强度介于两者之间（单峰）。
        双峰性越强 → 分数越高 → 越清晰。
        """
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # 计算像素强度直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # 简化的双峰性检测：计算直方图的"谷深度"
        # 找到两个峰值
        half = 128
        left_max = np.argmax(hist[:half]) if np.max(hist[:half]) > 0 else 0
        right_max = (half + np.argmax(hist[half:])) if np.max(hist[half:]) > 0 else 255

        if left_max >= right_max:
            return 0.5  # 无法区分

        # 谷深度：峰值之间的最小频率
        valley = np.min(hist[left_max:right_max + 1])
        peak_avg = (hist[left_max] + hist[right_max]) / 2

        if peak_avg < 0.001:
            return 0.5

        # 谷越深 = 双峰越明显 = 分数越高 (0~1)
        depth = 1.0 - valley / peak_avg
        return max(0.0, min(1.0, depth))

    def _normalize_scores(self, scores: List[ClarityScore]):
        """将Laplacian分数归一化到0~1（相对比较）"""
        if not scores:
            return

        laps = [s.laplacian for s in scores]
        lap_min = min(laps)
        lap_max = max(laps)

        if lap_max > lap_min:
            for s in scores:
                s.laplacian = (s.laplacian - lap_min) / (lap_max - lap_min)
        else:
            for s in scores:
                s.laplacian = 0.5
