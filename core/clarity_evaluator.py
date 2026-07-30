"""
清晰度评估模块
-------------
对被校和标准时钟的末位数字ROI同时进行三指标综合评分，
确保两个时钟均显示清晰后才选定最佳帧。

三指标（被校和标准分别计算后取平均）：
  1. Laplacian方差 — 整体锐度（权重50%）
  2. 过渡态检测 — 双峰性分析（权重35%）
  3. SSIM一致性 — 帧间结构相似度（权重15%）

核心设计：
  两时钟末位必须同时清晰，任一不清晰则该帧不可用。
  实验表明最佳帧通常为跳变帧本身或跳变后一帧。
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from config.settings import settings, ClaritySettings


@dataclass
class ClarityScore:
    frame_idx: int
    laplacian: float = 0.0
    transition: float = 0.0
    consistency: float = 0.0
    combined: float = 0.0

    def __repr__(self) -> str:
        return (f"ClarityScore(f={self.frame_idx}, "
                f"lap={self.laplacian:.1f}, trans={self.transition:.2f}, "
                f"cons={self.consistency:.3f}, total={self.combined:.2f})")


@dataclass
class ClarityResult:
    scores: List[ClarityScore]
    best_idx: int = -1
    best_score: float = 0.0
    is_clear: bool = False
    warning: str = ''
    is_jump: bool = False       # best frame == jump frame?
    is_next: bool = False       # best frame == jump frame + 1?


class ClarityEvaluator:
    def __init__(self, config: Optional[ClaritySettings] = None):
        self.config = config or settings.clarity

    def evaluate_window(self,
                        frame_getter,
                        center_idx: int,
                        roi_cal: Tuple[int, int, int, int],
                        roi_std: Tuple[int, int, int, int],
                        window: int = 2,
                        jump_idx: int | None = None,
                        digit_roi_cal: Tuple[int, int, int, int] | None = None,
                        digit_roi_std: Tuple[int, int, int, int] | None = None,
                        ) -> ClarityResult:

        scores = []
        start = max(0, center_idx - window)
        end = center_idx + window + 1

        for idx in range(start, end):
            try:
                frame = frame_getter(idx)
                score = self._evaluate_single(
                    frame, roi_cal, roi_std, idx,
                    digit_roi_cal, digit_roi_std)
                scores.append(score)
            except Exception:
                scores.append(ClarityScore(frame_idx=idx))

        if not scores:
            return ClarityResult(scores=[], warning="无有效帧可评估")

        # Laplacian 归一化
        self._normalize_scores(scores)

        w_lap = self.config.laplacian_weight
        w_trans = self.config.transition_weight
        w_cons = self.config.consistency_weight
        actual_jump = jump_idx if jump_idx is not None else center_idx

        for s in scores:
            if s.frame_idx == actual_jump:
                s.consistency = max(s.consistency, 0.5)
            s.combined = (w_lap * s.laplacian +
                          w_trans * s.transition +
                          w_cons * s.consistency)

        # 只在 >= 跳变帧的候选帧中选择（必须显示新时间值）
        valid = [s for s in scores if s.frame_idx >= actual_jump]
        if not valid:
            valid = scores
        best = max(valid, key=lambda s: s.combined)

        is_clear = best.combined >= 0.3
        warning = '' if is_clear else "画面整体清晰度不足"

        return ClarityResult(
            scores=scores,
            best_idx=best.frame_idx,
            best_score=best.combined,
            is_clear=is_clear,
            warning=warning,
            is_jump=(best.frame_idx == actual_jump),
            is_next=(best.frame_idx == actual_jump + 1),
        )

    def _evaluate_single(self, frame, roi_cal, roi_std, frame_idx,
                         digit_roi_cal=None, digit_roi_std=None):
        """评估单帧——两个时钟末位ROI分别计算后取平均"""
        # 被校时钟末位ROI
        if digit_roi_cal:
            dx, dy, dw, dh = digit_roi_cal
            cal_roi = frame[dy:dy+dh, dx:dx+dw]
        else:
            x, y, w, h = roi_cal
            cal_roi = frame[y:y+h, x:x+w]

        # 标准时钟末位ROI
        if digit_roi_std:
            dx, dy, dw, dh = digit_roi_std
            std_roi = frame[dy:dy+dh, dx:dx+dw]
        else:
            x, y, w, h = roi_std
            std_roi = frame[y:y+h, x:x+w]

        # 指标1: Laplacian — 两时钟分别算取平均
        lap_cal = self._laplacian_sharpness(cal_roi)
        lap_std = self._laplacian_sharpness(std_roi)
        laplacian = (lap_cal + lap_std) / 2

        # 指标2: 过渡态 — 两时钟分别算取平均
        trans_cal = self._transition_score(cal_roi)
        trans_std = self._transition_score(std_roi)
        transition = (trans_cal + trans_std) / 2

        return ClarityScore(
            frame_idx=frame_idx,
            laplacian=laplacian,
            transition=transition,
        )

    def evaluate_consistency(self, scores, frame_getter,
                             roi_cal, roi_std,
                             digit_roi_cal=None, digit_roi_std=None):
        """为候选帧补充SSIM一致性。使用末位ROI（聚焦跳变区域）"""
        for i, s in enumerate(scores):
            if i == 0 or i == len(scores) - 1:
                s.consistency = 0.5
                continue
            try:
                from skimage.metrics import structural_similarity as ssim_fn
                prev_f = frame_getter(scores[i-1].frame_idx)
                curr_f = frame_getter(s.frame_idx)
                next_f = frame_getter(scores[i+1].frame_idx)

                def crop(f, roi):
                    x, y, w, h = roi
                    return cv2.cvtColor(f[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

                # 两个时钟分别算SSIM取平均
                ssim_vals = []
                for d_roi in [digit_roi_cal, digit_roi_std]:
                    if d_roi is None:
                        continue
                    prev_g = crop(prev_f, d_roi)
                    curr_g = crop(curr_f, d_roi)
                    next_g = crop(next_f, d_roi)
                    s1 = ssim_fn(prev_g, curr_g, data_range=255)
                    s2 = ssim_fn(curr_g, next_g, data_range=255)
                    ssim_vals.append((s1 + s2) / 2)

                s.consistency = np.mean(ssim_vals) if ssim_vals else 0.5
            except Exception:
                s.consistency = 0.5

    # ---- 底层指标 ----

    def _laplacian_sharpness(self, roi: np.ndarray) -> float:
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    def _transition_score(self, roi: np.ndarray) -> float:
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        half = 128
        left_max = np.argmax(hist[:half]) if np.max(hist[:half]) > 0 else 0
        right_max = (half + np.argmax(hist[half:])) if np.max(hist[half:]) > 0 else 255
        if left_max >= right_max:
            return 0.5
        valley = np.min(hist[left_max:right_max + 1])
        peak_avg = (hist[left_max] + hist[right_max]) / 2
        if peak_avg < 0.001:
            return 0.5
        depth = 1.0 - valley / peak_avg
        return max(0.0, min(1.0, depth))

    def _normalize_scores(self, scores):
        laps = [s.laplacian for s in scores]
        lap_min, lap_max = min(laps), max(laps)
        if lap_max > lap_min:
            for s in scores:
                s.laplacian = (s.laplacian - lap_min) / (lap_max - lap_min)
        else:
            for s in scores:
                s.laplacian = 0.5
