"""
跳变帧检测模块（早停 + 顺序读取）
"""
import cv2, time, numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable
from config.settings import settings, JumpDetectionSettings
from utils.logger import logger


@dataclass
class JumpDetectionResult:
    jump_frame_idx: int = -1
    diff_score: float = 0.0
    ssim_score: Optional[float] = None
    timestamp: float = 0.0
    frames_scanned: int = 0
    threshold: float = 0.0
    detection_time_ms: float = 0.0
    found: bool = False

    @property
    def success(self) -> bool:
        return self.found and self.jump_frame_idx >= 0


class JumpDetector:
    def __init__(self, config: Optional[JumpDetectionSettings] = None):
        self.config = config or settings.jump_detection

    def detect(self, frame_getter, roi, frame_count, fps,
               start_frame=0, progress_callback=None, video_path=''
               ) -> JumpDetectionResult:
        t_start = time.time()

        x, y, w, h = roi
        digit_w = int(w * self.config.last_digit_width_ratio)
        digit_roi = (x + w - digit_w, y, digit_w, h)

        if video_path:
            return self._detect_seq(video_path, digit_roi, roi, frame_count,
                                    fps, start_frame, progress_callback, t_start)
        return self._detect_rand(frame_getter, digit_roi, roi, frame_count,
                                 fps, start_frame, progress_callback, t_start)

    # ---- 顺序读取（极快） ----
    def _detect_seq(self, vpath, droi, roi, fc, fps, start, prog, t0
                    ) -> JumpDetectionResult:
        result = JumpDetectionResult()
        cap = cv2.VideoCapture(vpath)
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        # 基线
        n_base = min(int(fps * 2), 200, fc - start - 2)
        diffs, prev = [], None
        for _ in range(n_base):
            ok, fm = cap.read()
            if not ok: break
            try:
                g = self._gray(fm, droi)
                if prev is not None: diffs.append(self._diff(prev, g))
                prev = g
            except: continue

        threshold = self._threshold(diffs)
        result.threshold = threshold

        # 扫描
        window, rd, hit, fi = 5, [], -1, start + n_base
        prev = None
        while True:
            ok, fm = cap.read()
            if not ok: break
            try:
                g = self._gray(fm, droi)
                if prev is not None:
                    d = self._diff(prev, g); rd.append(d)
                    if len(rd) > window: rd.pop(0)
                    if d >= threshold and len(rd) >= 3 and d >= max(rd):
                        hit = fi; break
                prev = g; fi += 1
                if prog and fi % 300 == 0:
                    prog(min(fi / fc, 0.99))
            except: fi += 1

        cap.release()
        result.frames_scanned = fi - start
        return self._finish(result, droi, vpath, hit, fps, t0, rd, prog,
                           seq=True, fg=None)

    # ---- 随机访问（回退） ----
    def _detect_rand(self, fg, droi, roi, fc, fps, start, prog, t0
                     ) -> JumpDetectionResult:
        result = JumpDetectionResult()
        n_base = min(start + int(fps * 2), start + 200, fc - 2)
        diffs, prev = [], None
        for i in range(start, n_base):
            try:
                g = self._gray(fg(i), droi)
                if prev is not None: diffs.append(self._diff(prev, g))
                prev = g
            except: continue

        threshold = self._threshold(diffs)
        result.threshold = threshold

        window, rd, hit, prev = 5, [], -1, None
        for i in range(start, fc - 1):
            try:
                g = self._gray(fg(i), droi)
                if prev is not None:
                    d = self._diff(prev, g); rd.append(d)
                    if len(rd) > window: rd.pop(0)
                    if d >= threshold and len(rd) >= 3 and d >= max(rd):
                        hit = i; break
                prev = g
                if prog and i % 100 == 0:
                    prog(min(i / fc, 0.99))
            except: continue

        result.frames_scanned = max(0, hit if hit >= 0 else fc)
        return self._finish(result, droi, None, hit, fps, t0, rd, prog,
                           seq=False, fg=fg)

    def _finish(self, result, droi, vpath, hit, fps, t0, rd, prog, seq,
                fg=None):
        if hit < 0:
            result.detection_time_ms = (time.time() - t0) * 1000
            return result

        # SSIM确认
        ssim_v = None
        try:
            from skimage.metrics import structural_similarity as ssim
            if seq and vpath:
                c = cv2.VideoCapture(vpath)
                c.set(cv2.CAP_PROP_POS_FRAMES, max(0, hit - 1))
                _, pf = c.read(); _, cf = c.read(); c.release()
            elif fg:
                pf = fg(max(0, hit - 1))
                cf = fg(hit)
            dx, dy, dw, dh = droi
            pg = cv2.cvtColor(pf[dy:dy+dh, dx:dx+dw], cv2.COLOR_BGR2GRAY)
            cg = cv2.cvtColor(cf[dy:dy+dh, dx:dx+dw], cv2.COLOR_BGR2GRAY)
            ssim_v = ssim(pg, cg, data_range=255)
        except: pass

        result.found = True
        result.jump_frame_idx = hit
        result.diff_score = rd[-1] if rd else 0
        result.ssim_score = ssim_v
        result.timestamp = hit / fps if fps > 0 else 0
        result.detection_time_ms = (time.time() - t0) * 1000
        if prog: prog(1.0)
        logger.info(f"跳变: Frame {hit} ({result.timestamp:.3f}s), "
                     f"扫描{result.frames_scanned}帧, {result.detection_time_ms:.0f}ms")
        return result

    # ---- 工具 ----
    def _gray(self, frame, droi):
        x, y, w, h = droi
        g = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g, self.config.gaussian_kernel, 0)

    def _diff(self, a, b):
        d = cv2.absdiff(a, b)
        _, t = cv2.threshold(d, self.config.diff_threshold, 255, cv2.THRESH_BINARY)
        return float(np.sum(t > 0) / t.size)

    def _threshold(self, diffs):
        if not diffs: return 0.0005
        arr = np.array(diffs)
        m = np.median(arr)
        p90 = np.percentile(arr, 90)
        th = m + 2.0 * (p90 - m)
        return max(th, 0.0005)
