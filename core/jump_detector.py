"""
跳变帧检测 -- MSER数字定位 + 折半查找(亮区窗口SSIM)
=================================================
统一管线(红绿LED通用, 全离线):
1. MSER定位末位数字 + 向右扩展捕获窄数字
2. 提取亮区窗口(数字笔画连通块)
3. 折半查找: 亮区窗口SSIM优先, 不足回退全图SSIM
"""
import cv2, time, numpy as np
from dataclasses import dataclass
from config.settings import settings
from utils.logger import logger


@dataclass
class JumpDetectionResult:
    jump_frame_idx: int = -1
    diff_score: float = 0.0
    timestamp: float = 0.0
    frames_scanned: int = 0
    threshold: float = 0.0
    detection_time_ms: float = 0.0
    found: bool = False

    @property
    def success(self) -> bool:
        return self.found and self.jump_frame_idx >= 0


class JumpDetector:
    def __init__(self, config=None):
        self.config = config or settings.jump_detection

    def detect(self, frame_getter, roi, frame_count, fps,
               start_frame=0, progress_callback=None, video_path=''
               ) -> JumpDetectionResult:
        t0 = time.time()
        if not video_path:
            return JumpDetectionResult()
        return self._detect(video_path, roi, fps, t0, start_frame)

    # ====== 1. MSER末位数字定位 ======
    def _find_digit_mser(self, f, roi):
        """MSER检测数字区域 + 向右扩展捕获紧邻窄数字(如'1')"""
        x, y, w, h = roi
        gray = cv2.cvtColor(f[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

        mser = cv2.MSER_create(
            delta=5, min_area=20, max_area=max(w * h // 2, 500),
            max_variation=0.25, min_diversity=0.2,
        )
        regions, _ = mser.detectRegions(gray)
        if not regions:
            return None

        boxes = []
        for pts in regions:
            rx, ry, rw, rh = cv2.boundingRect(pts)
            aspect = rh / max(rw, 1)
            if rw >= 3 and rh >= 8 and 1.2 <= aspect <= 7.0:
                if ry + rh // 2 > h * 0.35:
                    boxes.append((rx + x, ry + y, rw, rh))
        if not boxes:
            return None

        boxes.sort(key=lambda b: b[0], reverse=True)
        bx, by, bw, bh = boxes[0]

        # 向右扩展: 捕获窄数字(如"1")
        right_edge = bx + bw
        roi_right = x + w
        if right_edge < roi_right - 2:
            ext_w = min(int(bw * 1.5), roi_right - right_edge)
            right_strip = gray[:, bx - x + bw:bx - x + bw + ext_w]
            if right_strip.size > 0:
                bright = np.sum(right_strip > np.percentile(gray, 95))
                if bright > 10:
                    bw = min(roi_right - bx, bw + ext_w)

        pad = 2
        return (max(0, bx - pad), max(0, by - pad), bw + 2 * pad, bh + 2 * pad)

    # ====== 2. 亮区窗口提取 ======
    def _extract_bright_windows(self, gray_roi):
        """从灰度ROI提取亮区连通块(数字笔画)"""
        p98 = np.percentile(gray_roi, 98)
        if p98 > 30:
            _, b = cv2.threshold(gray_roi, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, b = cv2.threshold(gray_roi, max(p98 * 0.5, 10), 255,
                                 cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        wins = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw >= 3 and bh >= 5:
                wins.append((bx, by, bw, bh))
        return wins

    # ====== 3. 折半查找: 亮区窗口SSIM优先, 不足回退全图SSIM ======
    def _binary_search(self, vpath, dx, dy, dw, dh, start_frame, fc, t0):
        try:
            from skimage.metrics import structural_similarity as ssim
            cap = cv2.VideoCapture(vpath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            _, base_f = cap.read()
            cap.release()
            if base_f is None:
                return JumpDetectionResult()
            base_g = cv2.cvtColor(base_f[dy:dy+dh, dx:dx+dw], cv2.COLOR_BGR2GRAY)

            base_wins = self._extract_bright_windows(base_g)
            good = [(wx, wy, ww, wh) for (wx, wy, ww, wh) in base_wins
                    if ww >= 7 and wh >= 7]
            use_win = len(good) >= 2
            logger.info(f"亮区窗口: {len(base_wins)}个, 合格: {len(good)}个, "
                        f"策略: {'窗口SSIM' if use_win else '全图SSIM'}")

            def changed(fi):
                cap2 = cv2.VideoCapture(vpath)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, fi)
                _, fm = cap2.read()
                cap2.release()
                if fm is None:
                    return True
                cmp_g = cv2.cvtColor(fm[dy:dy+dh, dx:dx+dw], cv2.COLOR_BGR2GRAY)

                if use_win:
                    cmp_wins = self._extract_bright_windows(cmp_g)
                    all_set = {(wx, wy, ww, wh): True
                               for wx, wy, ww, wh in good + cmp_wins}
                    for (wx, wy, ww, wh) in all_set:
                        y2 = min(wy + wh, base_g.shape[0])
                        x2 = min(wx + ww, base_g.shape[1])
                        if y2 - wy < 7 or x2 - wx < 7:
                            continue
                        if ssim(base_g[wy:y2, wx:x2],
                                cmp_g[wy:y2, wx:x2],
                                data_range=255) < 0.92:
                            return True
                    return False
                else:
                    return ssim(base_g, cmp_g, data_range=255) < 0.92

            lo, hi = max(start_frame + 10, start_frame), fc - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if changed(mid):
                    hi = mid
                else:
                    lo = mid + 1
            for fi in range(max(start_frame + 5, lo - 3), min(fc - 1, lo + 4)):
                if changed(fi):
                    result = JumpDetectionResult()
                    result.found = True
                    result.jump_frame_idx = fi
                    result.detection_time_ms = (time.time() - t0) * 1000
                    logger.info(f"折半查找: Frame {fi}, {result.detection_time_ms:.0f}ms")
                    return result
        except Exception as e:
            logger.warning(f"折半查找失败: {e}")
        return JumpDetectionResult()

    # ====== 主检测 ======
    def _detect(self, vpath, roi, fps, t0, start_frame=0, lower_threshold=False):
        cap = cv2.VideoCapture(vpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(100, start_frame))
        ok, f = cap.read()
        cap.release()
        if not ok:
            return JumpDetectionResult()

        fc = int(cv2.VideoCapture(vpath).get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.VideoCapture(vpath).release()

        # 步骤1: MSER定位末位数字
        digit_roi = self._find_digit_mser(f, roi)
        if not digit_roi:
            logger.warning("MSER未定位到末位数字, 使用ROI右下象限回退")
            x, y, w, h = roi
            mid_y = y + h // 2
            strip_w = int(w * 0.30)
            digit_roi = (x + w - strip_w, mid_y, strip_w, h - h // 2)

        dx, dy, dw, dh = digit_roi
        logger.info(f"末位数字: {dw}x{dh}px @ ({dx},{dy})")

        # 步骤2: 折半查找(自动选择窗口SSIM或全图SSIM)
        return self._binary_search(vpath, dx, dy, dw, dh, start_frame, fc, t0)
