"""
图像工具模块
-----------
图像预处理、增强等通用函数。
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    OCR预处理管线：灰度化 → CLAHE → 双边滤波 → 自适应二值化。

    Args:
        image: BGR或灰度输入图像

    Returns:
        二值化后的图像，适合直接送入OCR
    """
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # CLAHE 自适应直方图均衡化（改善不均匀光照）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 双边滤波（去噪保边）
    denoised = cv2.bilateralFilter(enhanced, 5, 75, 75)

    # 自适应二值化
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    return binary


def extract_digit_region(roi: np.ndarray,
                         last_digit_ratio: float = 0.30) -> np.ndarray:
    """
    从时钟ROI中提取末位数字区域。

    Args:
        roi: 时钟显示区域图像
        last_digit_ratio: 末位数字区域占ROI宽度的比例

    Returns:
        末位数字区域图像
    """
    h, w = roi.shape[:2]
    digit_w = int(w * last_digit_ratio)
    return roi[:, w - digit_w:]


def detect_display_type(roi: np.ndarray) -> str:
    """
    检测时钟显示类型。

    通过边缘密度和轮廓形状分析判断是七段数码管还是LCD。

    Args:
        roi: 时钟显示区域图像（BGR或灰度）

    Returns:
        'seven_segment' | 'lcd' | 'unknown'
    """
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # Canny边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 霍夫线检测：七段数码管有大量细长直线段
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=30, minLineLength=10, maxLineGap=3)
    line_count = len(lines) if lines is not None else 0

    # 二值化后轮廓分析
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # 统计细长轮廓（七段特征）
    thin_contour_count = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 0 and h > 0:
            aspect_ratio = max(w / h, h / w)
            if aspect_ratio > 3:  # 细长形状（段）
                thin_contour_count += 1

    # 判断逻辑
    total_pixels = gray.shape[0] * gray.shape[1]
    line_density = line_count / (total_pixels / 10000)  # 归一化

    if line_density > 0.5 and thin_contour_count > 5:
        return 'seven_segment'
    elif line_count > 5:
        return 'lcd'
    else:
        return 'unknown'


def detect_clock_regions(frame: np.ndarray,
                         max_candidates: int = 2
                         ) -> list[Tuple[int, int, int, int]]:
    """
    自动检测画面中的数字时钟显示区域。

    策略：
      1. 找到所有可能的数字轮廓
      2. 按行分组（相同y≈同一行）
      3. 每行合并为一个完整的时钟ROI（至少覆盖HH:MM宽度）
      4. 相邻行可能是同一时钟的日期+时间，合并

    Args:
        frame: BGR视频帧
        max_candidates: 返回最多几个候选

    Returns:
        [(x, y, w, h), ...] 按y坐标排序的时钟ROI列表
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fh, fw = gray.shape[:2]

    # 亮度拉伸
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 > p2 + 10:
        gray = np.clip((gray.astype(float) - p2) * 255 / (p98 - p2), 0, 255).astype(np.uint8)

    # 极暗视频用更低的阈值
    mean_val = np.mean(gray)
    if mean_val < 30:
        # 暗视频：用固定低阈值（亮度>均值+5就算亮）
        thresh_val = min(mean_val + 10, 40)
        _, bright = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(bright > 0) / bright.size > 0.7:
        bright = cv2.bitwise_not(bright)

    # 找所有数字轮廓
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 收集所有可能的数字块
    digit_blocks = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 5 or ch < 10:  # 太小，跳过
            continue
        aspect = ch / cw if cw > 0 else 0
        if aspect < 0.3 or aspect > 5:  # 形状不像数字
            continue
        digit_blocks.append((x, y, cw, ch))

    if len(digit_blocks) < 3:
        return []

    # 按y坐标分组（同一行）
    digit_blocks.sort(key=lambda b: b[1])  # 按y排序
    rows = []
    current_row = [digit_blocks[0]]
    row_y_center = digit_blocks[0][1] + digit_blocks[0][3] / 2

    for block in digit_blocks[1:]:
        x, y, cw, ch = block
        block_center_y = y + ch / 2
        if abs(block_center_y - row_y_center) < 30:  # 同一行（30px容差）
            current_row.append(block)
        else:
            rows.append(current_row)
            current_row = [block]
        row_y_center = block_center_y
    rows.append(current_row)

    # 每行合并为一个ROI（加边距）
    clock_rows = []
    for row in rows:
        if len(row) < 2:  # 至少2个数字块才像时钟
            continue
        min_x = min(b[0] for b in row)
        min_y = min(b[1] for b in row)
        max_x = max(b[0] + b[2] for b in row)
        max_y = max(b[1] + b[3] for b in row)

        # 加边距
        pad_x = 20
        pad_y = 15
        rx = max(0, min_x - pad_x)
        ry = max(0, min_y - pad_y)
        rw = min(fw - rx, max_x - min_x + 2 * pad_x)
        rh = min(fh - ry, max_y - min_y + 2 * pad_y)

        if rw > 60 and rh > 20:  # 至少能容纳HH:MM
            clock_rows.append((rx, ry, rw, rh))

    # 合并相邻行
    if len(clock_rows) >= 2:
        clock_rows.sort(key=lambda r: r[1])
        merged = []
        i = 0
        while i < len(clock_rows):
            if i + 1 < len(clock_rows):
                r1, r2 = clock_rows[i], clock_rows[i + 1]
                r1_bottom = r1[1] + r1[3]
                gap = r2[1] - r1_bottom
                overlap = r1_bottom - r2[1]
                if gap < 80 or overlap > -20:  # 合并或重叠
                    nx = min(r1[0], r2[0])
                    ny = r1[1]
                    nw = max(r1[0]+r1[2], r2[0]+r2[2]) - nx
                    nh = r2[1]+r2[3] - r1[1]
                    merged.append((nx, ny, nw, nh))
                    i += 2
                    continue
            merged.append(clock_rows[i])
            i += 1
        clock_rows = merged

    # 返回前max_candidates个
    clock_rows.sort(key=lambda r: r[1])  # 按y从上到下
    return clock_rows[:max_candidates]


def _rect_iou(a: Tuple[int, int, int, int],
              b: Tuple[int, int, int, int]) -> float:
    """计算两个矩形的IoU"""
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def resize_keep_aspect(image: np.ndarray,
                       target_size: Tuple[int, int]) -> np.ndarray:
    """
    保持宽高比的缩放。

    Args:
        image: 输入图像
        target_size: (宽, 高)

    Returns:
        缩放后的图像
    """
    h, w = image.shape[:2]
    tw, th = target_size

    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized
