"""
图像工具模块
-----------
图像预处理、增强等通用函数。
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def detect_clock_regions(frame: np.ndarray,
                         max_candidates: int = 2
                         ) -> list[Tuple[int, int, int, int]]:
    """
    自动检测画面中的数字时钟显示区域。

    策略（多级回退）：
      1. 【时间方差法】读多帧，数字变化处方差大（最鲁棒，不受亮度影响）
      2. 多级亮度阈值（OTSU/P95/低阈值）
      3. 找数字轮廓 → 按行分组 → 合并相邻行

    Args:
        frame: BGR视频帧（用作单帧分析的基准帧）
        max_candidates: 返回最多几个候选
        video_cap: cv2.VideoCapture对象（可选，用于时间方差分析）

    Returns:
        [(x, y, w, h), ...] 按y坐标排序的时钟ROI列表
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fh, fw = gray.shape[:2]

    # ---- 策略0：颜色掩码法（红/绿数码管检测，最准确） ----
    B = frame[:,:,0].astype(float)
    G = frame[:,:,1].astype(float)
    R = frame[:,:,2].astype(float)

    # 红色掩码（被校时钟通常是红色LED，提高阈值避免暗视频噪点）
    red_mask = ((R > G * 1.8) & (R > B * 1.8) & (R > 35)).astype(np.uint8) * 255
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small)
    # 水平膨胀：将同一行的散点聚合成连通块（数字是水平排列的）
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_h)
    red_regions = _extract_regions_from_binary(red_mask, fh, fw)

    # 绿色掩码（标准时钟通常是绿色LED）
    green_mask = ((G > R * 1.8) & (G > B * 1.8) & (G > 35)).astype(np.uint8) * 255
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel_small)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel_h)
    green_regions = _extract_regions_from_binary(green_mask, fh, fw)

    # 分别合并红色和绿色区域（不能混在一起！）
    red_merged = _merge_nearby_regions(red_regions, fh, fw)
    green_merged = _merge_nearby_regions(green_regions, fh, fw)



    if len(red_merged) >= 1 and len(green_merged) >= 1:
        # 红色（上面=被校）+ 绿色（下面=标准）
        all_regions = red_merged + green_merged
        all_regions.sort(key=lambda r: r[1])
        if len(all_regions) >= 2:
            return all_regions[:max_candidates]

    if len(red_merged) >= 2:
        red_merged.sort(key=lambda r: r[1])
        return red_merged[:max_candidates]
    if len(green_merged) >= 2:
        green_merged.sort(key=lambda r: r[1])
        return green_merged[:max_candidates]
    fh, fw = gray.shape[:2]

    # 亮度拉伸到0-255
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 > p2 + 10:
        gray = np.clip((gray.astype(float) - p2) * 255 / (p98 - p2), 0, 255).astype(np.uint8)
    else:
        # 极暗且对比度极低：不拉伸，直接处理原始灰度
        pass

    mean_val = np.mean(gray)
    p95_val = np.percentile(gray, 95)

    # ---- 多级阈值策略 ----
    bright = None
    strategies_tried = []

    # 策略1：OTSU（通常对亮度正常的图像效果最好）
    if mean_val >= 30:
        _, candidate = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        strategies_tried.append(('otsu', candidate))

    # 策略2：P95百分比阈值（对暗视频更鲁棒）
    pct_thresh = max(p95_val * 0.6, 15)
    _, candidate = cv2.threshold(gray, pct_thresh, 255, cv2.THRESH_BINARY)
    strategies_tried.append((f'p95x0.6={pct_thresh:.0f}', candidate))

    # 策略3：更低阈值兜底
    low_thresh = max(mean_val + 8, 10)
    _, candidate = cv2.threshold(gray, low_thresh, 255, cv2.THRESH_BINARY)
    strategies_tried.append((f'mean+8={low_thresh:.0f}', candidate))

    # 选择最佳策略：选亮像素占比在0.5%~30%之间的（合理范围）
    best_strategy = None
    best_bright = None
    best_bright_pct = 0
    for name, candidate in strategies_tried:
        bright_pct = np.sum(candidate > 0) / candidate.size
        # 理想范围：0.5%~30%
        if 0.005 <= bright_pct <= 0.30:
            if best_strategy is None or abs(bright_pct - 0.05) < abs(best_bright_pct - 0.05):
                best_strategy = name
                best_bright = candidate
                best_bright_pct = bright_pct
        elif best_strategy is None:
            # 都不在理想范围，选最接近的
            best_strategy = name
            best_bright = candidate
            best_bright_pct = bright_pct

    bright = best_bright

    # 如果白色像素太多（反色图像），反转
    if np.sum(bright > 0) / bright.size > 0.7:
        bright = cv2.bitwise_not(bright)

    # 形态学去噪：开运算去除孤立噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

    # 找所有数字轮廓
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 收集所有可能的数字块
    digit_blocks = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 4 or ch < 8:  # 太小，跳过
            continue
        aspect = ch / cw if cw > 0 else 0
        if aspect < 0.3 or aspect > 5:  # 形状不像数字
            continue
        # 面积不能太大（排除大块背景）
        if cw * ch > fw * fh * 0.3:
            continue
        digit_blocks.append((x, y, cw, ch))

    if len(digit_blocks) < 3:
        # 尝试不加形态学去噪再试一次
        if best_bright is not None:
            contours_raw, _ = cv2.findContours(best_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            digit_blocks = []
            for cnt in contours_raw:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw < 4 or ch < 8:
                    continue
                aspect = ch / cw if cw > 0 else 0
                if aspect < 0.3 or aspect > 5:
                    continue
                if cw * ch > fw * fh * 0.3:
                    continue
                digit_blocks.append((x, y, cw, ch))

    if len(digit_blocks) < 3:
        return []

    # 按y坐标分组（同一行），宽行容差适应不同分辨率
    digit_blocks.sort(key=lambda b: b[1])
    row_tolerance = max(30, fh * 0.04)  # 至少30px或画面高度的4%
    rows = []
    current_row = [digit_blocks[0]]
    row_y_center = digit_blocks[0][1] + digit_blocks[0][3] / 2

    for block in digit_blocks[1:]:
        x, y, cw, ch = block
        block_center_y = y + ch / 2
        if abs(block_center_y - row_y_center) < row_tolerance:
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

        # 加边距（按画面比例）
        pad_x = max(15, int((max_x - min_x) * 0.1))
        pad_y = max(10, int((max_y - min_y) * 0.15))
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
                merge_threshold = max(80, r1[3] * 1.5)  # 合并阈值自适应
                if gap < merge_threshold or overlap > -20:
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


def _extract_regions_from_binary(binary, fh, fw):
    """从二值图中提取连通区域作为ROI, 同时捕获宽区域附近的窄高数字"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wide_regions = []
    narrow_regions = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 6 or ch < 15 or cw > fw * 0.8:
            continue
        aspect = ch / cw if cw > 0 else 0
        pad_x = 10; pad_y = 8
        rx = max(0, x - pad_x); ry = max(0, y - pad_y)
        rw = min(fw - rx, cw + 2 * pad_x); rh = min(fh - ry, ch + 2 * pad_y)
        if cw > 60:
            wide_regions.append((rx, ry, rw, rh))
        elif aspect > 2.5:
            # 窄高数字(如"1"), 后面会合并到最近的宽区域
            narrow_regions.append((rx, ry, rw, rh))

    # 将窄高数字合并到垂直对齐的宽区域
    for nr in narrow_regions:
        nx, ny, nw, nh = nr
        best_wr, best_dist = None, 999
        for wr in wide_regions:
            wx, wy, ww, wh = wr
            # 垂直对齐: y中心接近, x接近且在右侧
            if abs((ny+nh//2) - (wy+wh//2)) < wh * 1.2:
                dist = nx - (wx + ww)
                if 0 <= dist < ww * 1.5 and dist < best_dist:
                    best_dist = dist
                    best_wr = wr
        if best_wr:
            wx, wy, ww, wh = best_wr
            new_r = (wx, wy, max(wx+ww, nx+nw) - wx, max(wh, nh))
            wide_regions.remove(best_wr)
            wide_regions.append(new_r)

    return wide_regions


def _merge_nearby_regions(regions, fh, fw):
    """迭代合并垂直方向接近的区域（同一时钟的日期行+时间行）"""
    if len(regions) < 2:
        return regions
    # 迭代合并直到稳定
    changed = True
    while changed and len(regions) > 1:
        changed = False
        regions.sort(key=lambda r: r[1])
        merged = []
        skip_next = False
        for i in range(len(regions)):
            if skip_next:
                skip_next = False
                continue
            r1 = regions[i]
            if i + 1 < len(regions):
                r2 = regions[i + 1]
                gap = r2[1] - (r1[1] + r1[3])
                # 垂直距离小于高者1.5倍 → 同一时钟
                if gap < max(r1[3], r2[3]) * 1.5:
                    nx = min(r1[0], r2[0])
                    ny = r1[1]
                    nw = max(r1[0]+r1[2], r2[0]+r2[2]) - nx
                    nh = r2[1] + r2[3] - r1[1]
                    merged.append((nx, ny, nw, nh))
                    skip_next = True
                    changed = True
                    continue
            merged.append(r1)
        regions = merged
    return regions
