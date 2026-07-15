"""
七段数码管OCR（规则法）
--------------------
基于7段区域亮度判定的数字识别，无需训练模型。

七段布局:       aaa
               f   b
               f   b
                ggg
               e   c
               e   c
                ddd

每段一个检测区域，统计亮像素占比 → 高于阈值则该段点亮。
7位二进制码 → 映射到数字0-9。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# 七段码 → 数字映射 (a,b,c,d,e,f,g)
SEGMENT_TO_DIGIT = {
    0b1111110: '0',  # a,b,c,d,e,f
    0b0110000: '1',  # b,c
    0b1101101: '2',  # a,b,d,e,g
    0b1111001: '3',  # a,b,c,d,g
    0b0110011: '4',  # b,c,f,g
    0b1011011: '5',  # a,c,d,f,g
    0b1011111: '6',  # a,c,d,e,f,g
    0b1110000: '7',  # a,b,c
    0b1111111: '8',  # a,b,c,d,e,f,g
    0b1111011: '9',  # a,b,c,d,f,g
}

# 允许1bit容错的映射（某些段可能因光照/角度误判）
_FUZZY_MAP = {}
for code, digit in SEGMENT_TO_DIGIT.items():
    _FUZZY_MAP[code] = digit
    # 翻转1位
    for bit in range(7):
        flipped = code ^ (1 << bit)
        if flipped not in _FUZZY_MAP:
            _FUZZY_MAP[flipped] = digit


class SevenSegmentOCR:
    """
    七段数码管数字识别器。

    用法:
        ocr = SevenSegmentOCR()
        text = ocr.read_display(roi_image)  # → "08:26:00"
    """

    def __init__(self, brightness_threshold: float = 0.25):
        """
        Args:
            brightness_threshold: 段点亮阈值
        """
        self.brightness_threshold = brightness_threshold
        self._templates = None  # 懒加载模板

    def read_display(self, roi: np.ndarray) -> str:
        """
        识别整个时钟ROI中的时间字符串。

        Args:
            roi: 时钟显示区域图像（BGR或灰度）

        Returns:
            识别出的时间字符串，如 "08:26:00"
        """
        # 预处理
        binary = self._preprocess(roi)

        # 找数字轮廓
        digit_contours = self._find_digit_contours(binary)

        if not digit_contours:
            return ""

        # 按x坐标排序（从左到右）
        digit_contours.sort(key=lambda c: cv2.boundingRect(c)[0])

        # 识别每个字符
        chars = []
        prev_x2 = -100
        for cnt in digit_contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # 过滤太小的区域
            if w < 6 or h < 12:
                continue

            # 提取数字ROI
            digit_roi = binary[y:y + h, x:x + w]

            # 判断是数字还是分隔符（冒号/点）
            aspect = h / w if w > 0 else 99

            if aspect > 3.5:
                # 高宽比极大的 → 可能是冒号（两个点）
                if self._is_colon(digit_roi, binary[y:y + h, x:x + w]):
                    chars.append(':')
                    continue

            # 判断是否为小数点
            if h < 20 and w < 12:
                if self._is_dot(digit_roi):
                    chars.append('.')
                    continue

            # 正常数字 → 7段识别
            digit = self._recognize_digit(digit_roi)
            if digit != '?':
                # 避免重复识别（同一个数字可能被分成多个轮廓）
                if chars and chars[-1] not in ':. ' and x - prev_x2 < w * 0.3:
                    continue
                chars.append(digit)
                prev_x2 = x + w

        # 组装字符串
        return self._assemble(chars)

    def recognize_single(self, digit_roi: np.ndarray) -> str:
        """识别单个数字图像 → '0'~'9' 或 '?'"""
        binary = self._preprocess(digit_roi)
        result = self._recognize_digit(binary)
        if result == '?':
            result = self._recognize_density(binary)
        return result

    # ---- 像素密度分类器（通用回退） ----

    def _recognize_density(self, binary: np.ndarray) -> str:
        """
        基于像素密度特征的数字识别。

        将数字图像分为4x4网格，统计每格亮像素比例，
        与标准0-9模板比较欧氏距离。
        适用于任何数字字体，无需训练。
        """
        h, w = binary.shape
        if h < 8 or w < 4:
            return '?'

        # 缩放到标准尺寸
        std_w, std_h = 16, 24
        scaled = cv2.resize(binary, (std_w, std_h), interpolation=cv2.INTER_AREA)
        scaled = (scaled > 64).astype(np.uint8) * 255

        # 4x4网格特征
        grid = 4
        features = np.zeros(grid * grid)
        for gy in range(grid):
            for gx in range(grid):
                cell = scaled[gy * std_h // grid:(gy + 1) * std_h // grid,
                              gx * std_w // grid:(gx + 1) * std_w // grid]
                features[gy * grid + gx] = np.sum(cell > 0) / cell.size

        # 与模板比较
        templates = self._get_density_templates()
        best_digit, best_dist = '?', 999
        for digit, tmpl in templates.items():
            dist = np.sum((features - tmpl) ** 2)
            if dist < best_dist:
                best_dist, best_digit = dist, digit

        return best_digit if best_dist < 2.0 else '?'

    def _get_density_templates(self) -> dict:
        """生成0-9的像素密度模板（从合成七段数字提取）"""
        if self._templates is not None:
            return self._templates

        templates = {}
        std_w, std_h = 16, 24
        grid = 4

        for digit in '0123456789':
            # 绘制合成数字
            img = np.zeros((std_h, std_w), dtype=np.uint8)
            self._draw_seven_seg(img, digit, std_h, std_w)

            # 提取密度特征
            feat = np.zeros(grid * grid)
            for gy in range(grid):
                for gx in range(grid):
                    cell = img[gy * std_h // grid:(gy + 1) * std_h // grid,
                               gx * std_w // grid:(gx + 1) * std_w // grid]
                    feat[gy * grid + gx] = np.sum(cell > 0) / cell.size
            templates[digit] = feat

        self._templates = templates
        return templates

    # ---- 内部方法 ----

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """预处理：灰度化 → 亮度拉伸 → 高斯去噪 → 自适应二值化"""
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # 亮度拉伸
        mean_val = np.mean(gray)
        if mean_val < 80:
            p2, p98 = np.percentile(gray, (2, 98))
            if p98 > p2 + 5:
                gray = np.clip((gray.astype(float) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)

        # 高斯去噪（比双边滤波更适合七段数码管）
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        # 自适应二值化（比OTSU更擅长保留细笔划）
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 15, 3)

        # 确保暗底亮字
        white_ratio = np.sum(binary > 0) / binary.size
        if white_ratio > 0.7:
            binary = cv2.bitwise_not(binary)

        return binary

    def _find_digit_contours(self, binary: np.ndarray) -> list:
        """找到所有可能的数字轮廓"""
        # 轻微膨胀连接断裂笔画
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    def _get_templates(self, h: int, w: int) -> dict:
        """生成七段数码管0-9的模板图像，匹配输入尺寸"""
        templates = {}
        for digit in '0123456789':
            tmpl = np.zeros((h, w), dtype=np.uint8)
            self._draw_seven_seg(tmpl, digit, h, w)
            templates[digit] = tmpl
        return templates

    def _draw_seven_seg(self, img: np.ndarray, digit: str, h: int, w: int):
        """在图像上绘制一个七段数字"""
        segs = {  # 每段的像素坐标范围 (相对比例)
            '0': ['a','b','c','d','e','f'],
            '1': ['b','c'],
            '2': ['a','b','d','e','g'],
            '3': ['a','b','c','d','g'],
            '4': ['b','c','f','g'],
            '5': ['a','c','d','f','g'],
            '6': ['a','c','d','e','f','g'],
            '7': ['a','b','c'],
            '8': ['a','b','c','d','e','f','g'],
            '9': ['a','b','c','d','f','g'],
        }
        active = set(segs.get(digit, []))
        m = 3  # margin
        t = 3  # thickness
        seg_defs = {
            'a': (m, m, w-2*m, t),
            'b': (w-m-t, m, t, h//2-m),
            'c': (w-m-t, h//2, t, h//2-m),
            'd': (m, h-m-t, w-2*m, t),
            'e': (m, h//2, t, h//2-m),
            'f': (m, m, t, h//2-m),
            'g': (m, h//2-t//2, w-2*m, t),
        }
        for name in active:
            rx, ry, rw, rh = seg_defs[name]
            rx = max(0, min(rx, w-1))
            ry = max(0, min(ry, h-1))
            rw = max(1, min(rw, w-rx))
            rh = max(1, min(rh, h-ry))
            img[ry:ry+rh, rx:rx+rw] = 255

    def _recognize_digit(self, binary: np.ndarray) -> str:
        """
        识别单个数字：
        先模板匹配（整体形状最可靠），失败后用七段区域法。
        """
        h, w = binary.shape
        if h < 8 or w < 4:
            return '?'

        # 方法1：模板匹配（合成七段模板，整体形状匹配）
        result = self._recognize_template(binary)
        if result != '?':
            return result

        # 方法2：七段区域法回退
        return self._recognize_segment(binary)

    def _recognize_segment(self, binary: np.ndarray) -> str:
        """七段区域亮度法"""
        h, w = binary.shape
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.08)
        seg_h = int(h * 0.15)
        seg_w = int(w * 0.20)
        mid_y = h // 2

        regions = {
            'a': (margin_x, margin_y, w-2*margin_x, seg_h),
            'b': (w-margin_x-seg_w, margin_y+seg_h, seg_w, mid_y-margin_y-seg_h//2),
            'c': (w-margin_x-seg_w, mid_y+seg_h//2, seg_w, mid_y-margin_y-seg_h//2),
            'd': (margin_x, h-margin_y-seg_h, w-2*margin_x, seg_h),
            'e': (margin_x, mid_y+seg_h//2, seg_w, mid_y-margin_y-seg_h//2),
            'f': (margin_x, margin_y+seg_h, seg_w, mid_y-margin_y-seg_h//2),
            'g': (margin_x, mid_y-seg_h//2, w-2*margin_x, seg_h),
        }

        code = 0
        for i, seg in enumerate(['a','b','c','d','e','f','g']):
            rx, ry, rw, rh = regions[seg]
            rx = max(0, min(rx, w-1))
            ry = max(0, min(ry, h-1))
            rw = max(2, min(rw, w-rx))
            rh = max(2, min(rh, h-ry))
            seg_area = binary[ry:ry+rh, rx:rx+rw]
            if np.sum(seg_area > 0) / seg_area.size > self.brightness_threshold:
                code |= (1 << i)

        if code in SEGMENT_TO_DIGIT:
            return SEGMENT_TO_DIGIT[code]
        if code in _FUZZY_MAP:
            return _FUZZY_MAP[code]
        # 汉明距离最近
        best_d, best_dist = '?', 99
        for kc, d in SEGMENT_TO_DIGIT.items():
            dist = bin(code ^ kc).count('1')
            if dist < best_dist:
                best_dist, best_d = dist, d
        return best_d if best_dist <= 2 else '?'

    def _recognize_template(self, binary: np.ndarray) -> str:
        """模板匹配法"""
        h, w = binary.shape
        templates = self._get_templates(h, w)
        best_digit, best_score = '?', 0
        for digit, tmpl in templates.items():
            score = cv2.matchTemplate(binary, tmpl, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best_score, best_digit = score, digit
        return best_digit if best_score > 0.15 else '?'

    def _is_colon(self, roi: np.ndarray, full_roi: np.ndarray) -> bool:
        """判断是否为冒号（两个上下排列的亮点）"""
        h, w = roi.shape
        if h < w * 2:
            return False

        # 检查是否有两个亮点
        top = roi[:h // 3, :]
        bottom = roi[2 * h // 3:, :]

        top_bright = np.sum(top > 0) / top.size if top.size > 0 else 0
        bottom_bright = np.sum(bottom > 0) / bottom.size if bottom.size > 0 else 0

        return top_bright > 0.3 and bottom_bright > 0.3

    def _is_dot(self, roi: np.ndarray) -> bool:
        """判断是否为小数点"""
        bright = np.sum(roi > 0) / roi.size if roi.size > 0 else 0
        return bright > 0.4

    def _assemble(self, chars: List[str]) -> str:
        """组装字符列表为时间字符串"""
        text = ''.join(chars)
        # 移除首尾的非数字字符
        text = text.strip(':. ')
        return text
