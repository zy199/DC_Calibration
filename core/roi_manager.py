"""
ROI坐标管理模块
--------------
ROI坐标的序列化/反序列化、验证。
"""

import json
from typing import Tuple, Optional, Dict, Any


def roi_to_dict(roi: Tuple[int, int, int, int], label: str = '') -> Dict[str, Any]:
    """ROI元组转为字典"""
    x, y, w, h = roi
    d = {'x': x, 'y': y, 'w': w, 'h': h}
    if label:
        d['label'] = label
    return d


def roi_from_dict(d: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """字典转为ROI元组"""
    return (d['x'], d['y'], d['w'], d['h'])


def roi_to_json(roi: Tuple[int, int, int, int], label: str = '') -> str:
    """ROI序列化为JSON字符串"""
    return json.dumps(roi_to_dict(roi, label), ensure_ascii=False)


def roi_from_json(json_str: str) -> Optional[Tuple[int, int, int, int]]:
    """从JSON字符串反序列化ROI"""
    try:
        d = json.loads(json_str)
        return roi_from_dict(d)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def validate_roi(roi: Tuple[int, int, int, int],
                 image_width: int,
                 image_height: int) -> bool:
    """
    验证ROI坐标是否在图像范围内。

    Args:
        roi: (x, y, w, h)
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        是否有效
    """
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0:
        return False
    if x + w > image_width or y + h > image_height:
        return False
    return True


def clamp_roi(roi: Tuple[int, int, int, int],
              image_width: int,
              image_height: int) -> Tuple[int, int, int, int]:
    """
    将ROI裁剪到图像范围内。

    Args:
        roi: (x, y, w, h)
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        裁剪后的ROI
    """
    x, y, w, h = roi
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    w = max(1, min(w, image_width - x))
    h = max(1, min(h, image_height - y))
    return (x, y, w, h)
