"""
时间工具模块
-----------
时间解析、格式化、比对等辅助函数。

比对策略：
  当被校时钟只显示 HH:MM:SS，标准时钟显示 YYYY-MM-DD HH:MM:SS 时，
  只比较共同的时分秒部分，忽略年月日。
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List


# ---- 数据结构 ----

@dataclass
class TimeParseResult:
    """时间解析结果，含已解析字段信息"""
    datetime: datetime
    has_year: bool = False       # 原文是否包含年份
    has_month: bool = False      # 原文是否包含月份
    has_day: bool = False        # 原文是否包含日
    has_hour: bool = False       # 原文是否包含小时
    has_minute: bool = False     # 原文是否包含分钟
    has_second: bool = False     # 原文是否包含秒
    has_millisecond: bool = False  # 原文是否包含毫秒
    raw_text: str = ''           # 原始OCR文本

    @property
    def has_full_date(self) -> bool:
        """是否包含完整日期（年月日）"""
        return self.has_year and self.has_month and self.has_day

    @property
    def has_full_time(self) -> bool:
        """是否包含完整时间（时分秒）"""
        return self.has_hour and self.has_minute and self.has_second


# ---- 时间格式正则 ----

# 每个模式附带一个字段mask，标记该模式能匹配哪些日期组件
_TIME_PATTERN_SPECS = [
    # YYYY-MM-DD HH:MM:SS.mmm (完整日期+毫秒)
    (r'(?P<year>\d{4})[/\-年](?P<month>\d{1,2})[/\-月](?P<day>\d{1,2})\s*日?\s*(?P<hour>\d{1,2}):(?P<minute>\d{1,2}):(?P<second>\d{1,2})[\.:](?P<millisecond>\d{1,3})',
     dict(year=True, month=True, day=True, hour=True, minute=True, second=True, millisecond=True)),
    # YYYY-MM-DD HH:MM:SS (完整日期+时分秒)
    (r'(?P<year>\d{4})[/\-年](?P<month>\d{1,2})[/\-月](?P<day>\d{1,2})\s*日?\s*(?P<hour>\d{1,2}):(?P<minute>\d{1,2}):(?P<second>\d{1,2})',
     dict(year=True, month=True, day=True, hour=True, minute=True, second=True)),
    # HH:MM:SS.ms (时分秒+毫秒)
    (r'(?P<hour>\d{1,2}):(?P<minute>\d{1,2}):(?P<second>\d{1,2})[\.:](?P<millisecond>\d{1,3})',
     dict(hour=True, minute=True, second=True, millisecond=True)),
    # HH:MM:SS (时分秒)
    (r'(?P<hour>\d{1,2}):(?P<minute>\d{1,2}):(?P<second>\d{1,2})',
     dict(hour=True, minute=True, second=True)),
    # MM:SS (分秒，特殊场景)
    (r'(?P<minute>\d{1,2}):(?P<second>\d{1,2})',
     dict(minute=True, second=True)),
]


def parse_time_string(text: str) -> Optional[TimeParseResult]:
    """
    从OCR输出的文本中解析时间。

    按优先级尝试多种格式，返回第一个成功匹配的结果。
    缺失的日期字段用默认值填充（年份=2000，月=1，日=1），
    但通过 has_xxx 标记区分原文真实含有的字段。

    Args:
        text: OCR识别出的原始文本

    Returns:
        TimeParseResult 或 None
    """
    text = text.strip()

    for pattern, fields in _TIME_PATTERN_SPECS:
        match = re.search(pattern, text)
        if match:
            groups = match.groupdict()
            try:
                # 用2000-01-01作为"无日期信息"的占位基准
                year = int(groups.get('year', 2000))
                month = int(groups.get('month', 1))
                day = int(groups.get('day', 1))
                hour = int(groups.get('hour', 0))
                minute = int(groups.get('minute', 0))
                second = int(groups.get('second', 0))
                millisecond_str = groups.get('millisecond', '0')

                ms = int(millisecond_str.ljust(3, '0')[:3]) if millisecond_str else 0
                ms_micro = ms * 1000  # 毫秒 → 微秒

                return TimeParseResult(
                    datetime=datetime(year, month, day, hour, minute, second, ms_micro),
                    has_year=fields.get('year', False),
                    has_month=fields.get('month', False),
                    has_day=fields.get('day', False),
                    has_hour=fields.get('hour', False),
                    has_minute=fields.get('minute', False),
                    has_second=fields.get('second', False),
                    has_millisecond=fields.get('millisecond', False),
                    raw_text=text,
                )
            except (ValueError, OverflowError):
                continue

    return None


def normalize_times_for_comparison(
    cal: TimeParseResult,
    std: TimeParseResult
) -> tuple[datetime, datetime]:
    """
    规范化两个时间解析结果，使它们在同一基准上可比。

    规则：
    1. 如果任一缺少年月日 → 两者都对齐到同一基准日（2000-01-01）
    2. 如果任一缺少毫秒 → 毫秒部分置0
    3. 确保时间差计算只用两者共有的分量

    Args:
        cal: 被校时钟解析结果
        std: 标准时钟解析结果

    Returns:
        (cal_dt, std_dt) 规范化后的datetime对
    """
    cal_dt = cal.datetime
    std_dt = std.datetime

    # 如果任一时钟没有完整日期，将两者日期对齐到同一基准
    if not (cal.has_full_date and std.has_full_date):
        cal_dt = cal_dt.replace(year=2000, month=1, day=1)
        std_dt = std_dt.replace(year=2000, month=1, day=1)

    # 如果被校没有毫秒，只将被校毫秒置0（保留标准钟的毫秒精度）
    if not cal.has_millisecond:
        cal_dt = cal_dt.replace(microsecond=0)
    if not std.has_millisecond:
        std_dt = std_dt.replace(microsecond=0)

    return cal_dt, std_dt


def calculate_time_deviation(cal: TimeParseResult, std: TimeParseResult) -> float:
    """
    计算时间偏差（被校 - 标准），自动处理不同显示格式。

    Args:
        cal: 被校时钟解析结果
        std: 标准时钟解析结果

    Returns:
        时间偏差（秒），正值=被校比标准快
    """
    cal_dt, std_dt = normalize_times_for_comparison(cal, std)
    return (cal_dt - std_dt).total_seconds()


def format_time_deviation(seconds: float) -> str:
    """
    格式化时间偏差为可读字符串。

    Args:
        seconds: 偏差秒数（正=被校比标准快）

    Returns:
        格式化字符串，如 "+0.667 秒"
    """
    sign = '+' if seconds >= 0 else '-'
    abs_seconds = abs(seconds)

    if abs_seconds < 1:
        ms = abs_seconds * 1000
        return f"{sign}{ms:.3f} 毫秒"
    elif abs_seconds < 60:
        return f"{sign}{abs_seconds:.3f} 秒"
    elif abs_seconds < 3600:
        minutes = int(abs_seconds // 60)
        secs = abs_seconds % 60
        return f"{sign}{minutes}分{secs:.3f}秒"
    else:
        hours = int(abs_seconds // 3600)
        remainder = abs_seconds % 3600
        minutes = int(remainder // 60)
        secs = remainder % 60
        return f"{sign}{hours}时{minutes}分{secs:.3f}秒"


def format_datetime_iso(dt: datetime) -> str:
    """将datetime格式化为ISO 8601字符串"""
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def format_datetime_display(dt: datetime, has_date: bool, has_ms: bool) -> str:
    """
    按实际解析出的字段格式化时间显示。

    Args:
        dt: datetime对象
        has_date: 是否显示日期部分
        has_ms: 是否显示毫秒部分

    Returns:
        格式化字符串
    """
    if has_date and has_ms:
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    elif has_date:
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    elif has_ms:
        return dt.strftime('%H:%M:%S.%f')[:-3]
    else:
        return dt.strftime('%H:%M:%S')


def estimate_min_jump_interval(text: str) -> float:
    """
    根据显示时间文本，估算最小跳变间隔（秒）。

    Args:
        text: 显示时间的文本（如 "08:26:00.123"）

    Returns:
        预估的最小跳变间隔，单位秒
    """
    if '.' in text and len(text.split('.')[-1]) >= 3:
        return 0.001
    if '.' in text and len(text.split('.')[-1]) == 2:
        return 0.01
    if text.count(':') >= 2:
        return 1.0
    if ':' in text:
        return 1.0
    return 1.0
