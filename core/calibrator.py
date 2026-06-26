"""
校准计算模块
-----------
时间偏差计算：被校时间 - 标准时间。
自动处理不同显示格式的比对。
"""

from typing import Optional

from utils.time_utils import (
    TimeParseResult, parse_time_string,
    normalize_times_for_comparison, calculate_time_deviation,
    format_time_deviation, format_datetime_display,
)


class CalibrationResult:
    """一次校准的结果"""

    def __init__(self,
                 cal_result: TimeParseResult,
                 std_result: TimeParseResult):
        self.cal_result = cal_result
        self.std_result = std_result

        # 规范化后的比对时间
        self.cal_dt, self.std_dt = normalize_times_for_comparison(
            cal_result, std_result)

        # 时间偏差（秒，正=被校快）
        self.deviation_seconds = (
            self.cal_dt - self.std_dt).total_seconds()

    @property
    def deviation_str(self) -> str:
        """偏差的可读字符串"""
        return format_time_deviation(self.deviation_seconds)

    @property
    def deviation_ppm(self) -> Optional[float]:
        """
        相对频率偏差 (ppm)。

        仅在两个时钟都有毫秒分量时计算，
        基于钟差变化率估算。
        """
        return None  # 单帧校准不适用ppm，需多次测量

    @property
    def cal_display(self) -> str:
        """被校时间显示字符串"""
        return format_datetime_display(
            self.cal_result.datetime,
            self.cal_result.has_full_date,
            self.cal_result.has_millisecond,
        )

    @property
    def std_display(self) -> str:
        """标准时间显示字符串"""
        return format_datetime_display(
            self.std_result.datetime,
            self.std_result.has_full_date,
            self.std_result.has_millisecond,
        )

    @property
    def comparison_note(self) -> str:
        """比对方式说明"""
        cal = self.cal_result
        std = self.std_result

        parts = []
        if cal.has_full_date and std.has_full_date:
            parts.append("年月日+时分秒完整比对")
        elif cal.has_full_date or std.has_full_date:
            parts.append("仅比对时分秒（忽略年月日差异）")

        if cal.has_millisecond and std.has_millisecond:
            parts.append("含毫秒比对")
        elif cal.has_millisecond or std.has_millisecond:
            parts.append("毫秒部分已忽略")

        if not parts:
            parts.append("时分秒比对")

        return "，".join(parts)

    def __repr__(self) -> str:
        return (f"CalibrationResult(cal={self.cal_display}, "
                f"std={self.std_display}, "
                f"deviation={self.deviation_str})")


class Calibrator:
    """校准器：OCR文本 → 偏差计算"""

    @staticmethod
    def calibrate(cal_text: str, std_text: str) -> Optional[CalibrationResult]:
        """
        从两段OCR文本计算校准结果。

        Args:
            cal_text: 被校时钟OCR文本
            std_text: 标准时钟OCR文本

        Returns:
            CalibrationResult 或 None（解析失败）
        """
        cal = parse_time_string(cal_text)
        std = parse_time_string(std_text)

        if cal is None or std is None:
            return None

        return CalibrationResult(cal, std)
