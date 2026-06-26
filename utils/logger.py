"""
日志模块
-------
统一的日志配置，输出到控制台和文件。
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# 日志目录
LOG_DIR = Path(__file__).parent.parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# 日志文件（按日期命名）
LOG_FILE = LOG_DIR / f'calibration_{datetime.now().strftime("%Y%m%d")}.log'

# 日志格式
CONSOLE_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
FILE_FORMAT = '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'

_DATE_FORMAT = '%H:%M:%S'


def setup_logger(name: str = 'DC_Calibration',
                 level: int = logging.DEBUG) -> logging.Logger:
    """
    创建并配置logger实例。

    Args:
        name: logger名称
        level: 日志级别

    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(CONSOLE_FORMAT, datefmt=_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)

    # 文件handler
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(FILE_FORMAT, datefmt=_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 默认logger
logger = setup_logger()
