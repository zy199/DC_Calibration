"""
数据库模块
---------
SQLite连接管理、建表、基础操作。
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional

from utils.logger import logger

# 建表SQL
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS calibration_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,

    -- 设备信息
    device_name     TEXT DEFAULT '',
    device_model    TEXT DEFAULT '',
    device_serial   TEXT DEFAULT '',
    manufacturer    TEXT DEFAULT '',
    send_unit       TEXT DEFAULT '',

    -- 校准核心数据
    calibrated_time TEXT NOT NULL DEFAULT '',
    standard_time   TEXT NOT NULL DEFAULT '',
    time_deviation  REAL NOT NULL DEFAULT 0.0,

    -- 处理元数据
    video_path      TEXT DEFAULT '',
    video_filename  TEXT DEFAULT '',
    jump_frame_idx  INTEGER DEFAULT 0,
    clarity_frame_idx INTEGER DEFAULT 0,
    roi_calibrated  TEXT DEFAULT '',
    roi_standard    TEXT DEFAULT '',

    -- OCR元数据
    ocr_confidence  REAL DEFAULT 0.0,
    ocr_engine_used TEXT DEFAULT '',

    -- 记录元数据
    calibration_date TEXT DEFAULT '',
    calibrator      TEXT DEFAULT '',
    notes           TEXT DEFAULT '',

    -- 时间戳
    created_at      TIMESTAMP DEFAULT (datetime('now', 'localtime')),
    updated_at      TIMESTAMP DEFAULT (datetime('now', 'localtime'))
);
"""

# 索引SQL
CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_cal_date ON calibration_records(calibration_date);",
    "CREATE INDEX IF NOT EXISTS idx_device_name ON calibration_records(device_name);",
    "CREATE INDEX IF NOT EXISTS idx_device_serial ON calibration_records(device_serial);",
]


class Database:
    """SQLite数据库管理类"""

    def __init__(self, db_path: str = 'calibration_records.db'):
        """
        初始化数据库连接。

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        logger.info(f"数据库路径: {db_path}")

    @property
    def conn(self) -> sqlite3.Connection:
        """懒加载数据库连接"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row  # 支持按列名访问
            self._conn.execute("PRAGMA journal_mode=WAL;")  # WAL模式提升并发性能
            self._conn.execute("PRAGMA foreign_keys=ON;")
            logger.debug("数据库连接已建立")
        return self._conn

    def init_db(self):
        """初始化数据库表结构（含迁移）"""
        try:
            self.conn.execute(CREATE_TABLE_SQL)
            # 迁移：添加新列（如果不存在）
            for col in ['manufacturer', 'send_unit']:
                try:
                    self.conn.execute(f"ALTER TABLE calibration_records ADD COLUMN {col} TEXT DEFAULT ''")
                except sqlite3.OperationalError:
                    pass  # 列已存在
            for index_sql in CREATE_INDEXES_SQL:
                self.conn.execute(index_sql)
            self.conn.commit()
            logger.info("数据库表初始化完成")
        except sqlite3.Error as e:
            logger.error(f"数据库初始化失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("数据库连接已关闭")

    def backup(self, backup_path: str) -> bool:
        """
        备份数据库到指定路径。

        Args:
            backup_path: 备份文件路径

        Returns:
            是否成功
        """
        try:
            # 确保所有更改写入磁盘
            self.conn.commit()
            source = sqlite3.connect(self.db_path)
            dest = sqlite3.connect(backup_path)
            source.backup(dest)
            source.close()
            dest.close()
            logger.info(f"数据库已备份到: {backup_path}")
            return True
        except sqlite3.Error as e:
            logger.error(f"数据库备份失败: {e}")
            return False

    def restore(self, backup_path: str) -> bool:
        """
        从备份恢复数据库。

        注意：会覆盖当前数据库！

        Args:
            backup_path: 备份文件路径

        Returns:
            是否成功
        """
        if not os.path.exists(backup_path):
            logger.error(f"备份文件不存在: {backup_path}")
            return False
        try:
            self.close()
            source = sqlite3.connect(backup_path)
            dest = sqlite3.connect(self.db_path)
            source.backup(dest)
            source.close()
            dest.close()
            self._conn = None  # 重新懒加载
            logger.info(f"数据库已从备份恢复: {backup_path}")
            return True
        except sqlite3.Error as e:
            logger.error(f"数据库恢复失败: {e}")
            return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 默认数据库实例（使用配置中的路径）
_default_db: Optional[Database] = None


def get_database(db_path: str = 'calibration_records.db') -> Database:
    """获取默认数据库实例（单例模式）"""
    global _default_db
    if _default_db is None:
        _default_db = Database(db_path)
        _default_db.init_db()
    return _default_db
