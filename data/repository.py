"""
数据仓库模块
-----------
封装所有数据库CRUD操作，上层不直接写SQL。
"""

import csv
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from data.database import Database
from data.models import CalibrationRecord
from utils.logger import logger


class CalibrationRepository:
    """校准记录数据仓库"""

    def __init__(self, db: Database):
        """
        Args:
            db: Database实例
        """
        self.db = db

    def save(self, record: CalibrationRecord) -> int:
        """
        保存一条新记录。

        Args:
            record: 校准记录

        Returns:
            新记录的ID
        """
        d = record.to_dict()
        # 移除不应手动填充的字段
        for key in ['id', 'created_at', 'updated_at']:
            d.pop(key, None)

        columns = ', '.join(d.keys())
        placeholders = ', '.join(['?' for _ in d])
        values = list(d.values())

        sql = f"INSERT INTO calibration_records ({columns}) VALUES ({placeholders})"
        try:
            cursor = self.db.conn.execute(sql, values)
            self.db.conn.commit()
            record_id = cursor.lastrowid
            logger.info(f"记录已保存, ID={record_id}")
            return record_id
        except Exception as e:
            logger.error(f"保存记录失败: {e}")
            raise

    def get_by_id(self, record_id: int) -> Optional[CalibrationRecord]:
        """根据ID获取记录"""
        sql = "SELECT * FROM calibration_records WHERE id = ?"
        row = self.db.conn.execute(sql, (record_id,)).fetchone()
        if row:
            return CalibrationRecord.from_dict(dict(row))
        return None

    def get_all(self, order_by: str = 'created_at DESC',
                limit: int = 100, offset: int = 0) -> List[CalibrationRecord]:
        """获取所有记录（分页）"""
        sql = f"SELECT * FROM calibration_records ORDER BY {order_by} LIMIT ? OFFSET ?"
        rows = self.db.conn.execute(sql, (limit, offset)).fetchall()
        return [CalibrationRecord.from_dict(dict(row)) for row in rows]

    def search(self, filters: Dict[str, Any],
               order_by: str = 'created_at DESC',
               limit: int = 100, offset: int = 0) -> List[CalibrationRecord]:
        """
        多条件搜索。

        Args:
            filters: 筛选条件字典，支持的key:
                - device_name: 名称模糊匹配
                - device_serial: 出厂编号模糊匹配
                - calibration_date_from: 校准日期起始
                - calibration_date_to: 校准日期截止
            order_by: 排序
            limit: 每页条数
            offset: 偏移量

        Returns:
            匹配的记录列表
        """
        conditions = []
        params = []

        if 'device_name' in filters and filters['device_name']:
            conditions.append("device_name LIKE ?")
            params.append(f"%{filters['device_name']}%")

        if 'device_serial' in filters and filters['device_serial']:
            conditions.append("device_serial LIKE ?")
            params.append(f"%{filters['device_serial']}%")

        if 'calibration_date_from' in filters and filters['calibration_date_from']:
            conditions.append("calibration_date >= ?")
            params.append(filters['calibration_date_from'])

        if 'calibration_date_to' in filters and filters['calibration_date_to']:
            conditions.append("calibration_date <= ?")
            params.append(filters['calibration_date_to'])

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM calibration_records WHERE {where_clause} ORDER BY {order_by} LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.db.conn.execute(sql, params).fetchall()
        return [CalibrationRecord.from_dict(dict(row)) for row in rows]

    def update(self, record: CalibrationRecord):
        """更新一条记录"""
        d = record.to_dict()
        if 'id' not in d or d['id'] is None:
            raise ValueError("更新记录必须有id")

        d['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        record_id = d.pop('id')
        d.pop('created_at', None)

        set_clause = ', '.join([f"{k} = ?" for k in d.keys()])
        values = list(d.values()) + [record_id]

        sql = f"UPDATE calibration_records SET {set_clause} WHERE id = ?"
        try:
            self.db.conn.execute(sql, values)
            self.db.conn.commit()
            logger.info(f"记录已更新, ID={record_id}")
        except Exception as e:
            logger.error(f"更新记录失败: {e}")
            raise

    def delete(self, record_id: int):
        """删除一条记录"""
        sql = "DELETE FROM calibration_records WHERE id = ?"
        try:
            self.db.conn.execute(sql, (record_id,))
            self.db.conn.commit()
            logger.info(f"记录已删除, ID={record_id}")
        except Exception as e:
            logger.error(f"删除记录失败: {e}")
            raise

    def search_all_fields(self, keyword: str, limit: int = 500) -> List[CalibrationRecord]:
        """全字段模糊搜索"""
        sql = """SELECT * FROM calibration_records
                 WHERE device_name LIKE ? OR device_model LIKE ?
                 OR device_serial LIKE ? OR manufacturer LIKE ?
                 OR send_unit LIKE ? OR notes LIKE ?
                 OR calibrated_time LIKE ? OR standard_time LIKE ?
                 ORDER BY created_at DESC LIMIT ?"""
        kw = f"%{keyword}%"
        rows = self.db.conn.execute(sql, [kw, kw, kw, kw, kw, kw, kw, kw, limit]).fetchall()
        return [CalibrationRecord.from_dict(dict(row)) for row in rows]

    def count(self, filters: Dict[str, Any] = None) -> int:
        """统计记录总数"""
        if filters:
            conditions = []
            params = []
            if 'device_name' in filters and filters['device_name']:
                conditions.append("device_name LIKE ?")
                params.append(f"%{filters['device_name']}%")
            if 'device_serial' in filters and filters['device_serial']:
                conditions.append("device_serial LIKE ?")
                params.append(f"%{filters['device_serial']}%")
            where = " AND ".join(conditions) if conditions else "1=1"
            sql = f"SELECT COUNT(*) FROM calibration_records WHERE {where}"
            return self.db.conn.execute(sql, params).fetchone()[0]
        else:
            return self.db.conn.execute("SELECT COUNT(*) FROM calibration_records").fetchone()[0]

    def export_to_csv(self, filepath: str, filters: Dict[str, Any] = None) -> int:
        """
        导出记录到CSV文件。

        Args:
            filepath: CSV文件路径
            filters: 筛选条件（None=全部）

        Returns:
            导出的记录数
        """
        records = self.search(filters or {}, limit=999999)
        if not records:
            logger.warning("没有记录可导出")
            return 0

        fieldnames = [f for f in CalibrationRecord.__dataclass_fields__.keys()
                      if f not in ('id',)]

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = {}
                for k, v in record.to_dict().items():
                    if k not in fieldnames:
                        continue
                    # 时间字段加制表符前缀防Excel自动格式化
                    if k in ('calibrated_time', 'standard_time') and v:
                        row[k] = f"'{v}"  # 单引号前缀防Excel改写
                    else:
                        row[k] = v
                writer.writerow(row)

        logger.info(f"已导出 {len(records)} 条记录到 {filepath}")
        return len(records)
