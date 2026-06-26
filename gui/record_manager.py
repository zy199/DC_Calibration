"""
记录管理对话框
-------------
查看、搜索、导出、删除历史校准记录。
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QFileDialog, QFrame, QAbstractItemView,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from data.database import get_database
from data.repository import CalibrationRepository
from data.models import CalibrationRecord
from config.settings import settings
from utils.logger import logger


class RecordManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("校准记录管理")
        self.setMinimumSize(900, 550)
        self.resize(1000, 600)
        self.setStyleSheet("""
            QDialog { background-color: #0a0a14; color: #e0e0e0; }
            QTableWidget {
                background-color: #0d0d1f; color: #e0e0e0;
                border: 1px solid #333; gridline-color: #222;
                selection-background-color: #1a237e;
            }
            QHeaderView::section {
                background-color: #111; color: #aaa; border: 1px solid #333;
                padding: 4px;
            }
            QLineEdit {
                background-color: #0a0a14; color: #e0e0e0;
                border: 1px solid #555; border-radius: 4px; padding: 4px 8px;
            }
        """)

        db = get_database(settings.database.db_path)
        self._repo = CalibrationRepository(db)
        self._records = []
        self._page = 0
        self._page_size = 50

        self._setup_ui()
        self._load_records()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 搜索栏
        search_bar = QHBoxLayout()
        search_bar.addWidget(QLabel("搜索:"))
        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("名称/编号（回车搜索）")
        self.edit_search.returnPressed.connect(self._on_search)
        search_bar.addWidget(self.edit_search)
        self.btn_search = QPushButton("🔍 搜索")
        self.btn_search.clicked.connect(self._on_search)
        search_bar.addWidget(self.btn_search)
        self.btn_all = QPushButton("📋 全部")
        self.btn_all.clicked.connect(self._load_records)
        search_bar.addWidget(self.btn_all)
        search_bar.addStretch()
        self.label_count = QLabel("")
        self.label_count.setStyleSheet("color: #888;")
        search_bar.addWidget(self.label_count)
        layout.addLayout(search_bar)

        # 表格
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "ID", "名称", "型号", "编号", "厂家", "送检单位",
            "被校时间", "标准时间", "偏差", "日期"
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        for i, w in enumerate([40, 80, 70, 70, 80, 80, 130, 130, 90, 80]):
            self.table.setColumnWidth(i, w)
        layout.addWidget(self.table)

        # 按钮栏
        btn_bar = QHBoxLayout()
        self.btn_view = QPushButton("👁 查看详情")
        self.btn_view.clicked.connect(self._on_view)
        btn_bar.addWidget(self.btn_view)
        self.btn_delete = QPushButton("🗑 删除")
        self.btn_delete.clicked.connect(self._on_delete)
        btn_bar.addWidget(self.btn_delete)
        btn_bar.addStretch()
        self.btn_export = QPushButton("📥 导出CSV")
        self.btn_export.clicked.connect(self._on_export)
        btn_bar.addWidget(self.btn_export)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.accept)
        btn_bar.addWidget(self.btn_close)
        layout.addLayout(btn_bar)

    def _load_records(self):
        self._records = self._repo.get_all(limit=500)
        self._page = 0
        self._refresh_table()

    def _on_search(self):
        keyword = self.edit_search.text().strip()
        if not keyword:
            self._load_records()
            return
        # 全字段模糊搜索
        self._records = self._repo.search_all_fields(keyword, limit=500)
        self._refresh_table()

    def _refresh_table(self):
        self.table.setRowCount(len(self._records))
        for i, rec in enumerate(self._records):
            self.table.setItem(i, 0, QTableWidgetItem(str(rec.id or '')))
            self.table.setItem(i, 1, QTableWidgetItem(rec.device_name))
            self.table.setItem(i, 2, QTableWidgetItem(rec.device_model))
            self.table.setItem(i, 3, QTableWidgetItem(rec.device_serial))
            self.table.setItem(i, 4, QTableWidgetItem(rec.manufacturer))
            self.table.setItem(i, 5, QTableWidgetItem(rec.send_unit))
            self.table.setItem(i, 6, QTableWidgetItem(rec.calibrated_time))
            self.table.setItem(i, 7, QTableWidgetItem(rec.standard_time))
            dev_str = f"{rec.time_deviation:+.3f}s"
            item = QTableWidgetItem(dev_str)
            if rec.time_deviation > 0:
                item.setForeground(Qt.GlobalColor.red)
            elif rec.time_deviation < 0:
                item.setForeground(Qt.GlobalColor.green)
            self.table.setItem(i, 8, item)
            self.table.setItem(i, 9, QTableWidgetItem(rec.calibration_date[:10]))
        self.label_count.setText(f"共 {len(self._records)} 条")

    def _on_view(self):
        row = self.table.currentRow()
        if row < 0:
            return
        rec = self._records[row]
        msg = (
            f"ID: {rec.id}\n\n"
            f"设备名称: {rec.device_name}\n"
            f"型号: {rec.device_model}\n"
            f"出厂编号: {rec.device_serial}\n"
            f"生产厂家: {rec.manufacturer}\n"
            f"送检单位: {rec.send_unit}\n\n"
            f"被校时间: {rec.calibrated_time}\n"
            f"标准时间: {rec.standard_time}\n"
            f"时间偏差: {rec.time_deviation:+.6f} 秒\n\n"
            f"校准日期: {rec.calibration_date}\n"
            f"视频文件: {rec.video_filename}\n"
            f"跳变帧: #{rec.jump_frame_idx}  清晰帧: #{rec.clarity_frame_idx}\n"
            f"备注: {rec.notes}"
        )
        QMessageBox.information(self, "记录详情", msg)

    def _on_delete(self):
        row = self.table.currentRow()
        if row < 0:
            return
        rec = self._records[row]
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除记录 ID={rec.id} 吗？\n此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._repo.delete(rec.id)
            self._records.pop(row)
            self._refresh_table()

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出CSV", "calibration_records.csv",
            "CSV文件 (*.csv)")
        if path:
            count = self._repo.export_to_csv(path)
            QMessageBox.information(self, "导出完成", f"已导出 {count} 条记录到:\n{path}")
