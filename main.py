"""
数字时钟自动校准系统 — 程序入口
================================
基于图像识别的数字时钟校准工具。
4步流程：加载视频 → 框选时钟 → 检测跳变+确认清晰帧 → 查看结果。
"""

import sys, os, cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QMessageBox,
    QFileDialog, QLineEdit,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QFont, QIcon, QPixmap, QImage, QDragEnterEvent,
    QDropEvent,
)

from config.settings import settings
from data.database import get_database
from data.models import CalibrationSession
from utils.logger import logger
from gui.video_player import VideoPlayer
from gui.roi_selector import ROISelector
from gui.worker_threads import JumpDetectionWorker, OCRWorker
from gui.record_manager import RecordManagerDialog
from core.jump_detector import JumpDetectionResult
from core.clarity_evaluator import ClarityEvaluator, ClarityResult
from utils.image_utils import detect_clock_regions

ICON_PATH = os.path.join(PROJECT_ROOT, 'resources', 'icons', 'app_icon.png')

# ============================================================
# Step 0: 欢迎/视频加载页
# ============================================================
class WelcomePage(QWidget):
    video_loaded_signal = Signal(str)

    def __init__(self, session: CalibrationSession, parent=None):
        super().__init__(parent)
        self.session = session
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(80, 40, 80, 40)

        # 顶部图标和标题
        header = QHBoxLayout()
        icon_lbl = QLabel()
        if os.path.exists(ICON_PATH):
            icon_lbl.setPixmap(QPixmap(ICON_PATH).scaled(
                64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        header.addWidget(icon_lbl)

        title_col = QVBoxLayout()
        title = QLabel("数字时钟自动校准系统")
        title.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        title.setStyleSheet("color: #e0e0ff;")
        title_col.addWidget(title)

        subtitle = QLabel("基于视频图像识别的时间频率计量校准工具")
        subtitle.setFont(QFont("Microsoft YaHei", 11))
        subtitle.setStyleSheet("color: #888;")
        title_col.addWidget(subtitle)
        header.addLayout(title_col)
        header.addStretch()

        # 右上角版权
        copyright_lbl = QLabel("© 2026 贵州省计量测试院 科研开发部")
        copyright_lbl.setStyleSheet("color: #555; font-size: 9pt;")
        copyright_lbl.setAlignment(Qt.AlignRight | Qt.AlignTop)
        header.addWidget(copyright_lbl)

        layout.addLayout(header)

        layout.addSpacing(25)

        # 工作流程说明
        steps_frame = QFrame()
        steps_frame.setStyleSheet(
            "background: #0d0d1f; border: 1px solid #2a2a4a; border-radius: 10px;")
        steps_layout = QHBoxLayout(steps_frame)
        steps_layout.setContentsMargins(20, 15, 20, 15)

        for i, (emoji, text) in enumerate([
            ("📂", "加载视频"), ("🎯", "框选时钟"),
            ("🔍", "检测跳变"), ("📊", "查看结果"),
        ]):
            step = QVBoxLayout()
            icon = QLabel(emoji)
            icon.setFont(QFont("Microsoft YaHei", 22))
            icon.setAlignment(Qt.AlignCenter)
            step.addWidget(icon)
            desc = QLabel(text)
            desc.setAlignment(Qt.AlignCenter)
            desc.setStyleSheet("color: #aaa; font-size: 10pt;")
            step.addWidget(desc)
            steps_layout.addLayout(step)
            if i < 3:
                arrow = QLabel("▸")
                arrow.setFont(QFont("Microsoft YaHei", 14))
                arrow.setAlignment(Qt.AlignCenter)
                arrow.setStyleSheet("color: #5C6BC0;")
                arrow.setFixedWidth(20)
                steps_layout.addWidget(arrow)
        layout.addWidget(steps_frame)

        layout.addSpacing(20)

        # 功能特点
        features = QLabel(
            "• 支持七段数码管、LCD等多种数字时钟显示类型\n"
            "• 自动检测时钟末位跳变帧，定位进位瞬间\n"
            "• 智能清晰度评估，自动选择最佳识别帧\n"
            "• 支持毫秒级时间分辨力，适配高速摄影场景\n"
            "• 自动计算时间偏差，一键保存校准记录\n"
            "• 请保持拍摄设备稳定，确保时钟数字清晰可见"
        )
        features.setStyleSheet("color: #777; font-size: 10pt; line-height: 1.6;")
        layout.addWidget(features)

        layout.addSpacing(15)

        # 历史记录按钮
        btn_row = QHBoxLayout()
        self.btn_history = QPushButton("📋 查看历史校准记录")
        self.btn_history.setFixedHeight(38)
        self.btn_history.clicked.connect(self._open_records)
        btn_row.addWidget(self.btn_history)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()

        # 拖放区域
        self.drop_area = QFrame()
        self.drop_area.setObjectName("drop_area")
        self.drop_area.setMinimumHeight(160)
        drop_layout = QVBoxLayout(self.drop_area)
        drop_layout.setAlignment(Qt.AlignCenter)
        self.drop_icon = QLabel("🎬")
        self.drop_icon.setFont(QFont("Microsoft YaHei", 36))
        self.drop_icon.setAlignment(Qt.AlignCenter)
        drop_layout.addWidget(self.drop_icon)
        self.drop_text = QLabel("拖拽视频文件到此处，或点击下方按钮选择")
        self.drop_text.setAlignment(Qt.AlignCenter)
        self.drop_text.setStyleSheet("color: #888; font-size: 11pt;")
        drop_layout.addWidget(self.drop_text)
        layout.addWidget(self.drop_area)

        layout.addSpacing(10)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_select = QPushButton("📂 选择视频文件")
        self.btn_select.setFixedSize(180, 42)
        self.btn_select.clicked.connect(self._on_select_file)
        btn_row.addWidget(self.btn_select)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addSpacing(5)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #4CAF50; font-size: 10pt;")
        layout.addWidget(self.info_label)

    def _on_select_file(self):
        formats = "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;所有文件 (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", formats)
        if path:
            self._load_video(path)

    def _load_video(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开视频文件:\n{path}")
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = fc / fps if fps > 0 else 0
        cap.release()
        fname = os.path.basename(path)

        self.session.video_path = path
        self.session.video_filename = fname
        self.session.video_fps = fps
        self.session.video_frame_count = fc
        self.session.video_duration = dur

        self.drop_icon.setText("✅")
        self.drop_text.setText(f"✓ {fname}")
        self.drop_text.setStyleSheet("color: #4CAF50; font-size: 11pt;")
        self.info_label.setText(
            f"分辨率: {w}×{h}  |  帧率: {fps:.1f} fps  |  "
            f"总帧数: {fc}  |  时长: {int(dur//60)}分{int(dur%60)}秒")
        self.video_loaded_signal.emit(path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_area.setStyleSheet(
                "QFrame#drop_area { border: 2px dashed #4CAF50; "
                "border-radius: 12px; background-color: #0d1f0d; }")

    def dragLeaveEvent(self, event):
        self.drop_area.setStyleSheet("")

    def dropEvent(self, event: QDropEvent):
        self.drop_area.setStyleSheet("")
        urls = event.mimeData().urls()
        if urls:
            self._load_video(urls[0].toLocalFile())

    def _open_records(self):
        dialog = RecordManagerDialog(self)
        dialog.exec()

    def is_ready(self) -> bool:
        return bool(self.session.video_path)


# ============================================================
# Step 1: ROI选择页（保持原有逻辑，微调UI）
# ============================================================
class ROISelectPage(QWidget):
    def __init__(self, session: CalibrationSession, parent=None):
        super().__init__(parent)
        self.session = session
        self._video_player: VideoPlayer | None = None
        self._roi_selector: ROISelector | None = None
        self._auto_detected: bool = False  # 只首次自动检测
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("步骤 2/4：框选时钟区域")
        title.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        layout.addWidget(title)
        hint = QLabel(
            '请拖动红色框覆盖「被校时钟」显示区域，绿色框覆盖「标准时钟」显示区域\n'
            '拖拽矩形四角可调整大小，方向键可微调位置（每次1px）')
        hint.setStyleSheet("color: #777; margin-bottom: 4px;")
        layout.addWidget(hint)
        self._video_player = VideoPlayer()
        layout.addWidget(self._video_player)
        self.roi_info = QLabel("🔴 被校时钟: 未选择    |    🟢 标准时钟: 未选择")
        self.roi_info.setStyleSheet("color: #888; font-size: 10pt;")
        layout.addWidget(self.roi_info)

    def on_enter(self):
        """进入页面：首次自动检测时钟 → 预选框选；再次进入保留调整"""
        if self._video_player and self.session.video_path:
            self._video_player.load_video(self.session.video_path)
            if not self._roi_selector:
                self._roi_selector = ROISelector(self._video_player.scene)
                self._roi_selector.roi_changed.connect(self._on_roi_changed)
            # 仅首次自动检测
            if not self._auto_detected:
                self._auto_detect_rois()
                self._auto_detected = True
            self._roi_selector.show()

    def _auto_detect_rois(self):
        """自动检测时钟区域并预选（用时间方差法，不受光照影响）"""
        try:
            import cv2
            cap = cv2.VideoCapture(self.session.video_path)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return
            regions = detect_clock_regions(frame, max_candidates=2)
            cap.release()
            if len(regions) >= 2:
                regions.sort(key=lambda r: r[1])  # 上→下
                self._roi_selector.rect_calibrated.set_roi(*regions[0])
                self._roi_selector.rect_standard.set_roi(*regions[1])
                self.roi_info.setText("🔍 已自动框选时钟行，请确认或微调")
            else:
                self.roi_info.setText("未检测到时钟，请手动框选（至少覆盖HH:MM）")
        except Exception as e:
            logger.debug(f"自动检测失败: {e}")

    def on_leave(self):
        if self._roi_selector:
            self.session.roi_calibrated = self._roi_selector.roi_calibrated
            self.session.roi_standard = self._roi_selector.roi_standard

    def _on_roi_changed(self):
        if not self._roi_selector:
            return
        cal = self._roi_selector.roi_calibrated
        std = self._roi_selector.roi_standard
        cal_s = f"({cal[0]},{cal[1]},{cal[2]}×{cal[3]})" if cal else "未选择"
        std_s = f"({std[0]},{std[1]},{std[2]}×{std[3]})" if std else "未选择"
        self.roi_info.setText(f"🔴 被校时钟: {cal_s}    |    🟢 标准时钟: {std_s}")

    def is_ready(self) -> bool:
        if self._roi_selector:
            return (self._roi_selector.roi_calibrated is not None and
                    self._roi_selector.roi_standard is not None)
        return False


# ============================================================
# Step 2: 跳变检测 + 清晰帧确认（合并页）
# ============================================================
class DetectionConfirmPage(QWidget):
    """跳变检测 + 5帧浏览器 + 大图预览，合并为一步"""

    jump_found = Signal()

    def __init__(self, session: CalibrationSession, parent=None):
        super().__init__(parent)
        self.session = session
        self._worker: JumpDetectionWorker | None = None
        self._evaluator = ClarityEvaluator()
        self._detection_result: JumpDetectionResult | None = None
        self._clarity_result: ClarityResult | None = None
        self._frame_cache = {}  # idx -> QPixmap
        self._selected_idx: int = -1
        self._current_jump_idx: int = -1
        self._jump_attempt: int = 0
        self._video_path: str = ''
        self._detection_done: bool = False
        self._confirmed: bool = False  # 用户是否已确认当前清晰帧
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("步骤 3/4：检测跳变并确认清晰帧")
        title.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        layout.addWidget(title)

        # 主区域：大图预览 + 5帧选择
        main_area = QHBoxLayout()

        # 左侧大图预览
        self.big_frame = QLabel()
        self.big_frame.setMinimumSize(500, 280)
        self.big_frame.setAlignment(Qt.AlignCenter)
        self.big_frame.setStyleSheet(
            "background: #0a0a14; border: 1px solid #333; border-radius: 6px;")
        main_area.addWidget(self.big_frame, 3)

        # 右侧信息面板
        info_panel = QVBoxLayout()

        # 状态
        self.status_label = QLabel("准备检测...")
        self.status_label.setStyleSheet("color: #aaa; font-size: 11pt;")
        info_panel.addWidget(self.status_label)

        # 跳变信息
        self.jump_info = QLabel("")
        self.jump_info.setStyleSheet("color: #888; font-size: 10pt;")
        self.jump_info.setWordWrap(True)
        info_panel.addWidget(self.jump_info)

        info_panel.addStretch()

        # 重新寻找按钮
        self.btn_redetect = QPushButton("🔍 重新寻找跳变帧")
        self.btn_redetect.setVisible(False)
        self.btn_redetect.clicked.connect(self._on_redetect)
        info_panel.addWidget(self.btn_redetect)

        # 找下一个按钮
        self.btn_retry = QPushButton("🔄 找下一个跳变")
        self.btn_retry.setVisible(False)
        self.btn_retry.clicked.connect(self._find_next_jump)
        info_panel.addWidget(self.btn_retry)

        self.btn_confirm = QPushButton("✅ 确认此帧")
        self.btn_confirm.setVisible(False)
        self.btn_confirm.clicked.connect(self._on_confirm)
        info_panel.addWidget(self.btn_confirm)

        main_area.addLayout(info_panel, 1)
        layout.addLayout(main_area)

        # 5帧缩略图行
        thumbs_label = QLabel("跳变帧前后预览（点击切换大图）：")
        thumbs_label.setStyleSheet("color: #888; font-size: 10pt; margin-top: 8px;")
        layout.addWidget(thumbs_label)

        thumbs_layout = QHBoxLayout()
        self.thumb_buttons = []
        self.thumb_scores = []
        for i in range(5):
            vbox = QVBoxLayout()
            btn = QPushButton()
            btn.setFixedSize(160, 100)
            btn.setStyleSheet(
                "QPushButton { background: #0a0a14; border: 2px solid #333; "
                "border-radius: 4px; padding: 0; }"
                "QPushButton:hover { border-color: #666; }")
            btn.clicked.connect(lambda checked, idx=i: self._on_thumb_click(idx))
            vbox.addWidget(btn)
            lbl = QLabel("")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #888; font-size: 9pt;")
            vbox.addWidget(lbl)
            thumbs_layout.addLayout(vbox)
            self.thumb_buttons.append(btn)
            self.thumb_scores.append(lbl)
        layout.addLayout(thumbs_layout)

        # 进度条
        self.progress_frame = QFrame()
        self.progress_frame.setFixedHeight(8)
        self.progress_frame.setStyleSheet(
            "background: #3a3a4a; border: 1px solid #555; border-radius: 4px;")
        layout.addWidget(self.progress_frame)

        self.progress_fill = QFrame(self.progress_frame)
        self.progress_fill.setFixedHeight(6)
        self.progress_fill.move(1, 1)
        self.progress_fill.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #1a237e, stop:0.5 #3949ab, stop:1 #5c6bc0); "
            "border-radius: 3px;")
        self.progress_fill.setFixedWidth(0)

    def on_enter(self):
        """进入页面 → 首次自动检测，切换视频也自动重检"""
        if not self.session.video_path or not self.session.roi_calibrated:
            self.status_label.setText("❌ 请先完成前两步")
            return

        # 检测到切换了视频 → 清除旧缓存 + 立即清空UI
        if self._video_path and self._video_path != self.session.video_path:
            self._detection_done = False
            self._clarity_result = None
            self._frame_cache.clear()
            self._confirmed = False
            # 清空旧视频的缩略图和预览
            self.big_frame.clear()
            for btn in self.thumb_buttons:
                btn.setIcon(QIcon())
            for lbl in self.thumb_scores:
                lbl.setText("")
            self.jump_info.setText("")
            self.btn_retry.setVisible(False)
            self.btn_redetect.setVisible(False)
            self.btn_confirm.setVisible(False)
        self._video_path = self.session.video_path

        # 如果有已完成的有效缓存（同一视频），直接显示
        if self._detection_done and self._clarity_result:
            self._load_preview_frames()
            if self.session.clarity_frame_idx >= 0:
                self._select_frame(self.session.clarity_frame_idx)
            else:
                self._select_frame(self._clarity_result.best_idx)
            if self._confirmed:
                self.status_label.setText(
                    f"✅ 已确认帧 #{self._selected_idx} | 可以进入下一步")
                self.btn_confirm.setText("✅ 已确认")
                self.btn_confirm.setEnabled(False)
            else:
                self.status_label.setText(
                    f"✅ 已加载 | 推荐帧 #{self._selected_idx} | 请点击「确认此帧」")
                self.btn_confirm.setText("✅ 确认此帧")
                self.btn_confirm.setEnabled(True)
            self.btn_retry.setVisible(True)
            self.btn_redetect.setVisible(True)
            self.btn_confirm.setVisible(True)
            return

        # 新视频或无缓存 → 自动检测
        self._jump_attempt = 0
        self._start_detection(0)

    def on_leave(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()

    def _on_redetect(self):
        """手动重新从0开始寻找跳变帧"""
        self._detection_done = False
        self._clarity_result = None
        self._frame_cache.clear()
        self._jump_attempt = 0
        self._current_jump_idx = -1
        self._selected_idx = -1
        self.session.jump_frame_idx = -1
        self.session.clarity_frame_idx = -1
        self.btn_retry.setVisible(False)
        self.btn_redetect.setVisible(False)
        self.btn_confirm.setVisible(False)
        self.jump_info.setText("")
        self.big_frame.clear()
        for btn in self.thumb_buttons:
            btn.setIcon(QIcon())
        for lbl in self.thumb_scores:
            lbl.setText("")
        self._start_detection(0)

    def _start_detection(self, start_frame: int):
        """启动跳变检测"""
        self._jump_attempt += 1
        self._confirmed = False
        self.session.clarity_frame_idx = -1
        self.btn_confirm.setText("✅ 确认此帧")
        self.btn_confirm.setEnabled(True)
        self.status_label.setText(f"🔍 正在搜索跳变帧...")
        self.progress_fill.setFixedWidth(0)

        self._worker = JumpDetectionWorker(
            video_path=self._video_path,
            roi_calibrated=self.session.roi_calibrated,
        )
        # 注入start_frame
        self._worker._start_frame = start_frame
        self._worker.progress.connect(self._on_detect_progress)
        self._worker.finished.connect(self._on_detect_done)
        self._worker.error.connect(self._on_detect_error)
        self._worker.start()

    def _on_detect_progress(self, ratio: float):
        w = self.progress_frame.width()
        self.progress_fill.setFixedWidth(int(w * ratio))

    def _on_detect_done(self, result: JumpDetectionResult):
        self._detection_result = result

        if not result.success:
            self.status_label.setText(
                "❌ 未找到跳变帧 | 请检查ROI或点击重新寻找")
            self._update_jump_info(result)
            self.btn_redetect.setVisible(True)
            return

        self._current_jump_idx = result.jump_frame_idx
        self._update_jump_info(result)
        self.status_label.setText(
            f"✅ 找到跳变帧 #{result.jump_frame_idx}，正在评估清晰度...")
        self._evaluate_clarity(result.jump_frame_idx)

    def _update_jump_info(self, result: JumpDetectionResult):
        if result.success:
            self.jump_info.setText(
                f"跳变帧: 第 {result.jump_frame_idx} 帧\n"
                f"视频时间: {result.timestamp:.3f}s\n"
                f"检测耗时: {result.detection_time_ms:.0f}ms")
        else:
            self.jump_info.setText("状态: 未找到跳变")

    def _evaluate_clarity(self, jump_idx: int):
        """评估跳变帧及前后帧清晰度"""
        import cv2

        cap = cv2.VideoCapture(self._video_path)

        def getter(idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)))
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"无法读取帧{idx}")
            return frame

        result = self._evaluator.evaluate_window(
            getter, jump_idx,
            self.session.roi_calibrated,
            self.session.roi_standard,
            window=2,
            jump_idx=jump_idx,
        )
        # 补充SSIM一致性（evaluate_window已做跳变帧加分+85%规则，这里只补consistency）
        self._evaluator.evaluate_consistency(
            result.scores, getter, self.session.roi_calibrated)
        # 重算combined（evaluate_window已算过但consistency刚填充，需更新）
        w_lap = self._evaluator.config.laplacian_weight
        w_trans = self._evaluator.config.transition_weight
        w_cons = self._evaluator.config.consistency_weight
        for s in result.scores:
            if s.frame_idx == jump_idx:
                s.consistency = 1.0
            s.combined = w_lap * s.laplacian + w_trans * s.transition + w_cons * s.consistency
            if s.frame_idx == jump_idx:
                s.combined += 0.25
            elif s.frame_idx > jump_idx:
                s.combined += 0.03
        best = max(result.scores, key=lambda s: s.combined)
        jump_s = next((s for s in result.scores if s.frame_idx == jump_idx), None)
        if jump_s and best.frame_idx != jump_idx and jump_s.combined >= best.combined * 0.70:
            best = jump_s
        result.best_idx = best.frame_idx
        result.best_score = best.combined

        cap.release()
        self._clarity_result = result

        # 加载帧图像
        self._load_preview_frames()

        # 默认选中推荐帧
        self._select_frame(result.best_idx)

        # 如果不够清晰，提示用户
        if not result.is_clear and result.best_score < 0.25:
            self.status_label.setText(
                f"⚠️ 清晰度偏低（{result.best_score:.2f}），可点击「找下一个跳变」")
        else:
            self.status_label.setText(
                f"✅ 检测完成 | 推荐帧 #{result.best_idx} | "
                f"清晰度: {result.best_score:.2f}")

        self._detection_done = True
        self.btn_redetect.setVisible(True)
        self.btn_retry.setVisible(True)
        self.btn_confirm.setVisible(True)
        self.jump_found.emit()

    def _load_preview_frames(self):
        """加载5帧（跳变±2）到大图预览用的缓存"""
        if not self._clarity_result:
            return

        import cv2
        cap = cv2.VideoCapture(self._video_path)

        self._frame_cache.clear()
        for s in self._clarity_result.scores:
            idx = s.frame_idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
                self._frame_cache[idx] = QPixmap.fromImage(qimg)

        cap.release()

        # 更新缩略图
        thumb_labels = ["前2帧", "前1帧", "跳变帧", "后1帧", "后2帧"]
        for i, s in enumerate(self._clarity_result.scores):
            if s.frame_idx in self._frame_cache:
                pix = self._frame_cache[s.frame_idx].scaled(
                    160, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.thumb_buttons[i].setIcon(QIcon(pix))
                self.thumb_buttons[i].setIconSize(pix.size())
                self.thumb_scores[i].setText(
                    f"{thumb_labels[i]}\n#{s.frame_idx}  {s.combined:.2f}分")
            # 边框颜色
            if s.frame_idx == self._clarity_result.best_idx:
                self.thumb_buttons[i].setStyleSheet(
                    "QPushButton { background: #0a0a14; "
                    "border: 2px solid #4CAF50; border-radius: 4px; padding: 0; }"
                    "QPushButton:hover { border-color: #66BB6A; }")
            else:
                self.thumb_buttons[i].setStyleSheet(
                    "QPushButton { background: #0a0a14; "
                    "border: 2px solid #333; border-radius: 4px; padding: 0; }"
                    "QPushButton:hover { border-color: #666; }")

    def _on_thumb_click(self, thumb_idx: int):
        """点击缩略图 → 切换大图预览"""
        if not self._clarity_result:
            return
        if thumb_idx < len(self._clarity_result.scores):
            idx = self._clarity_result.scores[thumb_idx].frame_idx
            self._select_frame(idx)

    def _select_frame(self, idx: int):
        """选中某帧，更新大图和高亮"""
        self._selected_idx = idx
        if idx in self._frame_cache:
            pix = self._frame_cache[idx].scaled(
                self.big_frame.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation)
            self.big_frame.setPixmap(pix)

        # 更新缩略图高亮
        if self._clarity_result:
            for i, s in enumerate(self._clarity_result.scores):
                if s.frame_idx == idx:
                    self.thumb_buttons[i].setStyleSheet(
                        "QPushButton { background: #0a0a14; "
                        "border: 2px solid #2196F3; border-radius: 4px; padding: 0; }"
                        "QPushButton:hover { border-color: #42A5F5; }")
                elif s.frame_idx == self._clarity_result.best_idx:
                    self.thumb_buttons[i].setStyleSheet(
                        "QPushButton { background: #0a0a14; "
                        "border: 2px solid #4CAF50; border-radius: 4px; padding: 0; }"
                        "QPushButton:hover { border-color: #66BB6A; }")
                else:
                    self.thumb_buttons[i].setStyleSheet(
                        "QPushButton { background: #0a0a14; "
                        "border: 2px solid #333; border-radius: 4px; padding: 0; }"
                        "QPushButton:hover { border-color: #666; }")

    def _find_next_jump(self):
        """找下一个跳变帧"""
        if self._current_jump_idx < 0:
            return
        next_start = self._current_jump_idx + 2  # +2避免跳过紧邻的跳变
        if next_start >= self.session.video_frame_count:
            self.status_label.setText("⚠️ 已到达视频末尾，没有更多跳变")
            return
        self._confirmed = False
        self.btn_confirm.setText("✅ 确认此帧")
        self.btn_confirm.setEnabled(True)
        self._start_detection(next_start)

    def _on_confirm(self):
        """确认选中帧"""
        self._confirmed = True
        self.session.jump_frame_idx = self._current_jump_idx
        self.session.clarity_frame_idx = self._selected_idx
        self.status_label.setText(
            f"✅ 已确认帧 #{self._selected_idx} | 可以进入下一步")
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.setText("✅ 已确认")
        self._update_navbar_if_main()

    def _on_detect_error(self, err_msg: str):
        self.status_label.setText(f"❌ 检测出错: {err_msg}")

    def _update_navbar_if_main(self):
        main = self._find_main_window()
        if main:
            main._update_navbar()

    def _find_main_window(self):
        w = self.parent()
        while w:
            if isinstance(w, MainWindow):
                return w
            w = w.parent()
        return None

    def is_ready(self) -> bool:
        return self._confirmed and self.session.jump_frame_idx >= 0


# ============================================================
# Step 3: 结果确认页
# ============================================================
class ResultPage(QWidget):
    def __init__(self, session: CalibrationSession, parent=None):
        super().__init__(parent)
        self.session = session
        self._ocr_engine = None
        self._cal_result = None
        self._setup_ui()

    def _setup_ui(self):
        from PySide6.QtWidgets import QScrollArea
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        title = QLabel("步骤 4/4：确认识别结果并保存")
        title.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        layout.addWidget(title)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.viewport().setStyleSheet("background: transparent;")
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        cl = QVBoxLayout(content)
        cl.setSpacing(8)
        cl.setContentsMargins(0, 0, 0, 0)

        input_style = ("color: #e0e0e0; background: #0a0a14; border: 1px solid #555; "
                       "border-radius: 4px; padding: 5px 8px;")

        # === 时间识别区 ===
        time_box = QFrame()
        time_box.setStyleSheet("background: #0d0d1f; border: 1px solid #333; border-radius: 6px; padding: 10px;")
        tl = QVBoxLayout(time_box)
        tl.setSpacing(6)

        # 被校时钟行（预览图+输入框）
        row_cal = QHBoxLayout()
        self.img_cal = QLabel()
        self.img_cal.setFixedSize(300, 70)
        self.img_cal.setAlignment(Qt.AlignCenter)
        self.img_cal.setStyleSheet("background: #050510; border: 1px solid #333; border-radius: 3px;")
        self.img_cal.setScaledContents(True)
        row_cal.addWidget(self.img_cal)
        row_cal.addWidget(QLabel("🔴"))
        self.edit_cal = QLineEdit()
        self.edit_cal.setFont(QFont("Consolas", 18))
        self.edit_cal.setStyleSheet(input_style)
        self.edit_cal.setPlaceholderText("如 17:31:32.75")
        self.edit_cal.textChanged.connect(self._on_time_edited)
        row_cal.addWidget(self.edit_cal)
        self.cal_status = QLabel("")
        self.cal_status.setStyleSheet("color: #888; font-size: 9pt;")
        self.cal_status.setFixedWidth(130)
        row_cal.addWidget(self.cal_status)
        tl.addLayout(row_cal)

        # 标准时钟行
        row_std = QHBoxLayout()
        self.img_std = QLabel()
        self.img_std.setFixedSize(300, 70)
        self.img_std.setAlignment(Qt.AlignCenter)
        self.img_std.setStyleSheet("background: #050510; border: 1px solid #333; border-radius: 3px;")
        self.img_std.setScaledContents(True)
        row_std.addWidget(self.img_std)
        row_std.addWidget(QLabel("🟢"))
        self.edit_std = QLineEdit()
        self.edit_std.setFont(QFont("Consolas", 18))
        self.edit_std.setStyleSheet(input_style)
        self.edit_std.setPlaceholderText("如 2022-10-28 17:30:45.708")
        self.edit_std.textChanged.connect(self._on_time_edited)
        row_std.addWidget(self.edit_std)
        self.std_status = QLabel("")
        self.std_status.setStyleSheet("color: #888; font-size: 9pt;")
        self.std_status.setFixedWidth(130)
        row_std.addWidget(self.std_status)
        tl.addLayout(row_std)

        # 偏差
        self.deviation_label = QLabel("时间偏差: --")
        self.deviation_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.deviation_label.setAlignment(Qt.AlignCenter)
        self.deviation_label.setStyleSheet("color: #FFD54F; padding: 4px;")
        tl.addWidget(self.deviation_label)
        self.compare_note = QLabel("")
        self.compare_note.setAlignment(Qt.AlignCenter)
        self.compare_note.setStyleSheet("color: #777; font-size: 9pt;")
        tl.addWidget(self.compare_note)

        cl.addWidget(time_box)

        # === 设备信息区 ===
        info_box = QFrame()
        info_box.setStyleSheet("background: #0d0d1f; border: 1px solid #333; border-radius: 6px; padding: 10px;")
        il = QVBoxLayout(info_box)
        il.setSpacing(6)
        il.addWidget(QLabel("📋 设备信息（选填）"))

        # 第一行：名称 | 型号
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("名称"))
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("数字时钟A")
        self.edit_name.setStyleSheet(input_style)
        r1.addWidget(self.edit_name)
        r1.addWidget(QLabel("型号"))
        self.edit_model = QLineEdit()
        self.edit_model.setPlaceholderText("SYN3102")
        self.edit_model.setStyleSheet(input_style)
        r1.addWidget(self.edit_model)
        il.addLayout(r1)

        # 第二行：编号 | 生产厂家
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("编号"))
        self.edit_serial = QLineEdit()
        self.edit_serial.setPlaceholderText("202406001")
        self.edit_serial.setStyleSheet(input_style)
        r2.addWidget(self.edit_serial)
        r2.addWidget(QLabel("厂家"))
        self.edit_manufacturer = QLineEdit()
        self.edit_manufacturer.setPlaceholderText("生产厂家")
        self.edit_manufacturer.setStyleSheet(input_style)
        r2.addWidget(self.edit_manufacturer)
        il.addLayout(r2)

        # 第三行：送检单位 | 备注
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("送检"))
        self.edit_send_unit = QLineEdit()
        self.edit_send_unit.setPlaceholderText("送检单位")
        self.edit_send_unit.setStyleSheet(input_style)
        r3.addWidget(self.edit_send_unit)
        r3.addWidget(QLabel("备注"))
        self.edit_notes = QLineEdit()
        self.edit_notes.setPlaceholderText("温度23℃")
        self.edit_notes.setStyleSheet(input_style)
        r3.addWidget(self.edit_notes)
        il.addLayout(r3)

        cl.addWidget(info_box)

        # === 保存按钮 ===
        self.btn_save = QPushButton("💾 保存校准记录")
        self.btn_save.setFixedHeight(40)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save.setEnabled(False)
        cl.addWidget(self.btn_save)

        # === 新校准按钮 ===
        self.btn_new = QPushButton("🔄 开始新的校准")
        self.btn_new.setFixedHeight(40)
        self.btn_new.clicked.connect(self._on_new_calibration)
        self.btn_new.setVisible(False)
        cl.addWidget(self.btn_new)

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def on_enter(self):
        """进入页面：加载清晰帧 → 后台OCR识别（不卡UI）"""
        try:
            import cv2
            cap = cv2.VideoCapture(self.session.video_path)
            idx = self.session.clarity_frame_idx
            if idx < 0:
                idx = self.session.jump_frame_idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                self.cal_status.setText("❌ 无法读取帧")
                return

            # 显示ROI预览图
            self._show_roi_preview(frame)

            # 显示进度
            self.cal_status.setText("⏳ 识别中...")
            self.std_status.setText("⏳ 识别中...")
            self.edit_cal.setEnabled(False)
            self.edit_std.setEnabled(False)

            # 后台线程跑OCR
            self._ocr_worker = OCRWorker(
                frame,
                self.session.roi_calibrated,
                self.session.roi_standard,
            )
            self._ocr_worker.progress.connect(self._on_ocr_progress)
            self._ocr_worker.finished.connect(self._on_ocr_done)
            self._ocr_worker.error.connect(self._on_ocr_error)
            self._ocr_worker.start()

        except Exception as e:
            self.cal_status.setText(f"❌ 出错: {e}")

    def _on_ocr_progress(self, msg: str):
        self.cal_status.setText(f"⏳ {msg}")
        self.std_status.setText("")

    def _on_ocr_done(self, cal_text: str, std_text: str):
        self.edit_cal.setEnabled(True)
        self.edit_std.setEnabled(True)

        if cal_text and len(cal_text) >= 5:
            self.edit_cal.setText(cal_text)
            self.cal_status.setText("OCR自动识别")
        else:
            self.cal_status.setText("⚠️ 识别失败，请手动输入")

        if std_text and len(std_text) >= 5:
            self.edit_std.setText(std_text)
            self.std_status.setText("OCR自动识别")
        else:
            self.std_status.setText("⚠️ 识别失败，请手动输入")

        self._on_time_edited()

    def _on_ocr_error(self, err: str):
        self.edit_cal.setEnabled(True)
        self.edit_std.setEnabled(True)
        self.cal_status.setText(f"⚠️ OCR失败，请手动输入")
        self.std_status.setText("")

    def _show_roi_preview(self, frame):
        """显示被校和标准时钟的ROI预览图"""
        try:
            import cv2
            x1, y1, w1, h1 = self.session.roi_calibrated
            x2, y2, w2, h2 = self.session.roi_standard
            cal_roi = frame[y1:y1+h1, x1:x1+w1]
            std_roi = frame[y2:y2+h2, x2:x2+w2]

            for img_roi, label in [(cal_roi, self.img_cal), (std_roi, self.img_std)]:
                rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                qimg = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
                label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass

    def _on_time_edited(self):
        """用户编辑时间后实时重算偏差，并学习数字模板"""
        cal_text = self.edit_cal.text().strip()
        std_text = self.edit_std.text().strip()

        if cal_text and std_text:
            from utils.time_utils import parse_time_string, calculate_time_deviation, format_time_deviation
            cal_parse = parse_time_string(cal_text)
            std_parse = parse_time_string(std_text)

            if cal_parse and std_parse:
                deviation = calculate_time_deviation(cal_parse, std_parse)
                self._cal_result = (cal_parse, std_parse, deviation)
                self.deviation_label.setText(
                    f"时间偏差: {format_time_deviation(deviation)}")
                self.compare_note.setText(
                    f"{'被校比标准快' if deviation > 0 else '被校比标准慢' if deviation < 0 else '一致'}")
                self.btn_save.setEnabled(True)

                # 自学习：从当前帧和正确文本中提取数字模板
                self._learn_templates(cal_text, std_text)
                return

        self.deviation_label.setText("请输入有效时间")
        self.btn_save.setEnabled(False)

    def _learn_templates(self, cal_text: str, std_text: str):
        """从手动输入的准确文本学习数字模板"""
        try:
            from core.ocr_engine import OCREngine
            import cv2
            cap = cv2.VideoCapture(self.session.video_path)
            idx = self.session.clarity_frame_idx
            if idx < 0:
                idx = self.session.jump_frame_idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return

            x1, y1, w1, h1 = self.session.roi_calibrated
            x2, y2, w2, h2 = self.session.roi_standard
            engine = OCREngine()
            engine.learn_from_text(frame[y1:y1+h1, x1:x1+w1], cal_text)
            engine.learn_from_text(frame[y2:y2+h2, x2:x2+w2], std_text)
            logger.info("数字模板学习完成")
        except Exception as e:
            logger.debug(f"模板学习失败: {e}")

    def _on_save(self):
        """保存校准记录到数据库"""
        if not self._cal_result:
            QMessageBox.warning(self, "提示", "没有可保存的校准结果")
            return

        cal_parse, std_parse, deviation = self._cal_result

        from data.repository import CalibrationRepository
        from data.models import CalibrationRecord
        from datetime import datetime
        import json

        # 用编辑框显示的文本（保留用户确认的精度）
        cal_time_str = self.edit_cal.text().strip()
        std_time_str = self.edit_std.text().strip()

        record = CalibrationRecord(
            device_name=self.edit_name.text().strip(),
            device_model=self.edit_model.text().strip(),
            device_serial=self.edit_serial.text().strip(),
            manufacturer=self.edit_manufacturer.text().strip(),
            send_unit=self.edit_send_unit.text().strip(),
            calibrated_time=cal_time_str,
            standard_time=std_time_str,
            time_deviation=deviation,
            video_path=self.session.video_path,
            video_filename=self.session.video_filename,
            jump_frame_idx=self.session.jump_frame_idx,
            clarity_frame_idx=self.session.clarity_frame_idx,
            roi_calibrated=json.dumps(list(self.session.roi_calibrated)) if self.session.roi_calibrated else '',
            roi_standard=json.dumps(list(self.session.roi_standard)) if self.session.roi_standard else '',
            calibration_date=datetime.now().strftime('%Y-%m-%d'),
            notes=self.edit_notes.text().strip(),
        )

        try:
            db = get_database()
            repo = CalibrationRepository(db)
            record_id = repo.save(record)
            QMessageBox.information(self, "保存成功",
                                    f"校准记录已保存！\n记录ID: {record_id}")
            self.btn_save.setText(f"✅ 已保存 (ID: {record_id})")
            self.btn_save.setEnabled(False)
            # 显示"开始新的校准"按钮
            self.btn_new.setVisible(True)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _on_new_calibration(self):
        """开始新的校准：重置状态并返回首页"""
        # 重置session
        self.session.video_path = ''
        self.session.video_filename = ''
        self.session.roi_calibrated = None
        self.session.roi_standard = None
        self.session.jump_frame_idx = -1
        self.session.clarity_frame_idx = -1
        self._cal_result = None
        # 清空界面
        self.edit_cal.clear()
        self.edit_std.clear()
        self.edit_name.clear()
        self.edit_model.clear()
        self.edit_serial.clear()
        self.edit_manufacturer.clear()
        self.edit_send_unit.clear()
        self.edit_notes.clear()
        self.deviation_label.setText("时间偏差: --")
        self.compare_note.setText("")
        self.btn_save.setText("💾 保存校准记录")
        self.btn_save.setEnabled(False)
        self.btn_new.setVisible(False)
        # 返回首页
        main = self._find_main_window()
        if main:
            main.current_step = 0
            main.stack.setCurrentIndex(0)
            main._update_navbar()
            # 重置各页面状态
            main.pages[1]._auto_detected = False
            main.pages[2]._detection_done = False
            main.pages[2]._confirmed = False
            main.pages[2]._clarity_result = None

    def _find_main_window(self):
        w = self.parent()
        while w:
            if isinstance(w, MainWindow):
                return w
            w = w.parent()
        return None

    def is_ready(self) -> bool:
        return True


# ============================================================
# 主窗口（4步骤）
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db = get_database(settings.database.db_path)
        self.db.init_db()
        self.session = CalibrationSession()

        self.setWindowTitle(settings.ui.window_title)
        self.setMinimumSize(960, 640)
        self.resize(settings.ui.window_default_width,
                    settings.ui.window_default_height)

        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))

        self.stack = QStackedWidget()
        self.pages = {
            0: WelcomePage(self.session),
            1: ROISelectPage(self.session),
            2: DetectionConfirmPage(self.session),
            3: ResultPage(self.session),
        }
        for p in self.pages.values():
            self.stack.addWidget(p)

        # 信号连接
        self.pages[0].video_loaded_signal.connect(self._on_video_loaded)

        self._setup_navbar()

        self.current_step = 0
        self.stack.setCurrentIndex(0)
        self.statusBar().showMessage("就绪")

    def _on_video_loaded(self, path: str):
        self.statusBar().showMessage(f"已加载: {self.session.video_filename}")
        # 重置ROI自动检测
        if hasattr(self.pages[1], '_auto_detected'):
            self.pages[1]._auto_detected = False

    def _setup_navbar(self):
        navbar = QWidget()
        navbar.setFixedHeight(56)
        navbar.setStyleSheet("background: #0d0d1a; border-top: 1px solid #333;")
        layout = QHBoxLayout(navbar)
        layout.setContentsMargins(20, 8, 20, 8)

        self.step_indicator = QLabel("步骤 0/4")
        self.step_indicator.setStyleSheet("color: #aaa; font-size: 11pt;")
        layout.addWidget(self.step_indicator)

        # 步骤点
        self.step_dots = []
        for i in range(4):
            dot = QLabel("○")
            dot.setStyleSheet("color: #555; font-size: 14pt;")
            layout.addWidget(dot)
            self.step_dots.append(dot)

        layout.addStretch()

        self.btn_prev = QPushButton("◀ 上一步")
        self.btn_prev.setFixedWidth(100)
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._on_prev)
        layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("下一步 ▸")
        self.btn_next.setFixedWidth(100)
        self.btn_next.clicked.connect(self._on_next)
        layout.addWidget(self.btn_next)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.stack)
        main_layout.addWidget(navbar)
        self.setCentralWidget(central)

    def _on_prev(self):
        if self.current_step > 0:
            self._leave_current()
            self.current_step -= 1
            self.stack.setCurrentIndex(self.current_step)
            self._enter_current()
            self._update_navbar()

    def _on_next(self):
        page = self.pages[self.current_step]
        if hasattr(page, 'is_ready') and not page.is_ready():
            QMessageBox.information(self, "提示", "请先完成当前步骤再继续。")
            return
        if self.current_step < len(self.pages) - 1:
            self._leave_current()
            self.current_step += 1
            self.stack.setCurrentIndex(self.current_step)
            self._enter_current()
            self._update_navbar()

    def _leave_current(self):
        page = self.pages[self.current_step]
        if hasattr(page, 'on_leave'):
            page.on_leave()

    def _enter_current(self):
        page = self.pages[self.current_step]
        if hasattr(page, 'on_enter'):
            page.on_enter()

    def _update_navbar(self):
        step_1idx = self.current_step + 1  # 1-indexed显示
        self.step_indicator.setText(f"步骤 {step_1idx}/4")
        self.btn_prev.setEnabled(self.current_step > 0)
        self.btn_next.setText("完成 ✓" if self.current_step == 3 else "下一步 ▸")
        for i, dot in enumerate(self.step_dots):
            if i == self.current_step:
                dot.setText("●")
                dot.setStyleSheet("color: #5C6BC0; font-size: 14pt;")
            elif i < self.current_step:
                dot.setText("●")
                dot.setStyleSheet("color: #4CAF50; font-size: 14pt;")
            else:
                dot.setText("○")
                dot.setStyleSheet("color: #555; font-size: 14pt;")
        self.statusBar().showMessage("就绪")


# ============================================================
# 样式表
# ============================================================
APP_STYLESHEET = """
QMainWindow { background-color: #0a0a14; color: #e0e0e0; }
QWidget { background-color: #0a0a14; color: #e0e0e0; }
QLabel { color: #e0e0e0; background: transparent; }
QPushButton {
    background-color: #1a1a3e; color: #e0e0e0;
    border: 1px solid #444; border-radius: 6px;
    padding: 8px 16px; font-size: 11pt;
}
QPushButton:hover { background-color: #2a2a5e; border-color: #666; }
QPushButton:pressed { background-color: #151530; }
QPushButton:disabled { background-color: #111; color: #555; border-color: #222; }
QFrame#drop_area {
    border: 2px dashed #444; border-radius: 12px; background-color: #0d0d1f;
}
QFrame#drop_area:hover { border-color: #788; background-color: #12122a; }
QStatusBar { background: #0d0d1a; color: #888; border-top: 1px solid #333; }
QLineEdit {
    background-color: #0a0a14; color: #e0e0e0;
    border: 1px solid #555; border-radius: 4px; padding: 4px 8px;
}
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical {
    background: #0a0a14; width: 8px; border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #333; border-radius: 4px; min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QMessageBox {
    background-color: #0d0d1f; color: #e0e0e0;
}
QMessageBox QLabel { color: #e0e0e0; }
QMessageBox QPushButton {
    background-color: #1a1a3e; color: #e0e0e0;
    border: 1px solid #444; border-radius: 4px;
    padding: 6px 20px; min-width: 80px;
}
"""


def main():
    logger.info("=" * 50)
    logger.info("数字时钟自动校准系统 启动")
    logger.info("=" * 50)

    app = QApplication(sys.argv)
    app.setApplicationName("DC_Calibration")
    app.setStyleSheet(APP_STYLESHEET)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.show()
    logger.info("主窗口已显示")
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
