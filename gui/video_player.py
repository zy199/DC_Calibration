"""
视频播放器控件
------------
基于 QGraphicsView + OpenCV 的视频播放组件。
支持：播放/暂停、逐帧进退、进度条拖拽、鼠标滚轮缩放。
"""

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QSizePolicy, QStyle
)
from PySide6.QtCore import (
    Qt, QTimer, Signal, QRectF
)
from PySide6.QtGui import (
    QImage, QPixmap, QWheelEvent, QMouseEvent,
    QPainter, QFont
)

from core.video_io import VideoIO, FrameReadError
from utils.logger import logger


class VideoPlayer(QWidget):
    """
    视频播放器控件。

    信号:
        frame_changed(int): 当前帧索引变化时发射
        video_loaded(str): 视频加载成功时发射（传递路径）
        video_error(str): 视频加载失败时发射（传递错误信息）
    """

    frame_changed = Signal(int)
    video_loaded = Signal(str)
    video_error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._video: VideoIO | None = None
        self._current_frame_idx: int = 0
        self._playing: bool = False
        self._zoom: float = 1.0
        self._timer: QTimer | None = None
        self._scene: QGraphicsScene | None = None
        self._pixmap_item: QGraphicsPixmapItem | None = None

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        """构建UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ---- 视频显示区域 ----
        self.graphics_view = QGraphicsView()
        self.graphics_view.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        self.graphics_view.setTransformationAnchor(
            QGraphicsView.AnchorUnderMouse)
        self.graphics_view.setViewportUpdateMode(
            QGraphicsView.FullViewportUpdate)  # 消除拖影
        self.graphics_view.setStyleSheet("""
            QGraphicsView {
                background-color: #0a0a14;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)

        # Scene
        self._scene = QGraphicsScene(self)
        self.graphics_view.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        # 初始占位文本
        self._placeholder = self._scene.addText(
            "📁 请加载视频文件",
            QFont("Microsoft YaHei", 16)
        )
        self._placeholder.setDefaultTextColor(
            Qt.GlobalColor.gray)
        self._placeholder.setPos(200, 150)

        layout.addWidget(self.graphics_view)

        # ---- 控制栏 ----
        controls = QHBoxLayout()
        controls.setContentsMargins(8, 8, 8, 4)
        controls.setSpacing(6)

        btn_style = """
            QPushButton {
                font-size: 14pt; padding: 0;
                background: #1a1a3e; border: 1px solid #444;
                border-radius: 4px; color: #ccc;
            }
            QPushButton:hover { background: #2a2a5e; border-color: #777; }
            QPushButton:disabled { background: #111; color: #444; border-color: #222; }
        """

        # 播放/暂停
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(48, 40)
        self.btn_play.setToolTip("播放/暂停 (Space)")
        self.btn_play.setStyleSheet(btn_style)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        controls.addWidget(self.btn_play)

        # 上一帧
        self.btn_prev_frame = QPushButton("|◀")
        self.btn_prev_frame.setFixedSize(48, 40)
        self.btn_prev_frame.setToolTip("上一帧 (←)")
        self.btn_prev_frame.setStyleSheet(btn_style)
        self.btn_prev_frame.clicked.connect(self.prev_frame)
        self.btn_prev_frame.setEnabled(False)
        controls.addWidget(self.btn_prev_frame)

        # 下一帧
        self.btn_next_frame = QPushButton("▶|")
        self.btn_next_frame.setFixedSize(48, 40)
        self.btn_next_frame.setToolTip("下一帧 (→)")
        self.btn_next_frame.setStyleSheet(btn_style)
        self.btn_next_frame.clicked.connect(self.next_frame)
        self.btn_next_frame.setEnabled(False)
        controls.addWidget(self.btn_next_frame)

        # 进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setToolTip("拖动跳转到指定帧")
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.setEnabled(False)
        controls.addWidget(self.slider)

        # 帧号/时间 标签
        self.frame_label = QLabel("-- / --")
        self.frame_label.setStyleSheet("color: #aaa; font-size: 10pt;")
        self.frame_label.setMinimumWidth(140)
        self.frame_label.setAlignment(Qt.AlignCenter)
        controls.addWidget(self.frame_label)

        # 缩放按钮
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedSize(40, 40)
        self.btn_zoom_in.setToolTip("放大 (Ctrl+滚轮)")
        self.btn_zoom_in.setStyleSheet(btn_style)
        self.btn_zoom_in.clicked.connect(lambda: self.zoom(1.2))
        self.btn_zoom_in.setEnabled(False)
        controls.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.setFixedSize(40, 40)
        self.btn_zoom_out.setToolTip("缩小 (Ctrl+滚轮)")
        self.btn_zoom_out.setStyleSheet(btn_style)
        self.btn_zoom_out.clicked.connect(lambda: self.zoom(1 / 1.2))
        self.btn_zoom_out.setEnabled(False)
        controls.addWidget(self.btn_zoom_out)

        self.btn_zoom_fit = QPushButton("⊡")
        self.btn_zoom_fit.setFixedSize(40, 40)
        self.btn_zoom_fit.setToolTip("适应窗口")
        self.btn_zoom_fit.setStyleSheet(btn_style)
        self.btn_zoom_fit.clicked.connect(self.fit_to_window)
        self.btn_zoom_fit.setEnabled(False)
        controls.addWidget(self.btn_zoom_fit)

        layout.addLayout(controls)

    def _setup_timer(self):
        """设置播放定时器"""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)

    # ---- 视频加载 ----

    def load_video(self, path: str):
        """
        加载视频文件。

        Args:
            path: 视频文件路径
        """
        # 释放之前的视频
        if self._video:
            self._video.close()

        try:
            self._video = VideoIO(path)
            self._current_frame_idx = 0
            self._playing = False
            self._zoom = 1.0

            # 移除占位文本
            if self._placeholder:
                self._scene.removeItem(self._placeholder)
                self._placeholder = None

            # 更新进度条
            self.slider.setRange(0, self._video.frame_count - 1)
            self.slider.setValue(0)

            # 启用控件
            self.btn_play.setEnabled(True)
            self.btn_prev_frame.setEnabled(True)
            self.btn_next_frame.setEnabled(True)
            self.slider.setEnabled(True)
            self.btn_zoom_in.setEnabled(True)
            self.btn_zoom_out.setEnabled(True)
            self.btn_zoom_fit.setEnabled(True)

            # 显示第一帧
            self._show_frame(0)

            self.video_loaded.emit(path)
            logger.info(f"视频播放器已加载: {path}")

        except FrameReadError as e:
            self.video_error.emit(str(e))
            logger.error(f"视频加载失败: {e}")

    def _show_frame(self, index: int):
        """
        在GraphicsView中显示指定帧。

        Args:
            index: 帧索引
        """
        if not self._video:
            return

        try:
            frame = self._video.get_frame(index)

            # BGR → RGB → QImage → QPixmap
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line,
                          QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimg)
            self._pixmap_item.setPixmap(pixmap)
            self._scene.setSceneRect(QRectF(pixmap.rect()))

            # 更新状态
            self._current_frame_idx = index
            self.slider.blockSignals(True)
            self.slider.setValue(index)
            self.slider.blockSignals(False)

            timestamp = self._video.get_frame_timestamp(index)
            self.frame_label.setText(
                f"帧 {index + 1}/{self._video.frame_count}  "
                f"{timestamp:.2f}s"
            )

            self.frame_changed.emit(index)

        except FrameReadError as e:
            logger.error(f"显示帧失败: {e}")

    # ---- 播放控制 ----

    def toggle_play(self):
        """切换播放/暂停"""
        if not self._video:
            return

        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self):
        """开始播放"""
        if not self._video or self._playing:
            return

        self._playing = True
        self.btn_play.setText("⏸")
        interval = int(1000 / self._video.fps)
        self._timer.start(max(interval, 16))  # 最快60fps
        logger.debug("开始播放")

    def pause(self):
        """暂停播放"""
        self._playing = False
        self._timer.stop()
        self.btn_play.setText("▶")
        logger.debug("暂停播放")

    def _on_timer_tick(self):
        """定时器回调：播放下一帧"""
        if not self._video:
            return

        next_idx = self._current_frame_idx + 1
        if next_idx >= self._video.frame_count:
            # 播完停止
            self.pause()
            return

        self._show_frame(next_idx)

    def prev_frame(self):
        """跳转到上一帧"""
        if not self._video or self._current_frame_idx <= 0:
            return
        self._show_frame(self._current_frame_idx - 1)

    def next_frame(self):
        """跳转到下一帧"""
        if not self._video or \
                self._current_frame_idx >= self._video.frame_count - 1:
            return
        self._show_frame(self._current_frame_idx + 1)

    def seek(self, frame_idx: int):
        """跳转到指定帧"""
        if not self._video:
            return
        frame_idx = max(0, min(frame_idx, self._video.frame_count - 1))
        self._show_frame(frame_idx)

    # ---- 进度条 ----

    def _on_slider_pressed(self):
        """进度条按下：拖动时暂停播放"""
        self._was_playing = self._playing
        if self._playing:
            self.pause()

    def _on_slider_released(self):
        """进度条释放：跳转到目标帧"""
        if not self._video:
            return
        target = self.slider.value()
        self._show_frame(target)
        if self._was_playing:
            self.play()

    # ---- 缩放 ----

    def zoom(self, factor: float):
        """缩放视图"""
        self._zoom *= factor
        self.graphics_view.scale(factor, factor)

    def fit_to_window(self):
        """适应窗口大小"""
        self.graphics_view.fitInView(
            self._scene.sceneRect(), Qt.KeepAspectRatio)
        # 近似缩放比例
        transform = self.graphics_view.transform()
        self._zoom = transform.m11()

    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放"""
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom(1.1)
            else:
                self.zoom(1 / 1.1)
            event.accept()
        else:
            super().wheelEvent(event)

    # ---- 公共接口 ----

    @property
    def video(self) -> VideoIO | None:
        """当前的VideoIO实例"""
        return self._video

    @property
    def current_frame_idx(self) -> int:
        """当前帧索引"""
        return self._current_frame_idx

    @property
    def current_frame(self) -> np.ndarray | None:
        """当前帧图像"""
        if not self._video:
            return None
        try:
            return self._video.get_frame(self._current_frame_idx)
        except FrameReadError:
            return None

    @property
    def scene(self) -> QGraphicsScene:
        """QGraphicsScene（给ROI选择器使用）"""
        return self._scene

    def reset_view(self):
        """重置视图"""
        self._zoom = 1.0
        self.graphics_view.resetTransform()
        if self._scene:
            self.graphics_view.fitInView(
                self._scene.sceneRect(), Qt.KeepAspectRatio)

    def close_video(self):
        """关闭视频"""
        self.pause()
        if self._video:
            self._video.close()
            self._video = None

        self._current_frame_idx = 0
        self.btn_play.setEnabled(False)
        self.btn_prev_frame.setEnabled(False)
        self.btn_next_frame.setEnabled(False)
        self.slider.setEnabled(False)
        self.btn_zoom_in.setEnabled(False)
        self.btn_zoom_out.setEnabled(False)
        self.btn_zoom_fit.setEnabled(False)
        self.frame_label.setText("-- / --")
