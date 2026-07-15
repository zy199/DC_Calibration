"""
ROI框选覆盖层
------------
在QGraphicsScene上叠加可拖拽、可调整大小的矩形，
用于框选"被校时钟"和"标准时钟"区域。
"""

from enum import Enum
from typing import Tuple, Optional

from PySide6.QtWidgets import (
    QGraphicsRectItem, QGraphicsScene, QGraphicsItem,
    QGraphicsTextItem, QWidget
)
from PySide6.QtCore import (
    Qt, QRectF, QPointF, Signal, QObject
)
from PySide6.QtGui import (
    QPen, QBrush, QColor, QPainter, QFont,
    QCursor
)


class HandlePosition(Enum):
    """控制手柄位置"""
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    NONE = 4


class SelectableRect(QGraphicsRectItem):
    """
    可拖拽、可调整大小的矩形。

    特点：
    - 整体拖拽移动
    - 四角8px手柄拖拽调整大小
    - 自定义颜色和标签
    - 通过 geometry_changed 回调通知外部ROI变化
    """

    HANDLE_SIZE = 8  # 手柄半径（像素）

    def __init__(self, x: float, y: float, w: float, h: float,
                 color: QColor, label: str, parent=None):
        super().__init__(x, y, w, h, parent)

        self._color = color
        self._label = label
        self._dragging_handle = HandlePosition.NONE
        self._drag_start_pos: QPointF | None = None
        self._drag_start_rect: QRectF | None = None
        self.geometry_changed = None  # 外部设置的回调: callable()

        # 外观
        pen = QPen(color, 2, Qt.DashLine)
        pen.setCosmetic(True)  # 线宽不随缩放变化
        self.setPen(pen)

        brush = QBrush(QColor(color.red(), color.green(), color.blue(), 30))
        self.setBrush(brush)

        # 允许移动
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        # 标签文本
        self._text_item = QGraphicsTextItem(label, self)
        self._text_item.setDefaultTextColor(color)
        self._text_item.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        self._update_text_position()

    @property
    def roi(self) -> Tuple[int, int, int, int]:
        """返回场景坐标ROI (x, y, w, h)。
        QGraphicsRectItem的rect是局部坐标，pos是场景偏移。
        两者相加才是实际场景坐标。"""
        r = self.rect()
        p = self.pos()
        return (int(p.x() + r.x()), int(p.y() + r.y()),
                int(r.width()), int(r.height()))

    def set_roi(self, x: int, y: int, w: int, h: int):
        """设置场景坐标ROI"""
        self.setPos(0, 0)
        self.setRect(QRectF(x, y, w, h))
        self._update_text_position()

    def mouseReleaseEvent(self, event):
        """拖拽结束：合并pos→rect + 清理状态"""
        if self._dragging_handle == HandlePosition.NONE:
            r = self.rect(); p = self.pos()
            self.setPos(0, 0)
            self.setRect(QRectF(p.x() + r.x(), p.y() + r.y(), r.width(), r.height()))
            self._update_text_position()
        self._dragging_handle = HandlePosition.NONE
        self._drag_start_pos = None
        self._drag_start_rect = None
        self.setCursor(Qt.OpenHandCursor)
        self._notify_change()
        super().mouseReleaseEvent(event)

    def _update_text_position(self):
        """更新标签位置（矩形左上角上方）"""
        r = self.rect()
        self._text_item.setPos(r.x(), r.y() - 20)

    # ---- 交互 ----

    def hoverMoveEvent(self, event):
        """鼠标悬停：改变光标形状"""
        handle = self._handle_at(event.pos())
        if handle == HandlePosition.TOP_LEFT or \
                handle == HandlePosition.BOTTOM_RIGHT:
            self.setCursor(Qt.SizeFDiagCursor)
        elif handle == HandlePosition.TOP_RIGHT or \
                handle == HandlePosition.BOTTOM_LEFT:
            self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.setCursor(Qt.OpenHandCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        """鼠标按下：检测是否拖拽手柄"""
        handle = self._handle_at(event.pos())
        self._dragging_handle = handle
        self._drag_start_pos = event.scenePos()
        self._drag_start_rect = QRectF(self.rect())

        if handle == HandlePosition.NONE:
            # 整体移动
            self.setCursor(Qt.ClosedHandCursor)
            super().mousePressEvent(event)
        else:
            # 手柄拖拽 — 不调用父类，自己处理
            event.accept()

    def mouseMoveEvent(self, event):
        """鼠标移动：处理手柄拖拽"""
        if self._dragging_handle == HandlePosition.NONE:
            super().mouseMoveEvent(event)
            return

        delta = event.scenePos() - self._drag_start_pos
        r = QRectF(self._drag_start_rect)
        min_size = 20  # 最小尺寸

        if self._dragging_handle == HandlePosition.TOP_LEFT:
            new_left = r.left() + delta.x()
            new_top = r.top() + delta.y()
            new_width = r.right() - new_left
            new_height = r.bottom() - new_top
            if new_width >= min_size:
                r.setLeft(new_left)
            if new_height >= min_size:
                r.setTop(new_top)

        elif self._dragging_handle == HandlePosition.TOP_RIGHT:
            new_top = r.top() + delta.y()
            new_width = r.width() + delta.x()
            new_height = r.bottom() - new_top
            if new_width >= min_size:
                r.setWidth(new_width)
            if new_height >= min_size:
                r.setTop(new_top)

        elif self._dragging_handle == HandlePosition.BOTTOM_LEFT:
            new_left = r.left() + delta.x()
            new_width = r.right() - new_left
            new_height = r.height() + delta.y()
            if new_width >= min_size:
                r.setLeft(new_left)
            if new_height >= min_size:
                r.setHeight(new_height)

        elif self._dragging_handle == HandlePosition.BOTTOM_RIGHT:
            new_width = r.width() + delta.x()
            new_height = r.height() + delta.y()
            if new_width >= min_size:
                r.setWidth(new_width)
            if new_height >= min_size:
                r.setHeight(new_height)

        self.setRect(r)
        self._update_text_position()
        event.accept()

    def itemChange(self, change, value):
        """项目变化时更新标签位置"""
        if change == QGraphicsItem.ItemPositionHasChanged:
            self._update_text_position()
            self._notify_change()
        return super().itemChange(change, value)

    def _notify_change(self):
        """通知外部ROI已变化"""
        if self.geometry_changed:
            self.geometry_changed()

    # ---- 辅助 ----

    def _handle_at(self, pos: QPointF) -> HandlePosition:
        """判断鼠标位置在哪个手柄上"""
        r = self.rect()
        hs = self.HANDLE_SIZE

        # 左上角
        if QRectF(r.topLeft() - QPointF(hs, hs),
                  r.topLeft() + QPointF(hs, hs)).contains(pos):
            return HandlePosition.TOP_LEFT
        # 右上角
        if QRectF(r.topRight() - QPointF(hs, hs),
                  r.topRight() + QPointF(hs, -hs)).contains(pos):
            return HandlePosition.TOP_RIGHT
        # 左下角
        if QRectF(r.bottomLeft() - QPointF(hs, -hs),
                  r.bottomLeft() + QPointF(hs, hs)).contains(pos):
            return HandlePosition.BOTTOM_LEFT
        # 右下角
        if QRectF(r.bottomRight() - QPointF(hs, hs),
                  r.bottomRight() + QPointF(hs, hs)).contains(pos):
            return HandlePosition.BOTTOM_RIGHT

        return HandlePosition.NONE

    def paint(self, painter: QPainter, option, widget):
        """绘制矩形和手柄"""
        super().paint(painter, option, widget)

        # 绘制四角手柄
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self._color))
        hs = self.HANDLE_SIZE

        r = self.rect()
        corners = [
            r.topLeft(), r.topRight(),
            r.bottomLeft(), r.bottomRight()
        ]
        for corner in corners:
            painter.drawRect(QRectF(
                corner.x() - hs, corner.y() - hs,
                hs * 2, hs * 2
            ))

        painter.restore()


class ROISelector(QObject):
    """
    ROI选择器管理器。

    管理两个SelectableRect：
    - 红色 = 被校时钟
    - 绿色 = 标准时钟

    信号:
        roi_changed: 任一ROI改变时发射
    """

    roi_changed = Signal()

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(parent)
        self._scene = scene

        # 默认ROI矩形位置（占位，用户需调整）
        default_w, default_h = 200, 60

        self.rect_calibrated = SelectableRect(
            100, 80, default_w, default_h,
            QColor(255, 82, 82),  # 红色
            "被校时钟"
        )
        self.rect_calibrated.geometry_changed = self._emit_change

        self.rect_standard = SelectableRect(
            400, 80, default_w, default_h,
            QColor(76, 175, 80),  # 绿色
            "标准时钟"
        )
        self.rect_standard.geometry_changed = self._emit_change

        # 初始隐藏，视频加载后显示
        self.rect_calibrated.setVisible(False)
        self.rect_standard.setVisible(False)

        self._scene.addItem(self.rect_calibrated)
        self._scene.addItem(self.rect_standard)

    def show(self):
        """显示ROI选择矩形"""
        self.rect_calibrated.setVisible(True)
        self.rect_standard.setVisible(True)

    def hide(self):
        """隐藏ROI选择矩形"""
        self.rect_calibrated.setVisible(False)
        self.rect_standard.setVisible(False)

    @property
    def roi_calibrated(self) -> Optional[Tuple[int, int, int, int]]:
        """被校时钟ROI"""
        if self.rect_calibrated.isVisible():
            return self.rect_calibrated.roi
        return None

    @property
    def roi_standard(self) -> Optional[Tuple[int, int, int, int]]:
        """标准时钟ROI"""
        if self.rect_standard.isVisible():
            return self.rect_standard.roi
        return None

    def _emit_change(self):
        """矩形变化时发射信号"""
        self.roi_changed.emit()

    def clear(self):
        """重置ROI到默认位置"""
        self.rect_calibrated.set_roi(100, 80, 200, 60)
        self.rect_standard.set_roi(400, 80, 200, 60)
        self.hide()
