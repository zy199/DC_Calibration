"""
后台工作线程模块
---------------
封装耗时操作到QThread中，通过信号与GUI层通信。
"""

from PySide6.QtCore import QThread, Signal

from core.video_io import VideoIO
from core.jump_detector import JumpDetector, JumpDetectionResult
from utils.logger import logger


class JumpDetectionWorker(QThread):
    """
    跳变检测后台线程（早停模式）。

    信号:
        progress(float): 进度 0.0~1.0
        finished(JumpDetectionResult): 检测结果
        error(str): 出错信息
    """

    progress = Signal(float)
    finished = Signal(object)  # JumpDetectionResult
    error = Signal(str)

    def __init__(self,
                 video_path: str,
                 roi_calibrated: tuple,
                 sensitivity: float = 1.0,
                 diff_threshold: int = 0,
                 parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.roi_calibrated = roi_calibrated
        self.sensitivity = sensitivity
        self.diff_threshold = diff_threshold
        self._cancelled = False
        self._start_frame: int = 0  # 外部可设置

    def run(self):
        """线程主函数"""
        logger.info("跳变检测线程启动（早停模式）")

        try:
            video = VideoIO(self.video_path)
            from config.settings import JumpDetectionSettings
            config = JumpDetectionSettings(
                sensitivity=self.sensitivity,
                diff_threshold=self.diff_threshold,
            )
            detector = JumpDetector(config=config)

            def frame_getter(idx: int):
                if self._cancelled:
                    raise InterruptedError("用户取消")
                return video.get_frame(idx)

            def progress_cb(ratio: float):
                if self._cancelled:
                    raise InterruptedError("用户取消")
                self.progress.emit(ratio)

            result = detector.detect(
                frame_getter=frame_getter,
                roi=self.roi_calibrated,
                frame_count=video.frame_count,
                fps=video.fps,
                start_frame=self._start_frame,
                progress_callback=progress_cb,
                video_path=self.video_path,  # 顺序读取，10倍加速
            )

            video.close()
            self.finished.emit(result)

            if result.success:
                logger.info(
                    f"跳变检测成功: Frame {result.jump_frame_idx} "
                    f"({result.timestamp:.3f}s), "
                    f"扫描{result.frames_scanned}帧, "
                    f"耗时{result.detection_time_ms:.0f}ms")
            else:
                logger.warning("跳变检测未找到跳变帧")

        except InterruptedError:
            logger.info("跳变检测被用户取消")
            self.error.emit("已取消")
        except Exception as e:
            logger.error(f"跳变检测失败: {e}")
            self.error.emit(str(e))

    def cancel(self):
        """取消检测"""
        self._cancelled = True
        logger.info("请求取消跳变检测")


class OCRWorker(QThread):
    """OCR识别后台线程"""
    progress = Signal(str)   # 状态文本
    finished = Signal(str, str)  # (cal_text, std_text)
    error = Signal(str)

    def __init__(self, frame, roi_cal, roi_std, parent=None):
        super().__init__(parent)
        self.frame = frame
        self.roi_cal = roi_cal
        self.roi_std = roi_std

    def run(self):
        try:
            self.progress.emit("正在加载OCR模型...")
            from core.ocr_engine import OCREngine
            engine = OCREngine()
            self.progress.emit("正在识别被校时钟...")
            cal_text, std_text = engine.recognize_two_clocks(
                self.frame, self.roi_cal, self.roi_std)
            self.finished.emit(cal_text, std_text)
        except Exception as e:
            self.error.emit(str(e))
