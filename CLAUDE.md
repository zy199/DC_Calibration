# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

数字时钟自动校准系统——基于视频图像识别的时间频率计量校准工具。通过分析包含被校时钟和标准时钟的视频，自动找到时钟末位跳变帧，OCR识别两个时钟的时间，计算时间偏差并保存记录。

用户：贵州省计量测试院 科研开发部 张宇（高级工程师）

## 运行命令

```bash
cd "e:/时间频率实验室/A AI/DC_Calibration"
source venv/Scripts/activate
python main.py          # 启动GUI
```

依赖安装：
```bash
pip install PySide6 opencv-python numpy scikit-image Pillow easyocr pytesseract
```

## 架构

三层架构 + Qt信号驱动：

```
GUI层 (main.py + gui/*.py)     — PySide6, 4步QStackedWidget导航
  ↕ pyqtSignal / QThread
核心服务层 (core/*.py)          — 纯Python，不依赖Qt，可独立测试
  ↕ 函数调用
数据层 (data/*.py)              — SQLite + Repository模式
```

## 核心模块

### GUI流程（4步导航）

| 步骤 | 页面类 | 功能 |
|------|--------|------|
| 1/4 | `WelcomePage` | 加载视频、历史记录入口 |
| 2/4 | `ROISelectPage` | 框选被校(红)和标准(绿)时钟区域 |
| 3/4 | `DetectionConfirmPage` | 跳变检测+5帧清晰度浏览+确认 |
| 4/4 | `ResultPage` | OCR识别+偏差计算+设备信息+保存 |

### 核心算法

- **跳变检测** (`core/jump_detector.py`): 帧间差分+SSIM双指标，顺序读取早停模式，找到第一个末位进位帧立即返回
- **清晰度评估** (`core/clarity_evaluator.py`): Laplacian方差(50%)+过渡态检测(35%)+SSIM一致性(15%)，跳变帧加分+0.25，70%阈值强制选跳变帧
- **OCR识别** (`core/ocr_engine.py`): EasyOCR为主→七段规则法回退→自学习模板匹配，含互相纠错（用标准钟小时修正被校钟）、分行识别（日期+时间双行）、领域专用字符映射（/→1等）
- **时间偏差** (`utils/time_utils.py`): `TimeParseResult`含字段标记哪些时间分量是原文真实存在的，比对时自动只取共有分量

### 关键文件

- `main.py`: 程序入口，含4个页面类和MainWindow、全局样式表
- `gui/video_player.py`: QGraphicsView视频播放器（播放/暂停/逐帧/缩放）
- `gui/roi_selector.py`: ROI框选（双矩形拖拽+四角调整）
- `gui/worker_threads.py`: JumpDetectionWorker + OCRWorker（后台线程）
- `gui/record_manager.py`: 历史记录管理对话框（表格/搜索/导出CSV）
- `config/settings.py`: 全局配置dataclass，所有可调参数集中管理
- `data/database.py`: SQLite建表+迁移+备份恢复
- `data/repository.py`: CRUD+全字段搜索+CSV导出

### GUI交互细节

- 跳变检测页：首次进入自动检测，返回不重跑（缓存），`_confirmed`标志控制确认状态
- 更换视频/重新寻找/找下一个跳变 → 自动重置确认状态
- OCR在后台线程运行，进入结果页显示"⏳ 识别中..."不卡UI
- ROI选择：首次进入自动检测时钟区域并预选框选（`_auto_detected`标志防重复）
- 保存记录后显示"开始新的校准"按钮 → 重置全部状态回首页

### EasyOCR注意事项

- 首次加载需下载模型（~100MB），程序启动时后台线程预加载
- CPU运行较慢（每次OCR约5-10秒），GPU可显著加速
- 对该字体存在已知混淆：'7'→'1'、':'→'.'、'0'→'8'，通过`_normalize`和`_mutual_correct`缓解
- 领域专用纠错：'/' → '1'（时钟显示永远不会有'/'字符）
