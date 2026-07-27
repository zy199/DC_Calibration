"""
自动化测试脚本 - 8视频跳变帧 + OCR全面检测
==========================================
运行: python tests/test_all.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from utils.image_utils import detect_clock_regions
from core.jump_detector import JumpDetector
from core.ocr_engine import OCREngine

TARGETS = {
    '7.mp4':    6,     # 绿LED
    '271.mp4':  270,   # 红LED
    '383.mp4':  382,   # 红LED
    '443.mp4':  443,   # 红LED
    '585.mp4':  585,   # 红LED
    '864.mp4':  864,   # 红LED
    '1313.mp4': 1312,  # 红LED
    '1581.mp4': 1579,  # 红LED(暗)
}

def test_jump():
    print('='*60)
    print('跳变帧检测测试')
    print('='*60)
    ok = 0
    for vname, target in TARGETS.items():
        cap = cv2.VideoCapture(vname); ok_, f = cap.read()
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
        if not ok_: print(f'{vname}: 无法读取'); continue
        regions = detect_clock_regions(f, max_candidates=2)
        regions.sort(key=lambda r: r[1])
        cal = regions[0]
        d = JumpDetector()
        r = d.detect(None, cal, fc, fps, video_path=vname)
        diff = r.jump_frame_idx - target
        passed = abs(diff) <= 5
        if passed: ok += 1
        print(f'  {vname}: target={target}, got={r.jump_frame_idx}, diff={diff:+d} {"PASS" if passed else "FAIL"}')
    print(f'  Jump: {ok}/{len(TARGETS)} passed')
    return ok == len(TARGETS)

def test_ocr():
    print()
    print('='*60)
    print('OCR识别测试')
    print('='*60)
    engine = OCREngine()
    for vname in TARGETS:
        cap = cv2.VideoCapture(vname); ok_, f = cap.read(); cap.release()
        regions = detect_clock_regions(f, max_candidates=2)
        regions.sort(key=lambda r: r[1])
        cal_roi, std_roi = regions[0], regions[1]
        cap = cv2.VideoCapture(vname)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100); ok_, frame = cap.read(); cap.release()
        if not ok_: continue
        cal, std = engine.recognize_two_clocks(frame, cal_roi, std_roi)
        print(f'  {vname}:')
        print(f'    被校: "{cal}"')
        print(f'    标准: "{std}"')
    print('  (请人工检查OCR输出格式是否正确)')

if __name__ == '__main__':
    jump_ok = test_jump()
    test_ocr()
    print()
    print('ALL PASSED!' if jump_ok else 'JUMP DETECTION NOT ALL PASSED')
