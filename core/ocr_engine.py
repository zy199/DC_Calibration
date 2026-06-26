"""
OCR引擎 — 离线数字时钟识别
EasyOCR + 互相纠错 + 自学习模板
"""
import re, cv2, numpy as np
from typing import Tuple, Dict
from core.seven_segment_ocr import SevenSegmentOCR
from utils.logger import logger

DIGITS = set('0123456789')
VALID_CHARS = DIGITS | set(':. /时分秒年月日-')


class OCREngine:
    def __init__(self):
        self._seven_seg = SevenSegmentOCR()
        self._easyocr = None
        self._templates: Dict[str, np.ndarray] = {}

    def recognize(self, roi: np.ndarray) -> str:
        if roi.size == 0: return ""
        # EasyOCR → 模板 → 七段
        for method in [self._recognize_easyocr,
                       self._recognize_templates,
                       lambda r: self._seven_seg.read_display(r)]:
            text = method(roi)
            text = self._clean(text)
            if self._plausible(text):
                return text
        return ""

    def recognize_two_clocks(self, frame, roi_cal, roi_std) -> Tuple[str, str]:
        x1,y1,w1,h1 = roi_cal
        x2,y2,w2,h2 = roi_std
        cal = self.recognize(frame[y1:y1+h1, x1:x1+w1])
        # 标准时钟可能有两行（日期+时间），分行识别
        std = self._recognize_two_line(frame[y2:y2+h2, x2:x2+w2])
        cal, std = self._mutual_correct(cal, std)

        # 如果被校只有时间，标准钟去掉日期部分
        cal_p = self._parse_time(cal)
        std_p = self._parse_time(std)
        if cal_p and std_p and not cal_p.get('has_date') and std_p.get('has_date'):
            # 只保留标准钟的时间部分
            std = std_p.get('time_str', std)

        logger.info(f"OCR: 被校='{cal}', 标准='{std}'")
        return cal, std

    def _recognize_two_line(self, roi: np.ndarray) -> str:
        """识别可能有两行（日期+时间）的标准时钟"""
        h = roi.shape[0]
        # 如果高度>100px，可能是两行，拆开识别
        if h > 100:
            mid = h // 2
            # 在上半部分和下半部分之间找最佳分割点（最暗的行）
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            row_sums = np.sum(gray, axis=1)
            # 在中间1/3区域找最暗行作为分割线
            search_start, search_end = h // 3, 2 * h // 3
            split = search_start + np.argmin(row_sums[search_start:search_end])
            top_text = self.recognize(roi[:split, :])
            bot_text = self.recognize(roi[split:, :])
            # 判断哪行是日期哪行是时间
            date_text, time_text = self._classify_lines(top_text, bot_text)
            if date_text and time_text:
                return f"{date_text} {time_text}"
            if time_text:
                return time_text
        # 单行或回退
        return self.recognize(roi)

    def _classify_lines(self, t1: str, t2: str) -> Tuple[str, str]:
        """判断两行哪行是日期、哪行是时间"""
        import re
        def has_4digit(t):
            nums = re.findall(r'\d+', t)
            return any(len(n)==4 for n in nums)
        def has_time(t):
            return bool(re.search(r'\d{1,2}[:\s]\d{2}', t))
        def num_count(t):
            return len(re.findall(r'\d+', t))

        h1, h2 = has_4digit(t1), has_4digit(t2)
        t1_time, t2_time = has_time(t1), has_time(t2)

        # 有4位数字的是日期行
        if h1 and t2_time: return t1, t2
        if h2 and t1_time: return t2, t1
        # 数字更多的那行可能是日期（年月日=3组）+ 时间在另一行
        if num_count(t1) >= 3 and t2_time: return t1, t2
        if num_count(t2) >= 3 and t1_time: return t2, t1
        # 都没识别出日期格式，返回空让调用方处理
        return ('', '')

    def _parse_time(self, text: str) -> dict:
        """解析时间文本，检测年月日"""
        import re
        if not text: return {}
        nums = re.findall(r'\d+', text)
        if len(nums) < 2: return {}
        result = {'text': text, 'has_date': False, 'time_str': text}
        # 检查4位年份（含常见OCR错误如2822→2022）
        for n in nums:
            if len(n) == 4:
                # 接受以20/19开头的，也接受28xx→20xx的OCR错误
                if n.startswith(('20','19')) or (n.startswith('28') and n[2:].isdigit()):
                    result['has_date'] = True
                    break
        if result['has_date']:
            # 取最后一个时间匹配（日期行在前，时间行在后）
            matches = list(re.finditer(r'(\d{1,2}[:时分]\d{1,2}[:时分]\d{1,2}(?:[\.:]\d{1,3})?)', text))
            if matches:
                result['time_str'] = matches[-1].group(1)
        return result

    def learn_from_text(self, roi: np.ndarray, text: str):
        if not roi.size or not text: return
        binary = self._prep(roi)
        contours = self._contours(binary)
        if len(contours) < 3: return
        digits = [c for c in text if c in DIGITS]
        if len(digits) < len(contours): return
        contours_sorted = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        for i, cnt in enumerate(contours_sorted[:len(digits)]):
            x, y, cw, ch = cv2.boundingRect(cnt)
            img = cv2.resize(binary[y:y+ch, x:x+cw], (20,30), interpolation=cv2.INTER_AREA)
            self._templates[digits[i]] = img
        logger.info(f"学习了{len(self._templates)}个模板")

    # ── 内部 ──

    def _clean(self, t: str) -> str:
        if not t: return ""
        # 领域专用纠错：只可能出现的字符是 0-9 : . - 空格 时分秒年月日
        # 常见OCR混淆 → 强制修正
        t = t.replace('/','1')   # '/' 极像 '1'
        t = t.replace('\\','1')
        t = t.replace('(','1').replace(')','1')
        t = t.replace('[','1').replace(']','1')
        t = t.replace('|','1')
        t = ''.join(c for c in t if c in VALID_CHARS)
        t = t.replace('时',':').replace('分',':').replace('秒','')
        t = t.replace('年','-').replace('月','-').replace('日','').replace('/','-')
        t = re.sub(r'[:.\- ]{2,}', lambda m: m.group()[0], t).strip(' .:-')
        t = re.sub(r'\.$', '', t)
        return self._normalize(t)

    def _normalize(self, t: str) -> str:
        t = t.replace('O','0').replace('o','0').replace('S','5').replace('B','8')
        nums = re.findall(r'\d+', t)
        if not nums: return t
        parts = []
        yi = -1
        for i,n in enumerate(nums):
            # 接受20xx/19xx，也接受28xx→20xx的OCR错误
            if len(n)==4 and (n.startswith(('20','19')) or
               (n.startswith('28') and n[2:].isdigit())):
                yi = i; break
        if yi>=0 and len(nums)>=yi+3:
            y,mo,d = nums[yi], nums[yi+1], nums[yi+2]
            parts.append(f'{y}-{int(mo):02d}-{int(d):02d}')
            tn = nums[yi+3:]
        else:
            tn = nums
        if len(tn)>=2:
            h = tn[0]
            # 如果只有2组数字
            if len(tn)==2:
                # 如果第一个数字>2位 → SSmmm + 尾数
                if len(tn[0]) > 2:
                    s_str = tn[0]; extra = tn[1]
                    h = '00'
                    if len(s_str)>=3: s = int(s_str[:2]); ms = s_str[2:]
                    else: s = int(s_str); ms = ''
                    if len(extra)==1: ms += extra
                    mi = '00'
                else:
                    merged = tn[1]
                    if len(merged)>=5:
                        mi = merged[:2]; s = int(merged[2:4]); ms = merged[4:]
                    elif len(merged)>=3:
                        mi = '00'; s = int(merged[:2]); ms = merged[2:]
                    else:
                        mi = '00'; s = int(merged); ms = ''
            else:
                mi = tn[1]; s_str = tn[2]
                # 3组数字: H, M, SSmmm → s_str是秒+毫秒
                if len(s_str)>=3: s = int(s_str[:2]); ms = s_str[2:]
                else: s = int(s_str); ms = ''
            ts = f'{int(h):02d}:{int(mi):02d}:{s:02d}'
            if ms:
                # 如果后面还有单数字，追加到毫秒（OCR拆分修复）
                if len(tn) >= 4 and len(tn[3]) == 1:
                    ms += tn[3]
                ts += f'.{ms}'
            elif len(tn) >= 4 and len(tn[3]) <= 3:
                ts += f'.{tn[3]}'
            parts.append(ts)
            return ' '.join(parts)
        m = re.search(r'(\d{1,2})[:时分](\d{1,2})[:时分](\d{1,2})(?:[\.:](\d{1,3}))?', t)
        if m:
            h,mi,s = m.group(1),m.group(2),m.group(3)
            ms = m.group(4)
            r = f'{int(h):02d}:{int(mi):02d}:{int(s):02d}'
            if ms: r += f'.{ms}'; return r
        return t

    def _plausible(self, t: str) -> bool:
        if not t or len(t)<5: return False
        d = [c for c in t if c in DIGITS]
        return not (len(d)>=4 and len(set(d))<=2)

    def _mutual_correct(self, cal, std):
        """用标准钟时间部分的小时纠正被校钟OCR错误"""
        if not cal or not std: return cal, std
        cn = re.findall(r'\d+', cal)
        if len(cn)<1: return cal, std

        # 先提取标准钟的纯时间部分（去掉日期前缀）
        std_p = self._parse_time(std)
        time_str = std_p.get('time_str', std)

        # 在时间部分找 HH:MM:SS
        tm = re.search(r'(?<!\d)(\d{1,2})[:时分](\d{1,2})[:时分](\d{1,2})', time_str)
        if tm:
            try:
                sh = int(tm.group(1))
                ch = int(cn[0])
                if 1 <= abs(ch-sh) < 12:
                    cal = cal.replace(cn[0], str(sh), 1)
            except: pass
        return cal, std

    def _recognize_easyocr(self, roi):
        try:
            if self._easyocr is None:
                import easyocr; self._easyocr = easyocr.Reader(['en'], gpu=False)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi
            # 亮度拉伸
            p2,p98 = np.percentile(gray, (2,98))
            if p98>p2+10: gray = np.clip((gray.astype(float)-p2)*255/(p98-p2),0,255).astype(np.uint8)
            r = self._easyocr.readtext(gray, detail=0, allowlist='0123456789:-. /')
            return ' '.join(r)
        except: return ""

    def _recognize_templates(self, roi):
        if len(self._templates)<5: return ""
        binary = self._prep(roi)
        contours = self._contours(binary)
        if not contours: return ""
        cs = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        result, last_x2 = [], -50
        for cnt in cs:
            x,y,cw,ch = cv2.boundingRect(cnt)
            if cw<6 or ch<12: continue
            if x-last_x2 < -cw*0.3: continue
            aspect = ch/cw if cw>0 else 99
            if aspect>3.5: result.append(':'); continue
            if ch<20 and cw<15: result.append('.'); continue
            img = cv2.resize(binary[y:y+ch,x:x+cw], (20,30), interpolation=cv2.INTER_AREA)
            best_d, best_s = '?', 0
            for d,tmpl in self._templates.items():
                if tmpl.shape!=img.shape: continue
                s = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)[0][0]
                if s>best_s: best_s,best_d = s,d
            if best_s>0.4: result.append(best_d); last_x2=x+cw
        return ''.join(result)

    def _prep(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi.copy()
        p2,p98 = np.percentile(gray, (2,98))
        if p98>p2+10: gray = np.clip((gray.astype(float)-p2)*255/(p98-p2),0,255).astype(np.uint8)
        e = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
        d = cv2.bilateralFilter(e, 5, 75, 75)
        _, b = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if np.sum(b>0)/b.size>0.6: b = cv2.bitwise_not(b)
        return b

    def _contours(self, b):
        d = cv2.dilate(b, np.ones((2,2),np.uint8), iterations=1)
        cs, _ = cv2.findContours(d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in cs if cv2.boundingRect(c)[2]>5 and cv2.boundingRect(c)[3]>10]
