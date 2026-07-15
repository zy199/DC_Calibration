"""
跳变帧检测 — 颜色像素计数 + 亮度比双模
=====================================
1. 检测LED颜色(红/绿) → 颜色像素计数变化率
2. 找不到 → 亮度比法回退
"""
import cv2, time, numpy as np
from dataclasses import dataclass
from config.settings import settings, JumpDetectionSettings
from utils.logger import logger


@dataclass
class JumpDetectionResult:
    jump_frame_idx: int = -1
    diff_score: float = 0.0
    timestamp: float = 0.0
    frames_scanned: int = 0
    threshold: float = 0.0
    detection_time_ms: float = 0.0
    found: bool = False

    @property
    def success(self) -> bool:
        return self.found and self.jump_frame_idx >= 0


class JumpDetector:
    def __init__(self, config=None):
        self.config = config or settings.jump_detection

    def detect(self, frame_getter, roi, frame_count, fps,
               start_frame=0, progress_callback=None, video_path=''
               ) -> JumpDetectionResult:
        t0 = time.time()
        if not video_path: return JumpDetectionResult()
        # 首次尝试
        result = self._detect(video_path, roi, fps, t0, start_frame)
        if result.found: return result
        # 回退：降低阈值重试
        if start_frame > 0:
            logger.info("从start_frame未找到，降低阈值重试...")
            result = self._detect(video_path, roi, fps, t0, start_frame, lower_threshold=True)
        return result

    def _find_last_digit(self, f, roi):
        x,y,w,h=roi; g=cv2.cvtColor(f[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
        p98=np.percentile(g,98)
        if p98>30: _,b=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else: _,b=cv2.threshold(g,max(p98*0.5,10),255,cv2.THRESH_BINARY)
        contours,_=cv2.findContours(b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        blocks=[cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2]>=4 and cv2.boundingRect(c)[3]>=10]
        if len(blocks)<2: return None
        blocks.sort(key=lambda b:b[1])
        if len(blocks)>=4:
            gaps=[blocks[i+1][1]-(blocks[i][1]+blocks[i][3]) for i in range(len(blocks)-1)]
            if max(gaps)>10: blocks=blocks[gaps.index(max(gaps))+1:]
        blocks.sort(key=lambda b:b[0],reverse=True)
        last=blocks[0]; pad=2
        return (x+max(0,last[0]-pad),y+max(0,last[1]-pad),last[2]+2*pad,last[3]+2*pad)

    def _detect(self, vpath, roi, fps, t0, start_frame=0, lower_threshold=False):
        cap=cv2.VideoCapture(vpath); cap.set(cv2.CAP_PROP_POS_FRAMES, max(100,start_frame))
        ok,f=cap.read(); cap.release()
        if not ok: return JumpDetectionResult()

        x,y,w,h=roi; mid_y=y+h//2
        strip_w=int(w*0.30); strip_x=x+w-strip_w
        sx,sy,sw,sh=strip_x,mid_y,strip_w,y+h-mid_y

        # 检测LED颜色
        roi_sample=f[sy:sy+sh,sx:sx+sw]
        R=roi_sample[:,:,2].astype(float); G=roi_sample[:,:,1].astype(float)
        B=roi_sample[:,:,0].astype(float)
        is_red=np.sum((R>G*1.5)&(R>B*1.5)&(R>25))>np.sum((G>R*1.5)&(G>B*1.5)&(G>25))
        color_name='red' if is_red else 'green'
        logger.info(f"LED颜色: {color_name}, 区域: {sw}x{sh}px")

        # === 绿LED: 颜色像素计数法 ===
        if not is_red:
            result=self._scan_color(vpath,sx,sy,sw,sh,False,t0,start_frame,lower_threshold)
            if result.found: return result

        # === 红LED: 精确末位数字 + 亮度比法 ===
        digit_roi=self._find_last_digit(f,roi)
        if digit_roi:
            dx,dy,dw,dh=digit_roi
            logger.info(f"精确末位: {dw}x{dh}px")
            return self._scan_brightness(vpath,dx,dy,dw,dh,fps,t0,start_frame,lower_threshold)

        # 回退
        return self._scan_brightness(vpath,sx,sy,sw,sh,fps,t0,start_frame,lower_threshold)

    def _scan_color(self,vpath,sx,sy,sw,sh,is_red,t0,start_frame=0,lower_threshold=False):
        cap=cv2.VideoCapture(vpath)
        if start_frame>0: cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        prev_cnt=None; diffs=[]
        while True:
            ok,fm=cap.read()
            if not ok: break
            roi=fm[sy:sy+sh,sx:sx+sw]
            R=roi[:,:,2].astype(float); G=roi[:,:,1].astype(float); B=roi[:,:,0].astype(float)
            mask=(R>G*1.5)&(R>B*1.5)&(R>25) if is_red else (G>R*1.5)&(G>B*1.5)&(G>25)
            cnt=np.sum(mask)
            if prev_cnt is not None and prev_cnt>0:
                diffs.append(abs(cnt-prev_cnt)/prev_cnt)
            elif prev_cnt is not None: diffs.append(0.0)
            prev_cnt=cnt
        cap.release()
        return self._find_peak(np.array(diffs),'颜色计数',t0,start_frame,lower_threshold)

    def _scan_brightness(self,vpath,sx,sy,sw,sh,fps,t0,start_frame=0,lower_threshold=False):
        cap=cv2.VideoCapture(vpath)
        if start_frame>0: cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        bright=[]
        while True:
            ok,fm=cap.read()
            if not ok: break
            bright.append(float(np.mean(cv2.cvtColor(fm[sy:sy+sh,sx:sx+sw],cv2.COLOR_BGR2GRAY))))
        cap.release()
        bright=np.array(bright)

        # 全局CV中位数（稳定性参考）
        all_cvs=[]
        for i in range(10,len(bright)-10):
            win=bright[i-5:i+5]
            all_cvs.append(np.std(win)/np.mean(win) if np.mean(win)>0 else 0)
        global_cv=np.median(all_cvs) if all_cvs else 0.01

        w=3; scores=[]
        for i in range(w,len(bright)-w):
            before=np.mean(bright[i-w:i]); after=np.mean(bright[i:i+w])
            scores.append(abs(after-before)/max(before,after) if max(before,after)>0 else 0)
        scores=np.array(scores)

        p95=np.percentile(scores,95); p99=np.percentile(scores,99)
        gap=p99/p95 if p95>0 else 2.0
        threshold=p99*0.6 if gap>3 else p95*0.9
        if lower_threshold: threshold*=0.5  # 降低阈值重试
        if global_cv<0.005: threshold*=1.5
        elif global_cv<0.01: threshold*=1.2
        nr=max(15,int(fps*0.4))

        found=None
        is_find_next = (start_frame > 0)
        for i in range(len(scores)):
            if scores[i]>=threshold:
                fi=i+w+start_frame
                s_=max(0,i-nr); e=min(len(scores),i+nr+1)
                neighbors=np.concatenate([scores[s_:i],scores[i+1:e]])
                lm=np.median(neighbors)
                if len(neighbors)>0 and scores[i]>=lm*2.5:
                    # 找下一个模式：只检查局部峰值，不做稳定性（信号偏弱）
                    if is_find_next:
                        found=fi; break
                    # 首次检测：需要稳定性验证
                    rel_i = fi - start_frame
                    if rel_i >= 15 and rel_i + 15 < len(bright):
                        pre_start=rel_i-15; post_start=rel_i+3
                        post_end=min(len(bright),rel_i+15)
                        pre_mean=np.mean(bright[pre_start:rel_i])
                        pre_cv=np.std(bright[pre_start:rel_i])/pre_mean if pre_mean>0 else 999
                        post_mean=np.mean(bright[post_start:post_end])
                        post_cv=np.std(bright[post_start:post_end])/post_mean if post_mean>0 else 999
                        change=abs(post_mean-pre_mean)/pre_mean if pre_mean>0 else 0
                        if (post_cv<global_cv or post_cv<pre_cv*0.5) and change>0.003:
                            found=fi; break

        result=JumpDetectionResult()
        result.threshold=float(threshold)
        result.detection_time_ms=(time.time()-t0)*1000
        if found is not None:
            # 在±3帧内微调到信号最强帧（修正偏差，确保在5帧窗口内）
            best_fi,best_sc=found,0
            for df in range(-3,4):
                cf=found+df; ri=cf-start_frame-w
                if 0<=ri<len(scores) and scores[ri]>best_sc:
                    best_sc=scores[ri]; best_fi=cf
            result.found=True; result.jump_frame_idx=best_fi
            logger.info(f"亮度比: Frame {found}→{best_fi}, {result.detection_time_ms:.0f}ms")
        return result

    def _find_peak(self,diffs,method,t0,start_frame=0,lower_threshold=False):
        if len(diffs)<10: return JumpDetectionResult()
        nr=max(15,int(len(diffs)*0.01)); ratios=[]
        for i in range(len(diffs)):
            s=max(0,i-nr); e=min(len(diffs),i+nr+1)
            nn=np.concatenate([diffs[s:i],diffs[i+1:e]])
            lm=np.median(nn); ratios.append(diffs[i]/lm if lm>0 else 0)
        ratios=np.array(ratios)

        nonzero=ratios[ratios>0]
        if len(nonzero)<5: return JumpDetectionResult()
        p50=np.percentile(nonzero,50); p95=np.percentile(nonzero,95)
        mult=1.5 if lower_threshold else 2.5
        th=p50+(p95-p50)*mult

        found=None
        for i in range(3,len(ratios)):
            if ratios[i]>=th:
                s=max(0,i-nr); e=min(len(ratios),i+nr+1)
                nn=np.concatenate([ratios[s:i],ratios[i+1:e]])
                if len(nn)>0 and ratios[i]>=np.median(nn)*2:
                    found=i+1+start_frame; break

        result=JumpDetectionResult()
        result.threshold=float(th)
        result.detection_time_ms=(time.time()-t0)*1000
        if found is not None:
            result.found=True; result.jump_frame_idx=found
            logger.info(f"{method}跳变: Frame {found}, th={th:.1f}, {result.detection_time_ms:.0f}ms")
        return result
