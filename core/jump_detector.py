"""
跳变帧检测 — 折半查找 + 颜色/亮度 + 追踪窗确认
=============================================
1. 统一管线: 精确定位末位数字 → 追踪窗提取 → 折半查找(O log n)
2. 红绿LED均享受完整管线, 折半查找失败后按颜色分流回退
3. 绿LED: 颜色像素计数法回退 / 红LED: 亮度比法回退
"""
import cv2, time, numpy as np
from dataclasses import dataclass
from config.settings import settings
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
        result = self._detect(video_path, roi, fps, t0, start_frame)
        if result.found: return result
        if start_frame > 0:
            logger.info("从start_frame未找到，降低阈值重试...")
            result = self._detect(video_path, roi, fps, t0, start_frame, lower_threshold=True)
        return result

    # ====== 末位数字定位 ======
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

    # ====== 追踪窗（亮段区域） ======
    def _extract_tracking_windows(self, f, droi):
        dx,dy,dw,dh=droi
        gray=cv2.cvtColor(f[dy:dy+dh,dx:dx+dw],cv2.COLOR_BGR2GRAY)
        p98=np.percentile(gray,98)
        if p98>30: _,b=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else: _,b=cv2.threshold(gray,max(p98*0.5,10),255,cv2.THRESH_BINARY)
        contours,_=cv2.findContours(b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        windows=[]
        for cnt in contours:
            bx,by,bw,bh=cv2.boundingRect(cnt)
            if bw>=3 and bh>=5: windows.append((dx+bx,dy+by,bw,bh))
        return windows

    def _check_tracking_windows(self, vpath, windows, frame_idx):
        try:
            cap=cv2.VideoCapture(vpath)
            cap.set(cv2.CAP_PROP_POS_FRAMES,max(2,frame_idx-3)); _,pre=cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES,min(frame_idx+2,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)); _,post=cap.read()
            cap.release()
            if pre is None or post is None: return False
            for (wx,wy,ww,wh) in windows:
                pre_b=np.mean(cv2.cvtColor(pre[wy:wy+wh,wx:wx+ww],cv2.COLOR_BGR2GRAY))
                post_b=np.mean(cv2.cvtColor(post[wy:wy+wh,wx:wx+ww],cv2.COLOR_BGR2GRAY))
                if pre_b>15 and (pre_b-post_b)>8 and post_b<pre_b*0.5: return True
        except: pass
        return False

    # ====== 折半查找 O(log n) ======
    def _binary_search(self, vpath, dx, dy, dw, dh, start_frame, fc, t0):
        try:
            from skimage.metrics import structural_similarity as ssim
            cap=cv2.VideoCapture(vpath)
            cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame); _,base_f=cap.read(); cap.release()
            if base_f is None: return JumpDetectionResult()
            base_g=cv2.cvtColor(base_f[dy:dy+dh,dx:dx+dw],cv2.COLOR_BGR2GRAY)
            def changed(fi):
                cap2=cv2.VideoCapture(vpath); cap2.set(cv2.CAP_PROP_POS_FRAMES,fi)
                _,fm=cap2.read(); cap2.release()
                if fm is None: return True
                return ssim(base_g,cv2.cvtColor(fm[dy:dy+dh,dx:dx+dw],cv2.COLOR_BGR2GRAY),data_range=255)<0.92
            lo,hi=max(start_frame+10,start_frame),fc-1
            while lo<hi:
                mid=(lo+hi)//2
                if changed(mid): hi=mid
                else: lo=mid+1
            for fi in range(max(start_frame+5,lo-3),min(fc-1,lo+4)):
                if changed(fi):
                    result=JumpDetectionResult(); result.found=True
                    result.jump_frame_idx=fi
                    result.detection_time_ms=(time.time()-t0)*1000
                    logger.info(f"折半查找: Frame {fi}, {result.detection_time_ms:.0f}ms")
                    return result
        except Exception as e: logger.warning(f"折半查找失败: {e}")
        return JumpDetectionResult()

    # ====== 主检测 ======
    def _detect(self, vpath, roi, fps, t0, start_frame=0, lower_threshold=False):
        cap=cv2.VideoCapture(vpath); cap.set(cv2.CAP_PROP_POS_FRAMES,max(100,start_frame))
        ok,f=cap.read(); cap.release()
        if not ok: return JumpDetectionResult()

        x,y,w,h=roi; mid_y=y+h//2; strip_w=int(w*0.30); strip_x=x+w-strip_w
        sx,sy,sw,sh=strip_x,mid_y,strip_w,y+h-mid_y

        roi_sample=f[sy:sy+sh,sx:sx+sw]
        R=roi_sample[:,:,2].astype(float); G=roi_sample[:,:,1].astype(float); B=roi_sample[:,:,0].astype(float)
        is_red=np.sum((R>G*1.5)&(R>B*1.5)&(R>25))>np.sum((G>R*1.5)&(G>B*1.5)&(G>25))
        logger.info(f"LED颜色: {'红' if is_red else '绿'}, 区域: {sw}x{sh}px")

        # ===== 统一管线：红绿LED均享受完整管线 =====
        cap=cv2.VideoCapture(vpath); fc=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        track_wins=[]

        # 1. 精确末位数字定位 (灰度图处理, 红绿通用)
        digit_roi=self._find_last_digit(f,roi)

        if digit_roi:
            dx,dy,dw,dh=digit_roi
            # 2. 提取追踪窗 (灰度图处理, 红绿通用)
            track_wins=self._extract_tracking_windows(f,digit_roi)
            logger.info(f"精确末位: {dw}x{dh}px, 追踪窗: {len(track_wins)}个")

            # 3. 折半查找 O(log n) (SSIM灰度比较, 红绿通用)
            result=self._binary_search(vpath,dx,dy,dw,dh,start_frame,fc,t0)
            if result.found: return result

        # 4. 折半查找失败 → 按颜色分流回退
        if is_red:
            if digit_roi:
                dx,dy,dw,dh=digit_roi
                return self._scan_brightness(vpath,dx,dy,dw,dh,fps,t0,start_frame,lower_threshold,track_wins)
            else:
                return self._scan_brightness(vpath,sx,sy,sw,sh,fps,t0,start_frame,lower_threshold,[])
        else:
            return self._scan_color(vpath,sx,sy,sw,sh,False,t0,start_frame,lower_threshold)

    # ====== 颜色法（绿LED） ======
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
            if prev_cnt is not None and prev_cnt>0: diffs.append(abs(cnt-prev_cnt)/prev_cnt)
            elif prev_cnt is not None: diffs.append(0.0)
            prev_cnt=cnt
        cap.release()
        return self._find_peak(np.array(diffs),'颜色计数',t0,start_frame,lower_threshold)

    # ====== 亮度比法 ======
    def _scan_brightness(self,vpath,sx,sy,sw,sh,fps,t0,start_frame=0,lower_threshold=False,track_wins=None):
        if track_wins is None: track_wins=[]
        cap=cv2.VideoCapture(vpath)
        if start_frame>0: cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        bright=[]
        while True:
            ok,fm=cap.read()
            if not ok: break
            bright.append(float(np.mean(cv2.cvtColor(fm[sy:sy+sh,sx:sx+sw],cv2.COLOR_BGR2GRAY))))
        cap.release()
        bright=np.array(bright)

        all_cvs=[]
        for i in range(10,len(bright)-10):
            win=bright[i-5:i+5]; all_cvs.append(np.std(win)/np.mean(win) if np.mean(win)>0 else 0)
        global_cv=np.median(all_cvs) if all_cvs else 0.01

        w=3; scores=[]
        for i in range(w,len(bright)-w):
            before=np.mean(bright[i-w:i]); after=np.mean(bright[i:i+w])
            scores.append(abs(after-before)/max(before,after) if max(before,after)>0 else 0)
        scores=np.array(scores)

        p95=np.percentile(scores,95); p99=np.percentile(scores,99)
        gap=p99/p95 if p95>0 else 2.0
        threshold=p99*0.6 if gap>3 else p95*0.9
        if lower_threshold: threshold*=0.5
        if global_cv<0.005: threshold*=1.5
        elif global_cv<0.01: threshold*=1.2
        nr=max(15,int(fps*0.4))

        found=None; is_find_next=(start_frame>0)
        for i in range(len(scores)):
            if scores[i]>=threshold:
                fi=i+w+start_frame
                s_=max(0,i-nr); e=min(len(scores),i+nr+1)
                neighbors=np.concatenate([scores[s_:i],scores[i+1:e]])
                lm=np.median(neighbors)
                if len(neighbors)>0 and scores[i]>=lm*2.5:
                    accepted=False
                    if track_wins:
                        if self._check_tracking_windows(vpath,track_wins,fi): accepted=True
                    elif is_find_next: accepted=True
                    if not accepted and not is_find_next:
                        rel_i=fi-start_frame
                        if rel_i>=15 and rel_i+15<len(bright):
                            pre_start=rel_i-15; post_start=rel_i+3; post_end=min(len(bright),rel_i+15)
                            pre_mean=np.mean(bright[pre_start:rel_i])
                            pre_cv=np.std(bright[pre_start:rel_i])/pre_mean if pre_mean>0 else 999
                            post_mean=np.mean(bright[post_start:post_end])
                            post_cv=np.std(bright[post_start:post_end])/post_mean if post_mean>0 else 999
                            change=abs(post_mean-pre_mean)/pre_mean if pre_mean>0 else 0
                            if (post_cv<global_cv or post_cv<pre_cv*0.5) and change>0.003: accepted=True
                    if accepted: found=fi; break

        result=JumpDetectionResult(); result.threshold=float(threshold); result.detection_time_ms=(time.time()-t0)*1000
        if found is not None:
            best_fi,best_sc=found,0
            for df in range(-3,4):
                cf=found+df; ri=cf-start_frame-w
                if 0<=ri<len(scores) and scores[ri]>best_sc: best_sc=scores[ri]; best_fi=cf
            result.found=True; result.jump_frame_idx=best_fi
            logger.info(f"亮度比: Frame {found}→{best_fi}, {result.detection_time_ms:.0f}ms")
        return result

    # ====== 峰值查找 ======
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
        for i in range(3,len(ratios)):
            if ratios[i]>=th:
                s=max(0,i-nr); e=min(len(ratios),i+nr+1)
                nn=np.concatenate([ratios[s:i],ratios[i+1:e]])
                if len(nn)>0 and ratios[i]>=np.median(nn)*2:
                    result=JumpDetectionResult(); result.found=True
                    result.jump_frame_idx=i+1+start_frame
                    result.threshold=float(th); result.detection_time_ms=(time.time()-t0)*1000
                    logger.info(f"{method}跳变: Frame {result.jump_frame_idx}, th={th:.1f}, {result.detection_time_ms:.0f}ms")
                    return result
        return JumpDetectionResult()
