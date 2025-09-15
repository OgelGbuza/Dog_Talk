# dog_avatar.py — Реальний-час «балакучий» аватар з фото 🐺
# ---------------------------------------------------------
# Як працює (узагальнено):
# 1) Завантажуємо зображення собаки.
# 2) Користувач мишкою задає вісь шарніра (2 точки) та полігон нижньої щелепи.
# 3) Захоплюємо аудіо з динаміків (WASAPI loopback) або мікрофона.
# 4) Обчислюємо RMS гучності, згладжуємо EMA → кут відкриття щелепи.
# 5) За кожен кадр обертаємо ROI щелепи навколо осі, легкий head-bob.
# 6) Відображаємо результат у вікні 30 FPS. Натисніть Q для виходу.

import argparse
import logging
from collections import deque
import numpy as np
import cv2
import sounddevice as sd
import sys
import time

# --------- Налаштування логів ---------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --------- Параметри за замовчуванням ---------
TARGET_FPS = 30
EMA_ALPHA = 0.2          # Сила згладжування амплітуди
JAW_MAX_DEG = 18.0       # Максимальний кут відкриття щелепи (градуси)
JAW_SCALE = 0.03         # Додатковий масштаб нижньої щелепи при відкритті
HEAD_BOB_DEG = 2.0       # Легкий нахил голови (градуси) від амплітуди
AUDIO_BLOCK_SIZE = 1024  # Розмір блоку аудіо (1024 семпли ≈ 23ms @44.1k)
SAMPLE_RATE = 44100

# --------- Допоміжні функції ---------
def rms(x: np.ndarray) -> float:
    """RMS гучності."""
    _x = x.astype(np.float32)
    return float(np.sqrt(np.mean(_x * _x)) + 1e-9)

def rotate_points(pts, center, angle_deg):
    """Повернути набір точок на кут довкола center."""
    ang = np.deg2rad(angle_deg)
    R = np.array([[np.cos(ang), -np.sin(ang)],
                  [np.sin(ang),  np.cos(ang)]], dtype=np.float32)
    v = (pts - center).astype(np.float32)
    return (v @ R.T) + center

class EMA:
    def __init__(self, alpha=0.2):
        self.a = alpha
        self.y = 0.0
        self.init = False
    def update(self, x):
        if not self.init:
            self.y = x
            self.init = True
        else:
            self.y = self.a * x + (1 - self.a) * self.y
        return self.y

# --------- Клік-інструменти для розмітки ---------
class Annotator:
    def __init__(self, img):
        self.img = img.copy()
        self.clone = img.copy()
        self.hinge = []     # 2 точки: ліва/права
        self.poly = []      # полігон нижньої щелепи
        self.stage = 0      # 0: hinge, 1: poly
        cv2.namedWindow("Annotate")
        cv2.setMouseCallback("Annotate", self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.stage == 0:
                if len(self.hinge) < 2:
                    self.hinge.append((x, y))
                    logging.info(f"Added hinge point: {(x,y)}")
            elif self.stage == 1:
                self.poly.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.stage == 1:
            if self.poly:
                self.poly.pop()

    def run(self):
        info0 = "Click TWO mouth-hinge points (corners). Press ENTER to confirm."
        info1 = "Click polygon for LOWER JAW (LMB add, RMB undo). Press ENTER to finish."
        while True:
            vis = self.clone.copy()
            if self.stage == 0:
                cv2.putText(vis, info0, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,220,20), 2)
                for p in self.hinge:
                    cv2.circle(vis, p, 5, (0,255,0), -1)
                if len(self.hinge) == 2:
                    cv2.line(vis, self.hinge[0], self.hinge[1], (0,255,0), 2)
            else:
                cv2.putText(vis, info1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,220,20), 2)
                for i,p in enumerate(self.poly):
                    cv2.circle(vis, p, 3, (0,200,255), -1)
                    if i>0: cv2.line(vis, self.poly[i-1], p, (0,200,255), 1)
            cv2.imshow("Annotate", vis)
            key = cv2.waitKey(16) & 0xFF
            if key == 13:  # ENTER
                if self.stage == 0 and len(self.hinge) == 2:
                    self.stage = 1
                elif self.stage == 1 and len(self.poly) >= 3:
                    break
            elif key in (27, ord('q'), ord('Q')):  # ESC/Q
                sys.exit(0)
        cv2.destroyWindow("Annotate")
        return np.array(self.hinge, dtype=np.float32), np.array(self.poly, dtype=np.int32)

# --------- Аудіо захоплення ---------
class AudioRMS:
    """Потік, що повертає згладжену амплітуду з динаміків (loopback) або мікрофона."""
    def __init__(self, samplerate=SAMPLE_RATE, blocksize=AUDIO_BLOCK_SIZE):
        self.sr = samplerate
        self.bs = blocksize
        self.ema = EMA(EMA_ALPHA)
        self.value = 0.0
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            logging.debug(f"Audio status: {status}")
        mono = np.mean(indata, axis=1)
        v = rms(mono)
        self.value = self.ema.update(v)

    def start(self):
        try:
            if sys.platform.startswith("win"):
                wasapi = sd.WasapiSettings(loopback=True)
                self.stream = sd.InputStream(
                    samplerate=self.sr, blocksize=self.bs,
                    channels=2, dtype='float32',
                    callback=self._callback, extra_settings=wasapi
                )
                logging.info("Audio: WASAPI loopback (speakers).")
            else:
                raise RuntimeError("Loopback only auto-configured on Windows in this script.")
        except Exception as e:
            logging.warning(f"Loopback unavailable, falling back to microphone. Reason: {e}")
            self.stream = sd.InputStream(samplerate=self.sr, blocksize=self.bs,
                                         channels=1, dtype='float32',
                                         callback=self._callback)
            logging.info("Audio: Microphone mode.")
        self.stream.start()

    def read_level(self) -> float:
        return self.value

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

# --------- Основна анімація ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to dog image (e.g., dog.jpg)")
    parser.add_argument("--width", type=int, default=900, help="Resize width for performance")
    parser.add_argument("--no_head_bob", action="store_true", help="Disable head bob")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        logging.error("Failed to read image.")
        sys.exit(1)

    scale = args.width / img.shape[1]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    base = img.copy()

    # Розмітка користувачем
    annot = Annotator(img)
    hinge_pts, jaw_poly = annot.run()
    hinge_center = np.mean(hinge_pts, axis=0)
    jaw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(jaw_mask, [jaw_poly], 255)

    # ROI щелепи
    x,y,w,h = cv2.boundingRect(jaw_poly)
    jaw_roi_orig = base[y:y+h, x:x+w].copy()
    jaw_mask_roi = jaw_mask[y:y+h, x:x+w].copy()
    jaw_local_poly = (jaw_poly - np.array([x,y])).astype(np.float32)
    hinge_local = hinge_pts - np.array([x,y], dtype=np.float32)

    # Аудіо
    audio = AudioRMS()
    audio.start()

    # Відтворення
    period = 1.0 / TARGET_FPS
    t0 = time.time()
    jaw_open_deg = 0.0
    while True:
        frame = base.copy()
        amp = audio.read_level()
        gain = 900.0
        a = np.clip(amp * gain, 0.0, 1.0)
        jaw_open_deg = (1-EMA_ALPHA)*jaw_open_deg + EMA_ALPHA*(a * JAW_MAX_DEG)

        # Побудова трансформу для щелепи
        jaw = jaw_roi_orig.copy()
        M = cv2.getRotationMatrix2D(tuple(hinge_local.mean(axis=0)), jaw_open_deg, 1.0 + a*JAW_SCALE)
        warped = cv2.warpAffine(jaw, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_w = cv2.warpAffine(jaw_mask_roi, M, (w, h), flags=cv2.INTER_NEAREST)

        sub = frame[y:y+h, x:x+w]
        sub[mask_w>0] = warped[mask_w>0]
        frame[y:y+h, x:x+w] = sub

        if not args.no_head_bob:
            hb = a * HEAD_BOB_DEG
            R = cv2.getRotationMatrix2D(tuple(hinge_center), hb*np.sin(time.time()*6), 1.0)
            frame = cv2.warpAffine(frame, R, (frame.shape[1], frame.shape[0]),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        cv2.imshow("Dog Avatar 🐺 — press Q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)
        t0 = time.time()

    audio.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Unhandled error: %s", e)
