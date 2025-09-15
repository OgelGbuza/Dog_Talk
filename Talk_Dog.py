# dog_avatar_tuned.py - Real-time Talking Avatar with Advanced Tuning 🐺
#
# Description:
# This version adds command-line arguments and an on-screen display
# to easily tune the animation for your specific microphone and environment.

import argparse
import logging
import sys
import time
from collections import deque

import cv2
import numpy as np
import sounddevice as sd

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Default Animation & Audio Parameters ---
TARGET_FPS = 30
EMA_ALPHA = 0.25
HEAD_BOB_DEG = 1.5
HEAD_BOB_FREQ = 6.0
AUDIO_BLOCK_SIZE = 1024
SAMPLE_RATE = 44100

# --- Helper Classes & Functions (Unchanged from before) ---

class EMA:
    """Exponential Moving Average filter."""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = 0.0
        self.is_initialized = False

    def update(self, new_value: float) -> float:
        if not self.is_initialized:
            self.value = new_value
            self.is_initialized = True
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

class Annotator:
    """Handles the user interaction for defining the hinge and jaw polygon."""
    def __init__(self, image: np.ndarray):
        self.image = image.copy()
        self.clone = image.copy()
        self.window_name = "Annotate Avatar"
        self.hinge_points = []
        self.jaw_poly_points = []
        self.stage = 0  # 0 for hinge, 1 for polygon

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.stage == 0 and len(self.hinge_points) < 2:
                self.hinge_points.append((x, y))
                logging.info(f"Added hinge point: {(x, y)}")
            elif self.stage == 1:
                self.jaw_poly_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.stage == 1:
            if self.jaw_poly_points:
                self.jaw_poly_points.pop()

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        info_stage0 = "Click TWO mouth hinge points (corners). Press ENTER to confirm."
        info_stage1 = "Outline the LOWER JAW (LMB add, RMB undo). Press ENTER to finish."

        while True:
            vis = self.clone.copy()
            info_text = info_stage0 if self.stage == 0 else info_stage1
            cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if self.stage == 0:
                for p in self.hinge_points: cv2.circle(vis, p, 5, (0, 255, 0), -1)
                if len(self.hinge_points) == 2:
                    cv2.line(vis, self.hinge_points[0], self.hinge_points[1], (0, 255, 0), 2)
            else:
                if self.jaw_poly_points:
                    cv2.polylines(vis, [np.array(self.jaw_poly_points)], isClosed=False, color=(0, 200, 255), thickness=2)
                for p in self.jaw_poly_points: cv2.circle(vis, p, 4, (0, 200, 255), -1)

            cv2.imshow(self.window_name, vis)
            key = cv2.waitKey(16) & 0xFF
            if key == 13:
                if self.stage == 0 and len(self.hinge_points) == 2: self.stage = 1
                elif self.stage == 1 and len(self.jaw_poly_points) >= 3: break
            elif key in (27, ord('q')): sys.exit(0)

        cv2.destroyWindow(self.window_name)
        return np.array(self.hinge_points, dtype=np.float32), np.array(self.jaw_poly_points, dtype=np.int32)

class AudioHandler:
    """Captures and provides smoothed audio amplitude."""
    def __init__(self, samplerate, blocksize):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.ema = EMA(alpha=EMA_ALPHA)
        self.amplitude = 0.0
        self.stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status: logging.warning(f"Audio stream status: {status}")
        mono_signal = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata
        rms_amplitude = np.sqrt(np.mean(mono_signal**2))
        self.amplitude = self.ema.update(rms_amplitude)

    def start(self, force_mic: bool = False): # <-- NEW force_mic argument
        use_loopback = not force_mic and sys.platform == "win32"
        
        if use_loopback:
            try:
                # ... (WASAPI loopback code from previous version)
                wasapi_settings = sd.WasapiSettings(loopback=True)
                device_info = sd.query_devices(kind='input')
                hostapi_info = sd.query_hostapis(device_info['hostapi'])
                default_speaker = hostapi_info['default_output_device']
                device_details = sd.query_devices(default_speaker)
                self.stream = sd.InputStream(
                    samplerate=device_details['default_samplerate'], blocksize=self.blocksize,
                    device=device_info['name'], channels=device_details['max_input_channels'],
                    dtype='float32', callback=self._audio_callback, extra_settings=wasapi_settings
                )
                logging.info(f"Audio: Capturing speaker output (WASAPI loopback).")
                self.stream.start()
                return
            except Exception as e:
                logging.warning(f"Could not start audio loopback ({e}). Falling back to microphone.")

        # Fallback to microphone
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate, blocksize=self.blocksize,
                channels=1, dtype='float32', callback=self._audio_callback
            )
            logging.info("Audio: Capturing microphone input.")
            self.stream.start()
        except Exception as mic_e:
            logging.error(f"Failed to open microphone: {mic_e}")
            sys.exit(1)

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

# --- Main Application ---
def main():
    # --- NEW: Command-line arguments for tuning ---
    parser = argparse.ArgumentParser(description="Real-time talking avatar from a photo.")
    parser.add_argument("image", help="Path to the input image file.")
    parser.add_argument("--width", type=int, default=800, help="Resize image to this width.")
    parser.add_argument("--mic", action="store_true", help="Force using the microphone instead of system audio.")
    parser.add_argument("--gain", type=float, default=1000.0, help="Audio sensitivity (gain). Higher is more sensitive.")
    parser.add_argument("--jaw_angle", type=float, default=22.0, help="Maximum jaw opening angle in degrees.")
    parser.add_argument("--no_head_bob", action="store_true", help="Disable the head bobbing effect.")
    args = parser.parse_args()

    # 1. Load and Prepare Image
    original_image = cv2.imread(args.image)
    if original_image is None:
        logging.error(f"Failed to load image from path: {args.image}")
        sys.exit(1)

    scale = args.width / original_image.shape[1]
    img = cv2.resize(original_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    base_image = img.copy()

    # 2. User Annotation
    annotator = Annotator(img)
    hinge_points, jaw_poly = annotator.run()
    hinge_center = np.mean(hinge_points, axis=0)

    # 3. Create Jaw Assets & Inpaint Base Image
    jaw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(jaw_mask, [jaw_poly], 255)
    base_image = cv2.inpaint(base_image, jaw_mask, 3, cv2.INPAINT_TELEA)

    x, y, w, h = cv2.boundingRect(jaw_poly)
    jaw_roi_original = img[y:y+h, x:x+w].copy()
    jaw_mask_roi = jaw_mask[y:y+h, x:x+w].copy()

    # 4. Start Audio Capture
    audio = AudioHandler(samplerate=SAMPLE_RATE, blocksize=AUDIO_BLOCK_SIZE)
    audio.start(force_mic=args.mic) # <-- Pass the new argument

    # 5. Main Animation Loop
    frame_period = 1.0 / TARGET_FPS
    last_frame_time = time.time()
    
    while True:
        frame = base_image.copy()
        amplitude = audio.amplitude
        
        # Normalize amplitude using the gain from arguments
        norm_amp = np.clip(amplitude * args.gain, 0.0, 1.0)
        
        # Calculate jaw rotation using the angle from arguments
        jaw_open_deg = norm_amp * args.jaw_angle
        
        jaw_center_local = np.mean(hinge_points - np.array([x, y]), axis=0)
        M = cv2.getRotationMatrix2D(tuple(jaw_center_local), jaw_open_deg, 1.0)

        warped_jaw = cv2.warpAffine(jaw_roi_original, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        warped_mask = cv2.warpAffine(jaw_mask_roi, M, (w, h), flags=cv2.INTER_NEAREST)

        sub_frame = frame[y:y+h, x:x+w]
        sub_frame[warped_mask > 0] = warped_jaw[warped_mask > 0]
        frame[y:y+h, x:x+w] = sub_frame

        if not args.no_head_bob:
            bob_angle = norm_amp * HEAD_BOB_DEG * np.sin(time.time() * HEAD_BOB_FREQ)
            R = cv2.getRotationMatrix2D(tuple(hinge_center), bob_angle, 1.0)
            frame = cv2.warpAffine(frame, R, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

        # --- NEW: Draw the on-screen volume meter ---
        bar_x, bar_y, bar_w, bar_h = 10, 10, 200, 20
        cv2.putText(frame, "Mic Level", (bar_x, bar_y + bar_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 0, 0), -1) # Black background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * norm_amp), bar_y + bar_h), (0, 255, 0), -1) # Green bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1) # White border

        cv2.imshow("Talking Avatar - Press 'Q' to Quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break

        elapsed = time.time() - last_frame_time
        if (sleep_time := frame_period - elapsed) > 0: time.sleep(sleep_time)
        last_frame_time = time.time()

    audio.stop()
    cv2.destroyAllWindows()
    logging.info("Program finished.")

if __name__ == "__main__":
    main()