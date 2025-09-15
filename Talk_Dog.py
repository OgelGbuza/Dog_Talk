# talk_dog_dynamic_range.py - The most expressive Talking Avatar yet! 🎭
#
# Description:
# This version introduces a crucial animation concept: dynamic range. Instead of
# only opening the mouth from the source image's position, it can now both CLOSE
# and OPEN it. Two new command-line arguments, --close_amount and --open_amount,
# allow the user to define a "neutral" position and animate in both directions,
# creating a far more natural and expressive result, especially for images
# where the subject's mouth is already open.

import argparse
import logging
import sys
import time

import cv2
import numpy as np
import sounddevice as sd

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Default Parameters ---
TARGET_FPS = 60
EMA_ALPHA = 0.25
JAW_GRAVITY = 0.85
AUDIO_BLOCK_SIZE = 1024
SAMPLE_RATE = 44100

class PointSelector:
    """A helper class to draw a polygon on an image."""
    def __init__(self, window_name: str, image: np.ndarray):
        self.window_name = window_name
        self.image_clone = image.copy()
        self.original_image = image.copy()
        self.points = []
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.image_clone, (x, y), 5, (0, 255, 0), -1)
            if len(self.points) > 1:
                cv2.line(self.image_clone, self.points[-2], self.points[-1], (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.image_clone)

    def get_points(self) -> np.ndarray:
        print("Click to outline the polygon. Press 'd' or 'Enter' when done, 'r' to reset.")
        while True:
            cv2.imshow(self.window_name, self.image_clone)
            key = cv2.waitKey(20) & 0xFF
            if (key == ord("d") or key == 13) and len(self.points) >= 3: break
            elif key == ord("r"):
                self.points = []
                self.image_clone = self.original_image.copy()
                logging.info("Points reset.")
            elif key == 27: logging.info("Selection cancelled."); sys.exit(0)
        cv2.destroyWindow(self.window_name)
        return np.array(self.points, dtype=np.int32)

class AudioProcessor:
    """Handles audio input and smoothing."""
    def __init__(self, block_size, alpha):
        self.ema = EMA(alpha); self.amplitude = 0.0
        try:
            self.stream = sd.InputStream(callback=self._audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=block_size, dtype="float32")
        except Exception as e:
            logging.error(f"Failed to open audio stream: {e}. Is a microphone connected?")
            sys.exit(1)
    def _audio_callback(self, indata, frames, time, status):
        if status: logging.warning(status)
        amplitude = np.sqrt(np.mean(indata**2)); self.amplitude = self.ema.update(amplitude)
    def start(self): self.stream.start(); logging.info("Audio stream started.")
    def stop(self): self.stream.stop(); self.stream.close(); logging.info("Audio stream closed.")

class EMA:
    """Exponential Moving Average filter."""
    def __init__(self, alpha: float): self.alpha = alpha; self.value = 0.0; self.is_initialized = False
    def update(self, new_value: float) -> float:
        if not self.is_initialized: self.value = new_value; self.is_initialized = True
        else: self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

def get_jaw_quad(jaw_points: np.ndarray) -> np.ndarray:
    """Intelligently finds the 4 corners of the jaw for perspective warping."""
    if len(jaw_points) < 4: raise ValueError("Jaw selection must have at least 4 points.")
    
    sorted_y = jaw_points[jaw_points[:, 1].argsort()]
    top_points = sorted_y[:2]
    bottom_points = sorted_y[-2:]

    top_left = top_points[top_points[:, 0].argsort()][0]
    top_right = top_points[top_points[:, 0].argsort()][1]
    bottom_left = bottom_points[bottom_points[:, 0].argsort()][0]
    bottom_right = bottom_points[bottom_points[:, 0].argsort()][1]
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def main(args):
    try:
        original_frame = cv2.imread(args.image_path)
        if original_frame is None: raise FileNotFoundError(f"Image not found at {args.image_path}")
        rows, cols, _ = original_frame.shape

        win_jaw = "Step 1: Outline the entire LOWER JAW area"
        jaw_selector = PointSelector(win_jaw, original_frame.copy())
        jaw_points = jaw_selector.get_points()
        src_quad = get_jaw_quad(jaw_points)

        win_fg = "Step 2: Outline the FOREGROUND/UPPER LIP"
        fg_selector = PointSelector(win_fg, original_frame.copy())
        foreground_points = fg_selector.get_points()
        foreground_mask = np.zeros((rows, cols), dtype=np.uint8); cv2.fillPoly(foreground_mask, [foreground_points], 255)
        
        logging.info("Inpainting background...")
        jaw_mask_for_inpaint = np.zeros((rows, cols), dtype=np.uint8); cv2.fillPoly(jaw_mask_for_inpaint, [jaw_points], 255)
        background = cv2.inpaint(original_frame, jaw_mask_for_inpaint, 3, cv2.INPAINT_TELEA)
        logging.info("Inpainting complete.")

        audio_processor = AudioProcessor(AUDIO_BLOCK_SIZE, EMA_ALPHA)
        audio_processor.start()
        norm_amp = 0.0
        
        # --- NEW: Calculate the full dynamic range of movement ---
        movement_range = (args.open_amount + args.close_amount) * args.max_movement
        
        while True:
            raw_amp = audio_processor.amplitude
            amp_val = max(0, min(1, (raw_amp - args.threshold) * args.multiplier))
            norm_amp = max(norm_amp * JAW_GRAVITY, amp_val)
            
            # --- NEW: Map the 0-1 amplitude to a closing/opening movement ---
            # This maps the normalized amplitude to a range from negative (closing) to positive (opening)
            movement_offset = (norm_amp * movement_range) - (args.close_amount * args.max_movement)

            dest_quad = src_quad.copy()
            dest_quad[2, 1] += movement_offset # Bottom-right y
            dest_quad[3, 1] += movement_offset # Bottom-left y

            M = cv2.getPerspectiveTransform(src_quad, dest_quad)
            warped_jaw = cv2.warpPerspective(original_frame, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

            final_frame = background.copy()
            final_frame[jaw_mask_for_inpaint > 0] = warped_jaw[jaw_mask_for_inpaint > 0]
            final_frame[foreground_mask > 0] = original_frame[foreground_mask > 0]
            
            cv2.rectangle(final_frame, (10, 10), (210, 30), (0,0,0), -1)
            cv2.rectangle(final_frame, (10, 10), (10 + int(norm_amp * 200), 30), (0,255,0), -1)
            cv2.imshow("Talking Dog", final_frame)
            if cv2.waitKey(int(1000/TARGET_FPS)) & 0xFF == ord("q"): break

    except (Exception, KeyboardInterrupt) as e: 
        logging.error(f"An error occurred: {e}", exc_info=False)
    finally:
        if 'audio_processor' in locals() and audio_processor.stream.active: audio_processor.stop()
        cv2.destroyAllWindows(); logging.info("Application terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Talking Dog with Dynamic Range")
    parser.add_argument("image_path", help="Path to the dog's image.")
    parser.add_argument("--threshold", type=float, default=0.01, help="Mic volume threshold.")
    parser.add_argument("--multiplier", type=float, default=40.0, help="Mic volume sensitivity.")
    parser.add_argument("--max_movement", type=float, default=35.0, help="The base number of pixels for jaw movement.")
    # --- NEW ARGUMENTS ---
    parser.add_argument("--close_amount", type=float, default=0.5, help="How much to CLOSE the mouth on silence (0.0 to 1.0).")
    parser.add_argument("--open_amount", type=float, default=0.5, help="How much to OPEN the mouth on loud sounds (0.0 to 1.0).")
    args = parser.parse_args()
    main(args)
