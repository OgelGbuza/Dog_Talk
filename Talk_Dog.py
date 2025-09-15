"""
talk_dog_rotation.py – Hinge‑based rotation for realistic jaw movement 🐕🔧
--------------------------------------------------------------------------
This version introduces a hinge‑style rotation of the lower jaw, making the
mouth open along a circular arc rather than simply translating the jaw
downward.  The rotation pivot is determined from the upper corners of the
selected jaw region, and the jaw ROI is rotated around this pivot in
proportion to the audio amplitude.  An optional scale factor can be
applied to exaggerate the opening.  As before, ambient noise calibration,
dual smoothing, and alpha blending are used to produce a smooth and
stable animation.

Steps for the user:
    1. Outline the entire LOWER JAW area.
    2. Outline the FOREGROUND/UPPER LIP area (teeth/lips that should remain static).

The script computes the pivot automatically from the jaw polygon (the
midpoint between the two topmost corners).  If you need more control,
you can manually adjust the `max_angle` and `scale_factor` parameters.
"""

import argparse
import logging
import sys
import time

import cv2
import numpy as np
import sounddevice as sd

from dataclasses import dataclass

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Constants
TARGET_FPS = 60
AUDIO_BLOCK_SIZE = 1024
SAMPLE_RATE = 44100


class EMA:
    """Exponential moving average filter."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False

    def update(self, new_value: float) -> float:
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value
        return self.value


class PointSelector:
    """Interactive polygon selector."""

    def __init__(self, window_name: str, image: np.ndarray) -> None:
        self.window_name = window_name
        self.image_clone = image.copy()
        self.original_image = image.copy()
        self.points: list[tuple[int, int]] = []
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.image_clone, (x, y), 5, (0, 255, 0), -1)
            if len(self.points) > 1:
                cv2.line(self.image_clone, self.points[-2], self.points[-1], (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.image_clone)
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()
            self.image_clone = self.original_image.copy()
            for i, p in enumerate(self.points):
                cv2.circle(self.image_clone, p, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(self.image_clone, self.points[i - 1], p, (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.image_clone)

    def get_points(self) -> np.ndarray:
        print("Click to outline the polygon. Press 'd' or Enter when done, 'r' to reset.")
        while True:
            cv2.imshow(self.window_name, self.image_clone)
            key = cv2.waitKey(20) & 0xFF
            if (key == ord('d') or key == 13) and len(self.points) >= 3:
                break
            elif key == ord('r'):
                self.points.clear()
                self.image_clone = self.original_image.copy()
                logging.info("Points reset.")
            elif key == 27:
                logging.info("Selection cancelled.")
                sys.exit(0)
        cv2.destroyWindow(self.window_name)
        return np.array(self.points, dtype=np.int32)


class AudioProcessor:
    """Capture audio and compute smoothed RMS values."""

    def __init__(self, block_size: int, ema_alpha: float) -> None:
        self.ema = EMA(ema_alpha)
        self.amplitude: float = 0.0
        try:
            self.stream = sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=block_size,
                dtype='float32'
            )
        except Exception as e:
            logging.error(f"Failed to open audio stream: {e}. Is a microphone connected?")
            sys.exit(1)

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            logging.warning(status)
        rms = float(np.sqrt(np.mean(indata**2)))
        self.amplitude = self.ema.update(rms)

    def start(self) -> None:
        self.stream.start()
        logging.info("Audio stream started.")

    def stop(self) -> None:
        self.stream.stop()
        self.stream.close()
        logging.info("Audio stream closed.")


def get_jaw_corners(jaw_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the two upper corners (left and right) of the jaw polygon."""
    # Sort by y (ascending) and take the top two points
    sorted_by_y = jaw_points[jaw_points[:, 1].argsort()]
    top_points = sorted_by_y[:2]
    # Sort these by x to get left and right
    left = top_points[np.argmin(top_points[:, 0])]
    right = top_points[np.argmax(top_points[:, 0])]
    return left.astype(np.float32), right.astype(np.float32)


def compute_alpha_roi(mask_roi: np.ndarray, kernel_ratio: float) -> np.ndarray:
    """Compute a blurred alpha mask for the ROI."""
    h, w = mask_roi.shape
    k = int(max(h, w) * kernel_ratio)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(mask_roi.astype(np.float32), (k, k), 0)
    if blurred.max() > 0:
        blurred /= blurred.max()
    return blurred


def calibrate_noise(audio_proc: AudioProcessor, duration: float) -> float:
    """Measure ambient noise level and return the median amplitude."""
    logging.info(f"Calibrating ambient noise for {duration:.2f} seconds…")
    samples: list[float] = []
    start = time.time()
    while time.time() - start < duration:
        samples.append(audio_proc.amplitude)
        time.sleep(0.01)
    if samples:
        median_val = float(np.median(samples))
        logging.info(f"Ambient noise calibrated to {median_val:.6f}")
        return median_val
    return 0.0


def main(args: argparse.Namespace) -> None:
    try:
        image = cv2.imread(args.image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {args.image_path}")
        rows, cols, _ = image.shape
        # --- User input ---
        jaw_sel = PointSelector("Step 1: Outline the LOWER JAW", image.copy())
        jaw_pts = jaw_sel.get_points()
        fg_sel = PointSelector("Step 2: Outline the UPPER LIP/TEETH", image.copy())
        fg_pts = fg_sel.get_points()

        # Masks
        jaw_mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.fillPoly(jaw_mask, [jaw_pts], 255)
        foreground_mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.fillPoly(foreground_mask, [fg_pts], 255)
        # We will expand the foreground mask downward later once the ROI height is known.

        # Inpaint jaw region
        logging.info("Inpainting jaw region…")
        background = cv2.inpaint(image, jaw_mask, 3, cv2.INPAINT_TELEA)
        logging.info("Inpainting complete.")

        # Jaw bounding box and margin expansion
        x, y, w, h = cv2.boundingRect(jaw_pts)
        margin_x = int(w * args.margin_ratio)
        margin_y = int(h * args.margin_ratio)
        # Compute expanded ROI coordinates, clamped to image boundaries
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(cols, x + w + margin_x)
        y1 = min(rows, y + h + margin_y)
        # Compute ROI dimensions
        w_roi = x1 - x0
        h_roi = y1 - y0
        # Extract the expanded ROI and corresponding mask
        jaw_roi_orig = image[y0:y1, x0:x1].copy()
        jaw_mask_roi = jaw_mask[y0:y1, x0:x1].copy()

        # Now that we know h_roi, optionally expand the foreground mask downward to
        # ensure that the upper lip area always sits on top of the jaw.  We do
        # this by shifting the mask downward by `shift` pixels and OR'ing it
        # with the original.  This adds extra coverage below the original
        # foreground outline.
        if args.foreground_dilate_ratio > 0.0:
            shift_amount = max(1, int(h_roi * args.foreground_dilate_ratio))
            # Create a shifted copy of the mask (downwards)
            shifted = np.zeros_like(foreground_mask)
            shifted[shift_amount:, :] = foreground_mask[:-shift_amount, :]
            # Combine original and shifted masks
            foreground_mask = cv2.bitwise_or(foreground_mask, shifted)
        # Determine hinge points and pivot (global coordinates)
        left_hinge, right_hinge = get_jaw_corners(jaw_pts)
        pivot_global = ((left_hinge + right_hinge) / 2.0)
        # Convert pivot to local coordinates within the expanded ROI
        pivot_local = pivot_global - np.array([x0, y0], dtype=np.float32)
        # Precompute alpha mask for ROI blending (same size as ROI)
        alpha_roi_orig = compute_alpha_roi(jaw_mask_roi, args.blend_width)

        # Audio processing
        audio_proc = AudioProcessor(AUDIO_BLOCK_SIZE, args.ema_alpha)
        audio_proc.start()
        # Ambient noise calibration
        ambient = 0.0
        if args.calibration_time > 0.0:
            time.sleep(0.2)
            ambient = calibrate_noise(audio_proc, args.calibration_time)
        effective_threshold = ambient + args.threshold
        # Smoothing filters
        jaw_ema = EMA(args.jaw_alpha)
        while True:
            raw_amp = audio_proc.amplitude
            delta = raw_amp - effective_threshold
            amp_val = delta * args.multiplier if delta > 0 else 0.0
            if amp_val > 1.0:
                amp_val = 1.0
            jaw_val = jaw_ema.update(amp_val)
            # Compute rotation angle and optional scale factor
            angle = jaw_val ** args.gamma * args.max_angle
            scale = 1.0 + jaw_val * args.scale_factor
            # Rotation matrix around pivot
            M = cv2.getRotationMatrix2D((float(pivot_local[0]), float(pivot_local[1])), angle, scale)
            # Apply transformation to ROI, mask, and alpha masks with size equal to expanded ROI
            rotated_roi = cv2.warpAffine(jaw_roi_orig, M, (w_roi, h_roi), borderMode=cv2.BORDER_REFLECT)
            rotated_mask = cv2.warpAffine(jaw_mask_roi, M, (w_roi, h_roi), flags=cv2.INTER_NEAREST)
            rotated_alpha = cv2.warpAffine(alpha_roi_orig, M, (w_roi, h_roi), borderMode=cv2.BORDER_REFLECT)
            # Composite onto background: coordinates y0:y0+h_roi, x0:x0+w_roi
            composite_region = background[y0:y0+h_roi, x0:x0+w_roi].copy()
            # Blend rotated ROI and background using rotated_alpha
            comp = (rotated_roi.astype(np.float32) * rotated_alpha[..., None] + composite_region.astype(np.float32) * (1 - rotated_alpha[..., None])).astype(np.uint8)
            # Insert composite into a working frame
            frame = background.copy()
            frame[y0:y0+h_roi, x0:x0+w_roi] = comp
            # Restore foreground/upper lip
            frame[foreground_mask > 0] = image[foreground_mask > 0]
            # Optionally head bob
            if args.head_bob:
                hb_angle = jaw_val ** args.gamma * args.head_bob_max_angle * np.sin(time.time() * args.head_bob_freq)
                M_hb = cv2.getRotationMatrix2D((cols // 2, rows // 2), hb_angle, 1.0)
                frame = cv2.warpAffine(frame, M_hb, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
            # Draw amplitude bar
            bar_length = int(jaw_val ** args.gamma * 200)
            cv2.rectangle(frame, (10, 10), (210, 30), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (10 + bar_length, 30), (0, 255, 0), -1)
            cv2.imshow("Talking Dog (rotation)", frame)
            if cv2.waitKey(int(1000 / TARGET_FPS)) & 0xFF == ord('q'):
                break
    except (Exception, KeyboardInterrupt) as e:
        logging.error(f"An error occurred: {e}", exc_info=False)
    finally:
        try:
            audio_proc.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        logging.info("Application terminated.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Talking dog avatar with hinge‑based jaw rotation")
    parser.add_argument('image_path', help="Path to the dog's image")
    parser.add_argument('--threshold', type=float, default=0.02, help="Base volume threshold above ambient noise")
    parser.add_argument('--multiplier', type=float, default=40.0, help="Scaling from audio amplitude to open value")
    parser.add_argument('--ema_alpha', type=float, default=0.5, help="EMA smoothing factor for audio RMS (0–1)")
    parser.add_argument('--jaw_alpha', type=float, default=0.3, help="EMA smoothing factor for jaw movement (0–1)")
    parser.add_argument('--gamma', type=float, default=0.8, help="Non‑linear exponent for amplitude mapping")
    parser.add_argument('--max_angle', type=float, default=15.0, help="Maximum rotation angle for the jaw (degrees)")
    parser.add_argument('--scale_factor', type=float, default=0.1, help="Additional scale factor applied as jaw opens")
    parser.add_argument('--calibration_time', type=float, default=0.5, help="Seconds to measure ambient noise (0 disables)")
    parser.add_argument('--blend_width', type=float, default=0.05, help="Relative blur kernel width for blending mask (0–1)")
    parser.add_argument('--head_bob', action='store_true', help="Enable head bob animation")
    parser.add_argument('--head_bob_max_angle', type=float, default=2.0, help="Maximum head bob rotation (degrees)")
    parser.add_argument('--head_bob_freq', type=float, default=6.0, help="Head bob oscillation frequency (Hz)")
    parser.add_argument('--margin_ratio', type=float, default=0.3, help="Fraction of jaw bbox size to pad on each side to accommodate rotation (0–1)")
    parser.add_argument('--foreground_dilate_ratio', type=float, default=0.1, help="Fraction of the jaw ROI height to shift the foreground mask downward to prevent overlap")
    args = parser.parse_args()
    main(args)