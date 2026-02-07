"""
Screen Capture Module - The Eyes
Captures frames from the game window with zero latency.
"""
import mss
import numpy as np
import time
from typing import Optional, Tuple
import threading
from queue import Queue


class ScreenCapture:
    """Captures frames from the game window."""

    def __init__(
        self,
        monitor: Optional[dict] = None,
        fps: int = 60,
        buffer_size: int = 5,
        grayscale: bool = False
    ):
        """
        Initialize screen capture.

        Args:
            monitor: mss monitor dict. If None, captures primary monitor.
            fps: Target frames per second.
            buffer_size: Number of frames to buffer.
            grayscale: Convert frames to grayscale.
        """
        self.sct = mss.mss()
        self.monitor = monitor or self.sct.monitors[1]  # monitor[1] is primary
        self.fps = fps
        self.target_frametime = 1.0 / fps
        self.grayscale = grayscale
        self.buffer_size = buffer_size

        # Frame storage
        self.frame_queue = Queue(maxsize=buffer_size)
        self.latest_frame = None
        self.frame_count = 0

        # Capture thread
        self.running = False
        self.capture_thread = None

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the monitor.

        Returns:
            np.ndarray: BGR image, or None if capture failed.
        """
        try:
            screenshot = self.sct.grab(self.monitor)
            frame = np.array(screenshot)

            # Remove alpha channel if present (BGRA -> BGR)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]

            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.frame_count += 1
            return frame

        except Exception as e:
            print(f"Capture error: {e}")
            return None

    def start(self):
        """Start continuous capture in background thread."""
        if self.running:
            return

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        """Main capture loop running in background thread."""
        while self.running:
            start_time = time.time()

            frame = self.capture_frame()
            if frame is not None:
                self.latest_frame = frame

                # Add to queue if not full
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # Remove oldest frame and add new one
                    self.frame_queue.get()
                    self.frame_queue.put(frame)

            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, self.target_frametime - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        """Stop continuous capture."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame."""
        return self.latest_frame

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a frame from the buffer.

        Returns:
            np.ndarray: Frame from buffer, or None if buffer empty.
        """
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return self.get_latest_frame()

    def get_frame_sync(self) -> Optional[np.ndarray]:
        """
        Synchronously capture and return a frame.
        Use this when not running continuous capture.

        Returns:
            np.ndarray: Captured frame.
        """
        return self.capture_frame()

    def clear_buffer(self):
        """Clear the frame buffer."""
        while not self.frame_queue.empty():
            self.frame_queue.get()
