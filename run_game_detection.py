#!/usr/bin/env python3
"""
Real-time Game Detection Script
Runs YOLO detection on screen capture for game automation testing.
"""
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

class GameDetectionSystem:
    def __init__(self, model_path='models/train/weights/best.pt', confidence=0.5):
        """Initialize detection system."""
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.capture_config = self._get_screen_config()

        print(f'=== Game Detection System ===')
        print(f'Model: {model_path}')
        print(f'Classes: {self.model.names}')
        print(f'Confidence threshold: {confidence}')
        print(f'Capture region: {self.capture_config}')
        print()

    def _get_screen_config(self):
        """Get screen capture configuration."""
        import mss as mss_module
        with mss_module.mss() as sct:
            monitors = sct.monitors
            if len(monitors) > 1:
                return monitors[1]  # Primary monitor
            return monitors[0]

    def setup_capture_window(self, x=None, y=None, width=None, height=None):
        """Setup capture window region."""
        if width and height:
            current = self._get_screen_config()
            self.capture_config = {
                'top': y if y else 0,
                'left': x if x else 0,
                'width': width,
                'height': height,
                'mon': current['mon']
            }
            print(f'Capture region updated: {self.capture_config}')

    def capture_screen(self):
        """Capture screen region."""
        with mss() as sct:
            screenshot = sct.grab(self.capture_config)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def detect(self, img):
        """Run detection on image."""
        return self.model(img, conf=self.confidence, verbose=False)

    def run_once(self):
        """Run single detection."""
        img = self.capture_screen()
        results = self.detect(img)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[class_id]
                xyxy = box.xyxy[0].cpu().numpy()

                detections.append({
                    'label': label,
                    'confidence': conf,
                    'box': xyxy.tolist()
                })

        return img, results, detections

    def run_realtime(self, show_window=True):
        """Run real-time detection loop."""
        print('Starting real-time detection...')
        print('Press ESC to quit, SPACE to pause')

        paused = False

        try:
            with mss() as sct:
                while True:
                    if not paused:
                        img = self.capture_screen()
                        results = self.detect(img)

                        if show_window and len(results) > 0 and len(results[0].boxes) > 0:
                            annotated = results[0].plot()
                            cv2.imshow('Game Detection', annotated)

                    # Handle keyboard
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        print('Quitting...')
                        break
                    elif key == 32:  # SPACE
                        paused = not paused
                        print(f'Paused: {paused}')

                    # Check window
                    if show_window and cv2.getWindowProperty('Game Detection', cv2.WND_PROP_VISIBLE) < 1:
                        break

        except KeyboardInterrupt:
            print('\nInterrupted by user')
        finally:
            cv2.destroyAllWindows()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Game detection with trained YOLO model')
    parser.add_argument('--model', default='models/train/weights/best.pt', help='Path to trained model')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--x', type=int, help='Capture window x position')
    parser.add_argument('--y', type=int, help='Capture window y position')
    parser.add_argument('--width', type=int, help='Capture window width')
    parser.add_argument('--height', type=int, help='Capture window height')
    parser.add_argument('--test', action='store_true', help='Run single test instead of realtime')
    args = parser.parse_args()

    detector = GameDetectionSystem(args.model, args.confidence)

    if args.width and args.height:
        detector.setup_capture_window(args.x, args.y, args.width, args.height)

    if args.test:
        # Single test
        print('Running single detection test...')
        img, results, detections = detector.run_once()

        print(f'\nDetections found: {len(detections)}')
        for i, det in enumerate(detections, 1):
            print(f"  [{i}] {det['label']}: {det['confidence']:.3f}")

        # Save annotated image
        if len(results) > 0 and len(results[0].boxes) > 0:
            annotated = results[0].plot()
            cv2.imwrite('single_test_result.jpg', annotated)
            print('\nSaved result to: single_test_result.jpg')
    else:
        # Real-time detection
        detector.run_realtime()


if __name__ == '__main__':
    main()
