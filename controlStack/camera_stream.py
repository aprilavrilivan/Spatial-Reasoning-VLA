import threading
import time
from pathlib import Path

import cv2


class CameraStream:
    """Small threaded wrapper around OpenCV camera capture.

    The default source stays at 0 because that is the camera setup currently
    working in the lab.
    """

    def __init__(self, src: int = 0, buffer_size: int = 1):
        self.src = src
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.lock = threading.Lock()
        self.capture_lock = threading.Lock()
        self.stopped = False
        with self.capture_lock:
            self.grabbed, self.frame = self.stream.read()

    def start(self):
        threading.Thread(target=self._update_loop, daemon=True).start()
        return self

    def _update_loop(self):
        while not self.stopped:
            with self.capture_lock:
                grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                break
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.001)

    def flush(self, frames: int = 5):
        for _ in range(max(frames, 0)):
            self.update()
            time.sleep(0.01)

    def update(self) -> bool:
        with self.capture_lock:
            grabbed, frame = self.stream.read()
        if not grabbed:
            self.stop()
            return False
        with self.lock:
            self.grabbed = grabbed
            self.frame = frame
        return True

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def snapshot(self, output_path: str | Path | None = None, flush_frames: int = 3):
        self.flush(flush_frames)
        frame = self.read()
        if frame is None:
            raise RuntimeError(f"Camera {self.src} did not return a frame.")
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), frame)
        return frame

    def stop(self):
        self.stopped = True
        self.stream.release()


def display(frame, window_name: str = "CAMERA VIEW"):
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
