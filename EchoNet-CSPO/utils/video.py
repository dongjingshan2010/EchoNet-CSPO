"""视频 I/O: 用 OpenCV 读取 AVI 并 resize 成统一尺寸。"""
import cv2
import numpy as np


def read_video(path: str, size: int = 112) -> np.ndarray:
    """返回 (T, H, W, 3) uint8 RGB ndarray。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f'cannot open video: {path}')
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] != size or frame.shape[1] != size:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        # OpenCV 默认 BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise IOError(f'empty video: {path}')
    return np.stack(frames, axis=0)
