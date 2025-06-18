# scr/coreclasses/video_frame_grabber.py

import cv2
import os
import threading
import queue
import time

class VideoFrameGrabber:
    def __init__(self, 
                 video_path,
                 skip_frames=0,
                 max_frames=None,
                 start_frame=0,
                 batch_size=1,
                 batch_skip=0,
                 queue_size=32):
        
        self.video_path = video_path
        self.skip_frames = skip_frames
        self.max_frames = max_frames
        self.start_frame = start_frame
        self.batch_size = batch_size
        self.batch_skip = batch_skip
        self.queue_size = queue_size

        self.cap = None
        self.total_frames = None

        self.frame_queue = queue.Queue(maxsize=self.queue_size)
        self.reader_thread = None
        self.stop_event = threading.Event()
        self.frames_read = 0

    def open(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Cannot open video: {self.video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        print(f"🎞️ Opened video: {self.video_path}")
        print(f"🎞️ Total frames: {self.total_frames}")

        self.reader_thread = threading.Thread(target=self._reader_worker)
        self.reader_thread.daemon = True
        self.reader_thread.start()

    def _reader_worker(self):
        read_count = 0

        while not self.stop_event.is_set():
            if self.max_frames and read_count >= self.max_frames:
                break

            batch = []
            for _ in range(self.batch_size):
                ret, frame = self.cap.read()
                if not ret:
                    break

                batch.append(frame)
                read_count += 1

                # Apply per-frame skip inside batch
                for _ in range(self.skip_frames):
                    _ = self.cap.read()

            if not batch:
                break

            self.frame_queue.put(batch)

            # Apply batch_skip logic
            for _ in range(self.batch_skip):
                for _ in range(self.batch_size + self.skip_frames):
                    _ = self.cap.read()

        self.frame_queue.put(None)
        self.cap.release()

    def __iter__(self):
        self.open()
        while True:
            batch = self.frame_queue.get()
            if batch is None:
                break
            yield batch
        self.reader_thread.join()

    def close(self):
        self.stop_event.set()
        if self.cap:
            self.cap.release()
