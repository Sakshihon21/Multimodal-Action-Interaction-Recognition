"""
Performance Analytics Module
Tracks FPS, latency, and system performance metrics
"""

import time
from collections import deque
import statistics

class PerformanceAnalytics:
    def __init__(self):
        self.frame_times = deque(maxlen=60)  # Store last 60 frames
        self.frame_start_time = None
        self.frame_end_time = None
        self.fps_history = deque(maxlen=60)
        self.latency_history = deque(maxlen=60)
        
        self.total_frames = 0
        self.start_time = time.time()
        
        # Performance thresholds
        self.target_fps = 30
        self.max_latency_ms = 33  # ~30 FPS
        
    def start_frame(self):
        """Mark the start of a frame"""
        self.frame_start_time = time.time()
    
    def end_frame(self):
        """Mark the end of a frame and calculate metrics"""
        if self.frame_start_time is None:
            return
        
        self.frame_end_time = time.time()
        
        # Calculate frame time
        frame_time = self.frame_end_time - self.frame_start_time
        self.frame_times.append(frame_time)
        
        # Calculate FPS
        if len(self.frame_times) > 1:
            avg_frame_time = statistics.mean(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            self.fps_history.append(fps)
        
        # Calculate latency
        latency_ms = frame_time * 1000
        self.latency_history.append(latency_ms)
        
        self.total_frames += 1
        
        # Reset for next frame
        self.frame_start_time = None
    
    def get_fps(self):
        """Get current FPS"""
        if len(self.fps_history) == 0:
            return 0
        return self.fps_history[-1]
    
    def get_avg_fps(self):
        """Get average FPS over last N frames"""
        if len(self.fps_history) == 0:
            return 0
        return statistics.mean(self.fps_history)
    
    def get_latency(self):
        """Get current latency in milliseconds"""
        if len(self.latency_history) == 0:
            return 0
        return self.latency_history[-1]
    
    def get_avg_latency(self):
        """Get average latency"""
        if len(self.latency_history) == 0:
            return 0
        return statistics.mean(self.latency_history)
    
    def get_performance_status(self):
        """Get overall performance status"""
        fps = self.get_fps()
        latency = self.get_latency()
        
        status = "Good"
        if fps < self.target_fps * 0.8 or latency > self.max_latency_ms * 1.2:
            status = "Degraded"
        if fps < self.target_fps * 0.5 or latency > self.max_latency_ms * 2:
            status = "Poor"
        
        return status
    
    def print_summary(self):
        """Print performance summary"""
        runtime = time.time() - self.start_time
        avg_fps = self.get_avg_fps()
        avg_latency = self.get_avg_latency()
        
        print("\n" + "="*50)
        print("Performance Summary")
        print("="*50)
        print(f"Total Runtime: {runtime:.2f} seconds")
        print(f"Total Frames: {self.total_frames}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average Latency: {avg_latency:.2f} ms")
        print(f"Performance Status: {self.get_performance_status()}")
        print("="*50)

