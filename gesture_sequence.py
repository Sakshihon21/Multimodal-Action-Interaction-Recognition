"""
Gesture Sequence Recognition Module
Recognizes complex gesture patterns and sequences for advanced commands
"""

from collections import deque
import time

class GestureSequenceRecognizer:
    def __init__(self):
        self.sequence_patterns = {
            # Pattern: (gesture sequence, command name)
            ('FIST', 'OPEN_PALM', 'FIST'): 'SNAP',
            ('POINT', 'VICTORY', 'POINT'): 'SELECT_TOGGLE',
            ('THUMB_UP', 'THUMB_UP', 'THUMB_UP'): 'TRIPLE_ACTION',
            ('OPEN_PALM', 'FIST', 'OPEN_PALM'): 'RESET',
            ('VICTORY', 'VICTORY', 'POINT'): 'DRAW_AND_SELECT',
            ('POINT', 'OK', 'POINT'): 'CLICK_AND_MOVE',
            ('FOUR', 'THREE', 'TWO'): 'COUNTDOWN',
            ('OPEN_PALM', 'POINT', 'VICTORY'): 'MULTI_ACTION',
        }
        
        self.sequence_history = deque(maxlen=10)
        self.last_sequence_time = time.time()
        self.sequence_timeout = 2.0  # seconds
    
    def recognize_sequence(self, gesture_history):
        """Recognize gesture sequences from history"""
        if len(gesture_history) < 2:
            return None
        
        # Get recent gestures (last 3)
        recent_gestures = tuple(gesture_history[-3:])
        
        # Check for exact matches
        if recent_gestures in self.sequence_patterns:
            return self.sequence_patterns[recent_gestures]
        
        # Check for partial matches (last 2 gestures)
        if len(recent_gestures) >= 2:
            partial_pattern = recent_gestures[-2:]
            for pattern, command in self.sequence_patterns.items():
                if len(pattern) >= 2 and pattern[:2] == partial_pattern:
                    return f"Partial: {command}"
        
        return None
    
    def add_gesture(self, gesture):
        """Add gesture to sequence history"""
        current_time = time.time()
        
        # Reset if too much time has passed
        if current_time - self.last_sequence_time > self.sequence_timeout:
            self.sequence_history.clear()
        
        self.sequence_history.append(gesture)
        self.last_sequence_time = current_time
    
    def get_active_sequence(self):
        """Get currently active sequence being formed"""
        if len(self.sequence_history) == 0:
            return None
        
        # Return the last few gestures as a potential sequence
        return list(self.sequence_history)[-3:]
    
    def reset_sequence(self):
        """Reset the sequence history"""
        self.sequence_history.clear()
        self.last_sequence_time = time.time()
    
    def add_custom_sequence(self, gesture_sequence, command_name):
        """Allow adding custom gesture sequences"""
        self.sequence_patterns[tuple(gesture_sequence)] = command_name

