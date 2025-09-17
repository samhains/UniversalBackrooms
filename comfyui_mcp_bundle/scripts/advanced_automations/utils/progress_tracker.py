#!/usr/bin/env python3
"""
Progress Tracker

Handles progress tracking, logging, and user feedback for long-running automation tasks.
Provides checkpoint functionality and estimated time remaining calculations.
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List


class ProgressTracker:
    """
    Tracks progress for long-running automation tasks.
    Provides progress reporting, time estimates, and checkpoint functionality.
    """
    
    def __init__(self, total_steps: int, task_name: str = "Task"):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps in the task
            task_name: Human-readable name of the task
        """
        self.total_steps = total_steps
        self.task_name = task_name
        self.current_step = 0
        self.start_time = time.time()
        self.step_times: List[float] = []
        self.checkpoints: Dict[int, Dict[str, Any]] = {}
        self.last_update_time = self.start_time
    
    def start(self) -> None:
        """Mark the start of the task."""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        print(f"🚀 Starting {self.task_name} - {self.total_steps} steps total")
        self._print_progress()
    
    def step(self, description: Optional[str] = None) -> None:
        """
        Mark completion of a step.
        
        Args:
            description: Optional description of what was completed
        """
        current_time = time.time()
        step_duration = current_time - self.last_update_time
        self.step_times.append(step_duration)
        self.last_update_time = current_time
        
        self.current_step += 1
        
        if description:
            print(f"  ✓ Step {self.current_step}: {description}")
        
        self._print_progress()
    
    def checkpoint(self, data: Dict[str, Any]) -> None:
        """
        Save a checkpoint with current progress and data.
        
        Args:
            data: Checkpoint data to save
        """
        self.checkpoints[self.current_step] = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": self.get_elapsed_time(),
            "data": data
        }
        print(f"💾 Checkpoint saved at step {self.current_step}")
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the most recent checkpoint data."""
        if not self.checkpoints:
            return None
        
        latest_step = max(self.checkpoints.keys())
        return self.checkpoints[latest_step]
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since start in seconds."""
        return time.time() - self.start_time
    
    def get_estimated_remaining_time(self) -> Optional[float]:
        """
        Calculate estimated remaining time based on average step duration.
        
        Returns:
            Estimated seconds remaining, or None if no steps completed
        """
        if not self.step_times or self.current_step == 0:
            return None
        
        avg_step_time = sum(self.step_times) / len(self.step_times)
        remaining_steps = self.total_steps - self.current_step
        return avg_step_time * remaining_steps
    
    def get_progress_percentage(self) -> float:
        """Get current progress as percentage."""
        if self.total_steps == 0:
            return 100.0
        return (self.current_step / self.total_steps) * 100
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def _print_progress(self) -> None:
        """Print current progress information."""
        percentage = self.get_progress_percentage()
        elapsed = self.get_elapsed_time()
        
        # Create progress bar
        bar_width = 30
        filled_width = int(bar_width * (percentage / 100))
        bar = "█" * filled_width + "░" * (bar_width - filled_width)
        
        # Format basic progress info
        progress_line = f"📊 [{bar}] {percentage:.1f}% ({self.current_step}/{self.total_steps})"
        
        # Add time information
        elapsed_str = self._format_time(elapsed)
        time_info = f" | Elapsed: {elapsed_str}"
        
        # Add ETA if available
        eta = self.get_estimated_remaining_time()
        if eta is not None:
            eta_str = self._format_time(eta)
            time_info += f" | ETA: {eta_str}"
        
        print(progress_line + time_info)
    
    def print_summary(self) -> None:
        """Print final summary of the task."""
        elapsed = self.get_elapsed_time()
        elapsed_str = self._format_time(elapsed)
        
        if self.current_step == self.total_steps:
            print(f"✅ {self.task_name} completed successfully!")
        else:
            print(f"⚠️  {self.task_name} completed with {self.current_step}/{self.total_steps} steps")
        
        print(f"⏱️  Total time: {elapsed_str}")
        
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            print(f"📈 Average step time: {self._format_time(avg_step_time)}")
        
        if self.checkpoints:
            print(f"💾 Checkpoints saved: {len(self.checkpoints)}")
    
    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return self.current_step >= self.total_steps
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the task progress."""
        stats = {
            "task_name": self.task_name,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "progress_percentage": self.get_progress_percentage(),
            "elapsed_time": self.get_elapsed_time(),
            "is_complete": self.is_complete(),
            "checkpoints_count": len(self.checkpoints)
        }
        
        # Add time estimates if available
        eta = self.get_estimated_remaining_time()
        if eta is not None:
            stats["estimated_remaining_time"] = eta
        
        if self.step_times:
            stats["average_step_time"] = sum(self.step_times) / len(self.step_times)
            stats["fastest_step_time"] = min(self.step_times)
            stats["slowest_step_time"] = max(self.step_times)
        
        return stats