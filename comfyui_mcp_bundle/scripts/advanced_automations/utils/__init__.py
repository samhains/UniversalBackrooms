"""
Advanced Automations Utilities

Shared utility classes and functions for advanced automation workflows.
"""

from .eagle_image_pool import EagleImagePool
from .video_chain_manager import VideoChainManager  
from .video_stitcher import VideoStitcher
from .progress_tracker import ProgressTracker

__all__ = [
    "EagleImagePool",
    "VideoChainManager", 
    "VideoStitcher",
    "ProgressTracker"
]