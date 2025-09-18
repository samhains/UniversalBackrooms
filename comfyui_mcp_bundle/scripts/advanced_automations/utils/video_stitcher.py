#!/usr/bin/env python3
"""
Video Stitcher

Handles FFmpeg integration for concatenating multiple video segments into a single
seamless video. Supports different stitching methods and video optimization.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests


class VideoStitcher:
    """
    Handles video concatenation using FFmpeg.
    Supports direct concatenation and crossfade transitions.
    """
    
    def __init__(self, temp_dir: Optional[str] = None, cleanup: bool = True):
        """
        Initialize the video stitcher.
        
        Args:
            temp_dir: Directory for temporary files (default: system temp)
            cleanup: Whether to cleanup temp files after stitching
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.cleanup = cleanup
        self.work_dir = None
        
        # Check if FFmpeg is available
        self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> None:
        """Check if FFmpeg is available in the system."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            print("✓ FFmpeg is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg to use video stitching functionality.\n"
                "Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)"
            )
    
    def _create_work_directory(self) -> str:
        """Create a temporary working directory for this stitching operation."""
        self.work_dir = tempfile.mkdtemp(prefix="chain_video_", dir=self.temp_dir)
        print(f"📁 Created working directory: {self.work_dir}")
        return self.work_dir
    
    def _download_video(self, url: str, filename: str) -> str:
        """
        Download a video from URL to the working directory.
        
        Args:
            url: Video URL to download
            filename: Local filename to save as
            
        Returns:
            Path to downloaded file
        """
        local_path = os.path.join(self.work_dir, filename)
        
        try:
            print(f"⬇️  Downloading: {url}")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✓ Downloaded: {filename}")
            return local_path
            
        except Exception as e:
            raise Exception(f"Failed to download video from {url}: {e}")
    
    def _create_concat_file(self, video_paths: List[str]) -> str:
        """
        Create FFmpeg concat file listing all input videos.
        
        Args:
            video_paths: List of local video file paths
            
        Returns:
            Path to concat file
        """
        concat_file = os.path.join(self.work_dir, "concat_list.txt")
        
        with open(concat_file, 'w') as f:
            for video_path in video_paths:
                # FFmpeg concat format requires forward slashes even on Windows
                normalized_path = video_path.replace('\\', '/')
                f.write(f"file '{normalized_path}'\n")
        
        print(f"✓ Created concat file with {len(video_paths)} segments")
        return concat_file
    
    def stitch_videos_direct(self, video_urls: List[str], output_filename: str = "final_chain_video.mp4") -> str:
        """
        Stitch videos using direct concatenation (no transitions).
        
        Args:
            video_urls: List of video URLs to stitch together
            output_filename: Name of output file
            
        Returns:
            Path to stitched video file
        """
        if len(video_urls) < 2:
            raise ValueError("Need at least 2 videos to stitch together")
        
        # Create working directory
        if not self.work_dir:
            self._create_work_directory()
        
        try:
            # Download all videos
            video_paths = []
            for i, url in enumerate(video_urls):
                filename = f"segment_{i:03d}.mp4"
                local_path = self._download_video(url, filename)
                video_paths.append(local_path)
            
            # Create concat file
            concat_file = self._create_concat_file(video_paths)
            
            # Output path
            output_path = os.path.join(self.work_dir, output_filename)
            
            # FFmpeg command for direct concatenation
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",  # Copy streams without re-encoding for speed
                "-y",  # Overwrite output file
                output_path
            ]
            
            print("🎬 Stitching videos with direct concatenation...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"✅ Successfully stitched {len(video_urls)} videos")
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg concatenation failed: {e.stderr}")
        except Exception as e:
            raise Exception(f"Video stitching failed: {e}")
    
    def stitch_videos_crossfade(self, video_urls: List[str], crossfade_duration: float = 0.5, 
                               output_filename: str = "final_chain_video.mp4") -> str:
        """
        Stitch videos with crossfade transitions between segments.
        
        Args:
            video_urls: List of video URLs to stitch together
            crossfade_duration: Duration of crossfade in seconds
            output_filename: Name of output file
            
        Returns:
            Path to stitched video file
        """
        if len(video_urls) < 2:
            raise ValueError("Need at least 2 videos to stitch together")
        
        # Create working directory
        if not self.work_dir:
            self._create_work_directory()
        
        try:
            # Download all videos
            video_paths = []
            for i, url in enumerate(video_urls):
                filename = f"segment_{i:03d}.mp4"
                local_path = self._download_video(url, filename)
                video_paths.append(local_path)
            
            # Output path
            output_path = os.path.join(self.work_dir, output_filename)
            
            # Build complex FFmpeg filter for crossfade transitions
            # This is more complex but creates smoother transitions
            if len(video_paths) == 2:
                # Simple case: just two videos
                cmd = [
                    "ffmpeg",
                    "-i", video_paths[0],
                    "-i", video_paths[1],
                    "-filter_complex",
                    f"[0:v][1:v]xfade=transition=fade:duration={crossfade_duration}:offset=0[v]",
                    "-map", "[v]",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-y",
                    output_path
                ]
            else:
                # Multiple videos - use simpler direct concat for now
                # TODO: Implement complex multi-video crossfade
                print("⚠️  Multiple video crossfade not yet implemented, using direct concatenation")
                return self.stitch_videos_direct(video_urls, output_filename)
            
            print("🎬 Stitching videos with crossfade transitions...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"✅ Successfully stitched {len(video_urls)} videos with crossfade")
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg crossfade failed: {e.stderr}")
        except Exception as e:
            raise Exception(f"Video stitching failed: {e}")
    
    def stitch_videos(self, video_urls: List[str], method: str = "direct", 
                     output_filename: str = "final_chain_video.mp4", **kwargs) -> str:
        """
        Stitch videos using the specified method.
        
        Args:
            video_urls: List of video URLs to stitch together
            method: Stitching method ('direct' or 'crossfade')
            output_filename: Name of output file
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Path to stitched video file
        """
        if method == "direct":
            return self.stitch_videos_direct(video_urls, output_filename)
        elif method == "crossfade":
            crossfade_duration = kwargs.get("crossfade_duration", 0.5)
            return self.stitch_videos_crossfade(video_urls, crossfade_duration, output_filename)
        else:
            raise ValueError(f"Unknown stitching method: {method}. Use 'direct' or 'crossfade'")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about a video file using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            return json.loads(result.stdout)
            
        except Exception as e:
            print(f"Warning: Could not get video info for {video_path}: {e}")
            return {}
    
    def cleanup_temp_files(self) -> None:
        """Remove temporary working directory and all files."""
        if self.work_dir and os.path.exists(self.work_dir) and self.cleanup:
            try:
                shutil.rmtree(self.work_dir)
                print(f"🧹 Cleaned up temporary files: {self.work_dir}")
                self.work_dir = None
            except Exception as e:
                print(f"Warning: Failed to cleanup temp files: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_temp_files()