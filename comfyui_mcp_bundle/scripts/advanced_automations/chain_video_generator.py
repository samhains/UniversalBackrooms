#!/usr/bin/env python3
"""
Chain Video Generator

Advanced automation for creating smooth video chains using remix_image + f2f_video workflows.
Part of the advanced_automations suite for complex multi-step AI workflows.

Workflow:
1. Load Eagle images by tags
2. Create initial start image (2 random Eagle images → remix)
3. Create initial end image (2 different random Eagle images → remix)  
4. Generate f2f video (start → end)
5. Loop: previous end becomes new start, generate new end, create f2f video
6. Stitch all f2f videos into final compilation
7. Upload to database

Usage:
    python chain_video_generator.py --tags "environment landscape" --iterations 10 --width 720 --height 720 --frames 81 --enable-qwen-captioner --prompt "smooth transition"
    python -u chain_video_generator.py --folder "TOUHOU" --iterations 5 --width 1280 --height 720 --frames 180 --stitch-method direct
"""

import os
import sys
import json
import argparse
import requests
import tempfile
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Import utilities
from utils.eagle_image_pool import EagleImagePool
from utils.video_chain_manager import VideoChainManager
from utils.video_stitcher import VideoStitcher
from utils.progress_tracker import ProgressTracker

# Add src to path to import supabase client directly 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'core'))
from supabase_client import get_supabase_client

# Load environment variables
load_dotenv()


class ChainVideoGenerator:
    """
    Main class for generating chained videos using remix and f2f workflows.
    """
    
    def __init__(self, tags: List[str] = None, folder: str = None, iterations: int = None, 
                 width: int = 720, height: int = 720, frame_length: int = 81, stitch_method: str = "direct",
                 enable_qwen_captioner: bool = False, prompt: str = ""):
        """
        Initialize the chain video generator.
        
        Args:
            tags: List of Eagle image tags to filter by (optional if folder provided)
            folder: Eagle folder name to filter by (optional if tags provided)  
            iterations: Number of f2f video segments to create
            width: Video width in pixels
            height: Video height in pixels
            frame_length: Number of frames per f2f video
            stitch_method: Method for stitching videos ('direct' or 'crossfade')
            enable_qwen_captioner: Enable Qwen captioner for motion prompts (default: False)
            prompt: Custom prompt when Qwen captioner is disabled (default: empty string)
        """
        if not tags and not folder:
            raise ValueError("Either tags or folder must be provided")
        if iterations is None:
            raise ValueError("Iterations parameter is required")
            
        self.tags = tags
        self.folder = folder
        self.iterations = iterations
        self.width = width
        self.height = height
        self.frame_length = frame_length
        self.stitch_method = stitch_method
        self.enable_qwen_captioner = enable_qwen_captioner
        self.prompt = prompt
        
        # Initialize components
        self.eagle_pool = EagleImagePool(tags=tags, folder=folder)
        self.chain_manager = VideoChainManager()
        self.progress = ProgressTracker(
            total_steps=iterations + 2,  # iterations + initial setup + final stitching
            task_name=f"Chain Video Generation ({iterations} segments)"
        )
        
        # ComfyUI API endpoints
        self.comfyui_base = "http://0.0.0.0:9000"
        self.remix_endpoint = f"{self.comfyui_base}/remix_image"
        self.f2f_endpoint = f"{self.comfyui_base}/generate_f2f_video"
        
        # Session parameters for database tracking
        self.session_params = {
            "tags": tags,
            "folder": folder,
            "iterations": iterations,
            "width": width,
            "height": height,
            "frame_length": frame_length,
            "stitch_method": stitch_method,
            "enable_qwen_captioner": enable_qwen_captioner,
            "prompt": prompt
        }
        
        # Storage for generated content
        self.remix_images: List[Dict[str, Any]] = []
        self.f2f_videos: List[Dict[str, Any]] = []
        self.current_start_image: Optional[str] = None
    
    def _call_remix_api(self, image1_url: str, image2_url: str) -> str:
        """
        Call ComfyUI remix_image API.
        
        Args:
            image1_url: First source image URL
            image2_url: Second source image URL
            
        Returns:
            Generated remix image URL
        """
        try:
            payload = {
                "image1_url": image1_url,
                "image2_url": image2_url,
                "width": self.width,
                "height": self.height
            }
            
            print(f"🎨 Generating remix image...")
            response = requests.post(self.remix_endpoint, json=payload)
            response.raise_for_status()
            
            result = response.json()
            if "image_url" in result:
                return result["image_url"]
            elif "error" in result:
                raise Exception(f"ComfyUI remix error: {result['error']}")
            else:
                raise Exception(f"Unexpected response format: {result}")
                
        except Exception as e:
            raise Exception(f"Remix API call failed: {e}")
    
    def _call_f2f_api(self, start_image_url: str, end_image_url: str) -> str:
        """
        Call ComfyUI generate_f2f_video API.
        
        Args:
            start_image_url: Starting frame image URL
            end_image_url: Ending frame image URL
            
        Returns:
            Generated f2f video URL
        """
        try:
            payload = {
                "image1_url": start_image_url,
                "image2_url": end_image_url,
                "width": self.width,
                "height": self.height,
                "frame_length": self.frame_length,
                "enable_qwen_captioner": self.enable_qwen_captioner,
                "prompt": self.prompt
            }
            
            print(f"🎬 Generating f2f video...")
            response = requests.post(self.f2f_endpoint, json=payload)
            response.raise_for_status()
            
            result = response.json()
            if "video_url" in result:
                return result["video_url"]
            elif "error" in result:
                raise Exception(f"ComfyUI f2f error: {result['error']}")
            else:
                raise Exception(f"Unexpected response format: {result}")
                
        except Exception as e:
            raise Exception(f"F2F API call failed: {e}")
    
    def _upload_remix_to_storage(self, comfyui_url: str, iteration: int, role: str) -> str:
        """
        Download remix image from ComfyUI and upload to Supabase Storage.
        
        Args:
            comfyui_url: ComfyUI image URL
            iteration: Current iteration number
            role: Role of image ('start' or 'end')
            
        Returns:
            Supabase Storage URL
        """
        try:
            # Download image from ComfyUI
            response = requests.get(comfyui_url)
            response.raise_for_status()
            
            # Create unique filename
            timestamp = int(time.time())
            filename = f"chain_remix_{iteration}_{role}_{timestamp}.png"
            
            # Upload to Supabase Storage using proper client
            supabase = get_supabase_client()
            
            storage_path = f"images/{filename}"
            upload_response = supabase.storage.from_("eagle-images").upload(
                storage_path, response.content, file_options={"upsert": "true"}
            )
            
            # Get public URL
            public_url = supabase.storage.from_("eagle-images").get_public_url(storage_path)
            
            print(f"☁️  Uploaded {role} image to storage: {filename}")
            return public_url
            
        except Exception as e:
            raise Exception(f"Failed to upload remix image to storage: {e}")

    def _upload_video_to_storage(self, local_video_path: str, session_id: str) -> str:
        """
        Upload final stitched video to Supabase Storage.
        
        Args:
            local_video_path: Local path to the video file
            session_id: Session ID for unique filename
            
        Returns:
            Supabase Storage URL
        """
        try:
            # Read video file
            with open(local_video_path, 'rb') as video_file:
                video_content = video_file.read()
            
            # Create unique filename
            timestamp = int(time.time())
            filename = f"chain_video_{session_id}_{timestamp}.mp4"
            
            # Upload to Supabase Storage using proper client
            supabase = get_supabase_client()
            
            storage_path = f"videos/{filename}"
            upload_response = supabase.storage.from_("eagle-images").upload(
                storage_path, video_content, file_options={"upsert": "true"}
            )
            
            # Get public URL
            public_url = supabase.storage.from_("eagle-images").get_public_url(storage_path)
            
            print(f"☁️  Uploaded final video to storage: {filename}")
            return public_url
            
        except Exception as e:
            raise Exception(f"Failed to upload video to storage: {e}")
    
    def generate_initial_images(self) -> tuple[str, str]:
        """
        Generate the initial start and end images for the chain.
        
        Returns:
            Tuple of (start_image_url, end_image_url)
        """
        print("🎯 Generating initial start and end images...")
        
        # Generate start image
        start_eagle_imgs = self.eagle_pool.select_2_different_random_images()
        print(f"  Start sources: {start_eagle_imgs[0]['title']} + {start_eagle_imgs[1]['title']}")
        
        start_comfyui_url = self._call_remix_api(
            start_eagle_imgs[0]['image_url'], 
            start_eagle_imgs[1]['image_url']
        )
        
        # Upload start image to storage
        start_image_url = self._upload_remix_to_storage(start_comfyui_url, 0, "start")
        
        # Store start image in database
        self.chain_manager.store_remix_image(
            start_image_url,
            [start_eagle_imgs[0]['image_url'], start_eagle_imgs[1]['image_url']],
            0,
            "start",
            [start_eagle_imgs[0]['title'], start_eagle_imgs[1]['title']]
        )
        
        # Generate end image
        end_eagle_imgs = self.eagle_pool.select_2_different_random_images()
        print(f"  End sources: {end_eagle_imgs[0]['title']} + {end_eagle_imgs[1]['title']}")
        
        end_comfyui_url = self._call_remix_api(
            end_eagle_imgs[0]['image_url'],
            end_eagle_imgs[1]['image_url']
        )
        
        # Upload end image to storage
        end_image_url = self._upload_remix_to_storage(end_comfyui_url, 0, "end")
        
        # Store end image in database
        self.chain_manager.store_remix_image(
            end_image_url,
            [end_eagle_imgs[0]['image_url'], end_eagle_imgs[1]['image_url']],
            0,
            "end",
            [end_eagle_imgs[0]['title'], end_eagle_imgs[1]['title']]
        )
        
        print(f"✅ Generated initial images")
        return start_image_url, end_image_url
    
    def generate_chain_segment(self, iteration: int, start_image_url: str) -> tuple[str, str]:
        """
        Generate one segment of the chain (new end image + f2f video).
        
        Args:
            iteration: Current iteration number (1-based)
            start_image_url: Start image URL for this segment
            
        Returns:
            Tuple of (f2f_video_url, new_end_image_url)
        """
        print(f"\n🔗 Generating chain segment {iteration}/{self.iterations}")
        
        # Generate new end image
        end_eagle_imgs = self.eagle_pool.select_2_different_random_images()
        print(f"  End sources: {end_eagle_imgs[0]['title']} + {end_eagle_imgs[1]['title']}")
        
        end_comfyui_url = self._call_remix_api(
            end_eagle_imgs[0]['image_url'],
            end_eagle_imgs[1]['image_url']
        )
        
        # Upload end image to storage
        end_image_url = self._upload_remix_to_storage(end_comfyui_url, iteration, "end")
        
        # Store end image in database
        self.chain_manager.store_remix_image(
            end_image_url,
            [end_eagle_imgs[0]['image_url'], end_eagle_imgs[1]['image_url']],
            iteration,
            "end",
            [end_eagle_imgs[0]['title'], end_eagle_imgs[1]['title']]
        )
        
        # Generate f2f video
        f2f_video_url = self._call_f2f_api(start_image_url, end_image_url)
        
        # Store f2f video in database
        self.chain_manager.store_f2f_video(
            f2f_video_url,
            start_image_url,
            end_image_url,
            iteration
        )
        
        # Save checkpoint
        self.progress.checkpoint({
            "iteration": iteration,
            "start_image": start_image_url,
            "end_image": end_image_url,
            "f2f_video": f2f_video_url
        })
        
        print(f"✅ Completed segment {iteration}")
        return f2f_video_url, end_image_url
    
    def generate_full_chain(self) -> str:
        """
        Generate the complete video chain.
        
        Returns:
            URL of final stitched video
        """
        try:
            # Create session
            session_id = self.chain_manager.create_session(self.session_params)
            print(f"📝 Created session: {session_id}")
            
            # Start progress tracking
            self.progress.start()
            
            # Step 1: Generate initial images
            start_image_url, end_image_url = self.generate_initial_images()
            self.current_start_image = start_image_url
            self.progress.step("Generated initial start and end images")
            
            # Generate first f2f video
            first_f2f_url = self._call_f2f_api(start_image_url, end_image_url)
            self.chain_manager.store_f2f_video(first_f2f_url, start_image_url, end_image_url, 1)
            self.f2f_videos.append({"url": first_f2f_url, "iteration": 1})
            
            # Current start becomes the end image for next iteration
            current_start = end_image_url
            
            # Step 2-N: Generate remaining segments
            for iteration in range(2, self.iterations + 1):
                f2f_url, new_end_url = self.generate_chain_segment(iteration, current_start)
                self.f2f_videos.append({"url": f2f_url, "iteration": iteration})
                current_start = new_end_url  # End becomes start for next iteration
                self.progress.step(f"Generated segment {iteration}")
            
            # Final step: Stitch all videos together
            print(f"\n🎬 Stitching {len(self.f2f_videos)} video segments...")
            
            video_urls = [video["url"] for video in self.f2f_videos]
            
            with VideoStitcher() as stitcher:
                final_video_path = stitcher.stitch_videos(
                    video_urls, 
                    method=self.stitch_method,
                    output_filename=f"chain_video_{session_id}.mp4"
                )
                
                print(f"✅ Final video created: {final_video_path}")
                
                # Upload final video to Supabase Storage
                final_video_url = self._upload_video_to_storage(final_video_path, session_id)
                
                # Store final video in database with storage URL
                self.chain_manager.store_final_video(
                    final_video_url,  # Now using storage URL
                    video_urls,
                    self.session_params
                )
            
            self.progress.step("Stitched final video")
            
            # Complete session
            self.chain_manager.complete_session()
            
            # Print final summary
            self.progress.print_summary()
            print(f"\n🎉 Chain video generation completed!")
            print(f"   Session ID: {session_id}")
            print(f"   Segments: {len(self.f2f_videos)}")
            print(f"   Final video: {final_video_url}")
            
            return final_video_url
            
        except Exception as e:
            self.chain_manager.fail_session(str(e))
            raise Exception(f"Chain video generation failed: {e}")


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Generate chained videos using remix + f2f workflows")
    
    # Source selection - either tags or folder (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--tags", 
                             help="Space-separated list of Eagle image tags")
    source_group.add_argument("--folder", 
                             help="Eagle folder name to filter by")
    
    parser.add_argument("--iterations", type=int, required=True,
                       help="Number of f2f video segments to create")
    parser.add_argument("--width", type=int, default=720,
                       help="Video width in pixels (default: 720)")
    parser.add_argument("--height", type=int, default=720,
                       help="Video height in pixels (default: 720)")
    parser.add_argument("--frames", type=int, default=81,
                       help="Number of frames per f2f video (default: 81)")
    parser.add_argument("--stitch-method", choices=["direct", "crossfade"], default="direct",
                       help="Method for stitching videos together (default: direct)")
    parser.add_argument("--enable-qwen-captioner", action="store_true", default=False,
                       help="Enable Qwen captioner for motion prompts (default: disabled)")
    parser.add_argument("--prompt", type=str, default="",
                       help="Custom prompt when Qwen captioner is disabled (default: empty string)")
    parser.add_argument("--resume", type=str,
                       help="Resume from existing session ID")
    parser.add_argument("--status", action="store_true",
                       help="Show status of all chain video sessions")
    
    args = parser.parse_args()
    
    if args.status:
        # TODO: Implement status checking once automation_sessions table exists
        print("📊 Chain video session status (not yet implemented)")
        return
    
    if args.resume:
        # TODO: Implement resume functionality
        print(f"🔄 Resume functionality not yet implemented for session: {args.resume}")
        return
    
    try:
        # Parse source selection
        tags = args.tags.split() if args.tags else None
        folder = args.folder
        
        # Validate parameters
        if args.iterations < 1:
            raise ValueError("Iterations must be at least 1")
        if args.width < 1 or args.height < 1:
            raise ValueError("Width and height must be positive")
        if args.frames < 1:
            raise ValueError("Frame length must be positive")
        
        # Create and run generator
        generator = ChainVideoGenerator(
            tags=tags,
            folder=folder,
            iterations=args.iterations,
            width=args.width,
            height=args.height,
            frame_length=args.frames,
            stitch_method=args.stitch_method,
            enable_qwen_captioner=args.enable_qwen_captioner,
            prompt=args.prompt
        )
        
        # Print configuration
        print("🎬 Chain Video Generator")
        if tags:
            print(f"   Tags: {', '.join(tags)}")
        if folder:
            print(f"   Folder: {folder}")
        print(f"   Iterations: {args.iterations}")
        print(f"   Resolution: {args.width}x{args.height}")
        print(f"   Frame length: {args.frames}")
        print(f"   Stitch method: {args.stitch_method}")
        print(f"   Qwen captioner: {'enabled' if args.enable_qwen_captioner else 'disabled'}")
        if not args.enable_qwen_captioner and args.prompt:
            print(f"   Custom prompt: {args.prompt}")
        elif not args.enable_qwen_captioner:
            print(f"   Custom prompt: (empty)")
        
        # Check Eagle pool
        pool_stats = generator.eagle_pool.get_pool_stats()
        print(f"   Image pool: {pool_stats['total_images']} images")
        
        if pool_stats['total_images'] < 4:
            print("⚠️  Warning: Small image pool may result in repetitive content")
        
        # Generate the chain
        final_video_url = generator.generate_full_chain()
        
        print(f"\n🎯 Success! Final video: {final_video_url}")
        
    except KeyboardInterrupt:
        print("\n❌ Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Add missing import
    import time
    main()