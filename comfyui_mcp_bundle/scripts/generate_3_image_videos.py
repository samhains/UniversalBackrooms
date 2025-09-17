#!/usr/bin/env python3
"""
3-Image Video Generation Script
Processes pending 3-image video generation requests from the database using ComfyUI.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Global generation parameters (set by command line args)
generation_params = {
    'width': 1280,
    'height': 720,
    'frames': 150
}

def get_supabase_client() -> Client:
    """Initialize Supabase client"""
    if not SUPABASE_KEY:
        raise ValueError("SUPABASE_KEY environment variable not set")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def generate_3_image_video_with_comfyui(record: dict) -> dict:
    """
    Generate video using 3 reference images via ComfyUI HTTP API
    Following the pattern from process_pending_videos.py
    
    Args:
        record: Database record with generation parameters
        
    Returns:
        dict: Result with video_url or error information
    """
    import requests
    
    try:
        print(f"Generating 3-image video with prompt: {record['prompt_text'][:100]}...")
        
        # Use the 3-image video generation endpoint
        api_url = "http://100.75.77.33:9000/generate_3_image_video"
        
        # Use command-line parameters if provided, otherwise fall back to database values, then defaults
        payload = {
            "image1_url": record["image1_url"],
            "image2_url": record["image2_url"], 
            "image3_url": record["image3_url"],
            "frame_length": generation_params['frames'] if generation_params['frames'] != 150 else (record["frame_length"] or 150),
            "width": generation_params['width'] if generation_params['width'] != 1280 else (record["width"] or 1280),
            "height": generation_params['height'] if generation_params['height'] != 720 else (record["height"] or 720)
        }
        
        if record.get("audio_prompt"):
            payload["audio_prompt"] = record["audio_prompt"]
            print(f"Including audio prompt: {record['audio_prompt'][:50]}...")
        
        print("Calling ComfyUI HTTP API for 3-image video generation...")
        response = requests.post(api_url, json=payload, timeout=1200)  # 20 minute timeout
        
        if response.status_code == 200:
            result = response.json()
            if "video_url" in result:
                return {
                    "success": True,
                    "video_url": result["video_url"],
                    "message": "3-image video generated successfully via HTTP API"
                }
            elif "error" in result:
                return {
                    "success": False,
                    "error": f"ComfyUI error: {result['error']}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Unexpected response format: {result}"
                }
        else:
            return {
                "success": False,
                "error": f"HTTP error {response.status_code}: {response.text}"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"3-image video generation failed: {str(e)}"
        }

def process_pending_videos(limit: int = None) -> None:
    """Process pending 3-image video generation requests"""
    supabase = get_supabase_client()
    
    # Fetch pending records
    query = supabase.table("image_video_generation_results").select("*").eq("generation_status", "pending").order("created_at")
    
    if limit:
        query = query.limit(limit)
    
    response = query.execute()
    pending_records = response.data
    
    print(f"Found {len(pending_records)} pending 3-image video generation requests")
    
    for i, record in enumerate(pending_records, 1):
        print(f"\nProcessing {i}/{len(pending_records)}: ID {record['id']}")
        print(f"Prompt: {record['prompt_text'][:100]}...")
        
        # Extract filenames for display
        img1_name = record['image1_url'].split('/')[-1] if record['image1_url'] else 'N/A'
        img2_name = record['image2_url'].split('/')[-1] if record['image2_url'] else 'N/A'
        img3_name = record['image3_url'].split('/')[-1] if record['image3_url'] else 'N/A'
        print(f"Images: {img1_name}, {img2_name}, {img3_name}")
        
        # Generate video
        start_time = datetime.now()
        result = generate_3_image_video_with_comfyui(record)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Update database
        if result["success"]:
            print(f"✅ Video generated successfully: {result['video_url']}")
            supabase.table("image_video_generation_results").update({
                "generation_status": "success",
                "video_url": result["video_url"],
                "generation_duration_seconds": int(duration),
                "last_updated": datetime.now().isoformat()
            }).eq("id", record["id"]).execute()
        else:
            print(f"❌ Generation failed: {result['error']}")
            supabase.table("image_video_generation_results").update({
                "generation_status": "failed", 
                "failure_reason": result["error"],
                "generation_duration_seconds": int(duration),
                "last_updated": datetime.now().isoformat()
            }).eq("id", record["id"]).execute()

def process_single_video(video_id: int) -> None:
    """Process a single 3-image video generation request by ID"""
    supabase = get_supabase_client()
    
    response = supabase.table("image_video_generation_results").select("*").eq("id", video_id).execute()
    
    if not response.data:
        print(f"❌ No record found with ID {video_id}")
        return
    
    record = response.data[0]
    
    if record["generation_status"] not in ["pending", "failed"]:
        print(f"⚠️ Record {video_id} status is '{record['generation_status']}', not 'pending' or 'failed'")
        print("Use --force to regenerate anyway")
        return
    
    print(f"Processing 3-image video ID {video_id}")
    print(f"Prompt: {record['prompt_text']}")
    
    img1_name = record['image1_url'].split('/')[-1] if record['image1_url'] else 'N/A'
    img2_name = record['image2_url'].split('/')[-1] if record['image2_url'] else 'N/A'
    img3_name = record['image3_url'].split('/')[-1] if record['image3_url'] else 'N/A'
    print(f"Images: {img1_name}, {img2_name}, {img3_name}")
    
    start_time = datetime.now()
    result = generate_3_image_video_with_comfyui(record)
    duration = (datetime.now() - start_time).total_seconds()
    
    if result["success"]:
        print(f"✅ Video generated successfully: {result['video_url']}")
        supabase.table("image_video_generation_results").update({
            "generation_status": "success",
            "video_url": result["video_url"],
            "generation_duration_seconds": int(duration),
            "last_updated": datetime.now().isoformat()
        }).eq("id", video_id).execute()
    else:
        print(f"❌ Generation failed: {result['error']}")
        supabase.table("image_video_generation_results").update({
            "generation_status": "failed",
            "failure_reason": result["error"],
            "generation_duration_seconds": int(duration), 
            "last_updated": datetime.now().isoformat()
        }).eq("id", video_id).execute()

def main():
    parser = argparse.ArgumentParser(description="Generate 3-image videos using ComfyUI")
    parser.add_argument("--id", type=int, help="Process specific video by ID")
    parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if not pending")
    parser.add_argument("--width", type=int, default=1280, help="Video width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Video height (default: 720)")
    parser.add_argument("--frames", type=int, default=150, help="Frame length (default: 150)")
    
    args = parser.parse_args()
    
    # Store generation parameters globally so they can be used in the generation function
    global generation_params
    generation_params = {
        'width': args.width,
        'height': args.height,
        'frames': args.frames
    }
    
    if args.id:
        process_single_video(args.id)
    else:
        process_pending_videos(args.limit)

if __name__ == "__main__":
    main()