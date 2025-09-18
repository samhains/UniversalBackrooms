#!/usr/bin/env python3
"""
Simple script to process one pending video at a time using MCP ComfyUI
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path to import supabase client
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))
from core.supabase_client import get_supabase_client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_one_pending_video() -> Optional[Dict[str, Any]]:
    """Get the first pending video from video_generation_results table"""
    try:
        client = get_supabase_client()
        result = client.table("video_generation_results").select(
            "id", "source_report", "prompt_text", "audio_prompt"
        ).eq("generation_status", "pending").order("id").limit(1).execute()
        
        if result.data:
            logger.info(f"Found pending video: {result.data[0]['id']}")
            return result.data[0]
        else:
            logger.info("No pending videos found")
            return None
    except Exception as e:
        logger.error(f"Failed to get pending video: {e}")
        return None

def update_video_status(video_id: int, status: str, video_url: str = None, failure_reason: str = None):
    """Update video status in database"""
    try:
        client = get_supabase_client()
        
        update_data = {
            "generation_status": status,
            "last_updated": datetime.now(datetime.UTC).isoformat()
        }
        
        if status == "success" and video_url:
            update_data["video_url"] = video_url
            update_data["failure_reason"] = None
        elif status == "failed" and failure_reason:
            update_data["failure_reason"] = failure_reason
            update_data["video_url"] = None
            
        result = client.table("video_generation_results").update(update_data).eq("id", video_id).execute()
        
        if result.data:
            logger.info(f"✅ Updated video {video_id} status to: {status}")
            return True
        else:
            logger.warning(f"❌ No rows updated for video: {video_id}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to update video status: {e}")
        return False

def process_one_video():
    """Process one pending video"""
    logger.info("Looking for pending videos...")
    
    video = get_one_pending_video()
    if not video:
        return False
        
    video_id = video["id"]
    prompt_text = video["prompt_text"]
    
    logger.info(f"Processing video {video_id}")
    logger.info(f"Prompt: {prompt_text[:100]}...")
    
    # Mark as processing first (optional status update)
    logger.info("📹 Video is ready for manual processing with MCP tool")
    logger.info("Run this command in Claude CLI:")
    logger.info(f'mcp__comfyui__generate_video "{{"prompt": "{prompt_text}"}}"')
    
    # For now, just log the information
    print("\n" + "="*80)
    print(f"VIDEO ID: {video_id}")
    print(f"SOURCE: {video.get('source_report', 'Unknown')}")
    print(f"PROMPT: {prompt_text}")
    print(f"AUDIO PROMPT: {video.get('audio_prompt', 'None')}")
    print("="*80)
    print("\nTo generate this video, run:")
    print(f'just mcp-generate-video {video_id} "{prompt_text[:50]}..."')
    
    return True

if __name__ == "__main__":
    if process_one_video():
        logger.info("✅ Found pending video - see details above")
    else:
        logger.info("✅ No pending videos to process")
