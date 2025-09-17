#!/usr/bin/env python3
"""
Create 3-Image Video Generation Request
Creates new requests using Eagle images with automatic selection or manual URLs.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def get_supabase_client() -> Client:
    """Initialize Supabase client"""
    if not SUPABASE_KEY:
        raise ValueError("SUPABASE_KEY environment variable not set")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_eagle_image_url(storage_path: str) -> str:
    """Convert Eagle storage path to full Supabase URL"""
    return f"{SUPABASE_URL}/storage/v1/object/public/eagle-images/{storage_path}"

def search_eagle_images_by_tags(tags: List[str], limit: int = 50) -> List[Dict[str, Any]]:
    """Search Eagle images by tags using OR logic (any tag matches)"""
    supabase = get_supabase_client()
    
    # Use PostgreSQL array overlap operator (&&) to find images with ANY matching tags
    tags_array = "{" + ",".join(tags) + "}"
    
    response = supabase.table("eagle_images").select(
        "id, eagle_id, title, storage_path, tags, width, height, color_palette"
    ).filter(
        "tags", "ov", tags_array  # ov = overlap (any elements in common)
    ).limit(limit).execute()
    
    return response.data

def search_eagle_images_by_query(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search Eagle images by text query in title/tags"""
    supabase = get_supabase_client()
    
    # Search in title and convert tags array to text for search
    response = supabase.table("eagle_images").select(
        "id, eagle_id, title, storage_path, tags, width, height, color_palette"
    ).or_(
        f"title.ilike.%{query}%,tags.cs.{{{query}}}"
    ).limit(limit).execute()
    
    return response.data

def get_random_eagle_images(count: int = 3) -> List[Dict[str, Any]]:
    """Get random Eagle images"""
    supabase = get_supabase_client()
    
    # Get random images using PostgreSQL random()
    response = supabase.rpc("get_random_eagle_images", {"count": count}).execute()
    
    if response.data:
        return response.data
    else:
        # Fallback: get first N images
        response = supabase.table("eagle_images").select(
            "id, eagle_id, title, storage_path, tags, width, height, color_palette"
        ).limit(count).execute()
        return response.data

def select_diverse_images(images: List[Dict[str, Any]], count: int = 3, used_image_ids: set = None) -> List[Dict[str, Any]]:
    """
    Select diverse images based on aspect ratio and ensure no duplicates
    """
    if used_image_ids is None:
        used_image_ids = set()
    
    # Filter out already used images
    available_images = [img for img in images if img['eagle_id'] not in used_image_ids]
    
    if len(available_images) < count:
        print(f"⚠️ Only {len(available_images)} unused images available, need {count}")
        if len(available_images) == 0:
            return []
    
    selected = []
    import random
    
    # Shuffle for randomness
    random.shuffle(available_images)
    
    for img in available_images:
        if len(selected) >= count:
            break
            
        # Calculate aspect ratio
        aspect_ratio = img['width'] / img['height'] if img['height'] > 0 else 1.0
        
        # Check diversity (but be less strict when we need more images)
        is_diverse = True
        if len(selected) > 0:
            for selected_img in selected:
                selected_ratio = selected_img['width'] / selected_img['height'] if selected_img['height'] > 0 else 1.0
                
                # Too similar aspect ratio - be less strict if we need more images
                diversity_threshold = 0.2 if len(available_images) > count * 2 else 0.1
                if abs(aspect_ratio - selected_ratio) < diversity_threshold:
                    is_diverse = False
                    break
        
        if is_diverse or len(selected) == 0:
            selected.append(img)
    
    # Fill remaining slots with any remaining images if needed
    while len(selected) < count and len(selected) < len(available_images):
        for img in available_images:
            if img not in selected:
                selected.append(img)
                break
    
    return selected[:count]

def create_video_request(
    prompt: str,
    audio_prompt: str,
    image_urls: List[str],
    eagle_ids: Optional[List[str]] = None,
    source_report: str = "manual_request",
    frame_length: int = 150,
    width: int = 1280,
    height: int = 720,
    target_views: int = 50000,
    category: str = "user_generated"
) -> int:
    """
    Create a new 3-image video generation request
    
    Returns:
        int: ID of the created request
    """
    if len(image_urls) != 3:
        raise ValueError("Exactly 3 image URLs are required")
    
    supabase = get_supabase_client()
    
    # Create database record
    request_data = {
        "source_report": source_report,
        "prompt_text": prompt,
        "audio_prompt": audio_prompt,
        "image1_url": image_urls[0],
        "image2_url": image_urls[1],
        "image3_url": image_urls[2],
        "generation_status": "pending",
        "frame_length": frame_length,
        "width": width,
        "height": height,
        "target_views": target_views,
        "prompt_category": category
    }
    
    # Add Eagle IDs if provided
    if eagle_ids and len(eagle_ids) == 3:
        request_data.update({
            "eagle_image1_id": eagle_ids[0],
            "eagle_image2_id": eagle_ids[1],
            "eagle_image3_id": eagle_ids[2]
        })
    
    response = supabase.table("image_video_generation_results").insert(request_data).execute()
    
    if response.data:
        request_id = response.data[0]['id']
        return request_id
    else:
        raise Exception("Failed to create request in database")

def create_request_from_eagle_search(
    prompt: str,
    audio_prompt: str,
    search_tags: List[str] = None,
    search_query: str = None,
    **kwargs
) -> int:
    """Create request using Eagle image search"""
    
    # Search for images
    if search_tags:
        print(f"Searching Eagle images by tags: {search_tags}")
        images = search_eagle_images_by_tags(search_tags)
    elif search_query:
        print(f"Searching Eagle images by query: {search_query}")
        images = search_eagle_images_by_query(search_query)
    else:
        print("Getting random Eagle images")
        images = get_random_eagle_images(10)
    
    if len(images) < 3:
        raise ValueError(f"Not enough images found. Need at least 3, found {len(images)}")
    
    # Select diverse images
    selected_images = select_diverse_images(images, 3)
    
    # Build URLs and Eagle IDs
    image_urls = [get_eagle_image_url(img['storage_path']) for img in selected_images]
    eagle_ids = [img['eagle_id'] for img in selected_images]
    
    # Create request
    request_id = create_video_request(
        prompt=prompt,
        audio_prompt=audio_prompt,
        image_urls=image_urls,
        eagle_ids=eagle_ids,
        **kwargs
    )
    
    print(f"✅ Created 3-image video request #{request_id}")
    print("Selected reference images:")
    for i, img in enumerate(selected_images, 1):
        print(f"  {i}. {img['title']} ({img['eagle_id']})")
        print(f"     Tags: {', '.join(img['tags'][:5])}{'...' if len(img['tags']) > 5 else ''}")
    
    return request_id

def main():
    parser = argparse.ArgumentParser(description="Create 3-image video generation request")
    parser.add_argument("prompt", help="Video motion description")
    parser.add_argument("audio_prompt", help="Audio description")
    
    # Image selection options
    parser.add_argument("--tags", nargs="+", help="Search Eagle images by tags")
    parser.add_argument("--query", help="Search Eagle images by text query")
    parser.add_argument("--urls", nargs=3, help="Use specific image URLs (must provide exactly 3)")
    parser.add_argument("--random", action="store_true", help="Use random Eagle images")
    
    # Generation parameters
    parser.add_argument("--frame-length", type=int, default=150, help="Number of frames")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--target-views", type=int, default=50000, help="Target view count")
    parser.add_argument("--category", default="user_generated", help="Prompt category")
    parser.add_argument("--source", default="manual_request", help="Source report identifier")
    
    args = parser.parse_args()
    
    try:
        if args.urls:
            # Use provided URLs directly
            request_id = create_video_request(
                prompt=args.prompt,
                audio_prompt=args.audio_prompt,
                image_urls=args.urls,
                source_report=args.source,
                frame_length=args.frame_length,
                width=args.width,
                height=args.height,
                target_views=args.target_views,
                category=args.category
            )
            print(f"✅ Created 3-image video request #{request_id}")
            print(f"Using provided URLs:")
            for i, url in enumerate(args.urls, 1):
                print(f"  {i}. {url}")
        
        else:
            # Use Eagle image search
            request_id = create_request_from_eagle_search(
                prompt=args.prompt,
                audio_prompt=args.audio_prompt,
                search_tags=args.tags,
                search_query=args.query,
                source_report=args.source,
                frame_length=args.frame_length,
                width=args.width,
                height=args.height,
                target_views=args.target_views,
                category=args.category
            )
        
        print(f"\nTo generate the video, run:")
        print(f"just generate-3-image-video {request_id}")
        
    except Exception as e:
        print(f"❌ Error creating request: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()