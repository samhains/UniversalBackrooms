#!/usr/bin/env python3
"""
Test 3-Image Video Generation using MCP ComfyUI tools
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Dict, Any

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

def get_random_eagle_images(count: int = 3) -> List[Dict[str, Any]]:
    """Get random Eagle images from database"""
    supabase = get_supabase_client()
    
    # Get random images using RPC function
    try:
        response = supabase.rpc("get_random_eagle_images", {"count": count}).execute()
        if response.data:
            return response.data
    except:
        pass
    
    # Fallback: get first N images
    response = supabase.table("eagle_images").select(
        "id, eagle_id, title, storage_path, tags, width, height"
    ).limit(count).execute()
    
    return response.data

def search_eagle_images_by_tags(tags: List[str], limit: int = 20) -> List[Dict[str, Any]]:
    """Search Eagle images by tags"""
    supabase = get_supabase_client()
    
    # Use PostgreSQL array overlap operator
    tags_array = "{" + ",".join(tags) + "}"
    
    response = supabase.table("eagle_images").select(
        "id, eagle_id, title, storage_path, tags, width, height"
    ).filter(
        "tags", "ov", tags_array
    ).limit(limit).execute()
    
    return response.data

def build_image_urls(images: List[Dict[str, Any]]) -> List[str]:
    """Convert storage paths to full Supabase URLs"""
    return [f"{SUPABASE_URL}/storage/v1/object/public/eagle-images/{img['storage_path']}" 
            for img in images]

def test_random_3_image_video():
    """Test generating video with 3 random images"""
    print("🎬 Testing 3-image video generation with random images...")
    
    # Get 3 random images
    images = get_random_eagle_images(3)
    if len(images) < 3:
        print("❌ Not enough images in database")
        return
    
    # Build URLs
    image_urls = build_image_urls(images)
    
    # Display selected images
    print("\n📸 Selected images:")
    for i, img in enumerate(images, 1):
        print(f"  {i}. {img['title']} ({img['eagle_id']})")
        print(f"     Size: {img['width']}x{img['height']}")
        print(f"     Tags: {', '.join(img['tags'][:3])}...")
    
    print(f"\n🔗 Image URLs:")
    for i, url in enumerate(image_urls, 1):
        print(f"  {i}. {url}")
    
    print("\n✨ You can now use these URLs with the MCP ComfyUI generate_3_image_video tool!")
    
    return {
        "image_urls": image_urls,
        "images": images
    }

def test_tagged_3_image_video(tags: List[str]):
    """Test generating video with images matching specific tags"""
    print(f"🎬 Testing 3-image video generation with tags: {tags}")
    
    # Search for images with tags
    images = search_eagle_images_by_tags(tags, 20)
    if len(images) < 3:
        print(f"❌ Not enough images found with tags {tags}. Found {len(images)}")
        return
    
    # Take first 3 diverse images
    selected_images = images[:3]
    image_urls = build_image_urls(selected_images)
    
    # Display selected images
    print(f"\n📸 Selected images (from {len(images)} matches):")
    for i, img in enumerate(selected_images, 1):
        print(f"  {i}. {img['title']} ({img['eagle_id']})")
        print(f"     Size: {img['width']}x{img['height']}")
        print(f"     Matching tags: {[tag for tag in img['tags'] if tag in tags]}")
    
    print(f"\n🔗 Image URLs:")
    for i, url in enumerate(image_urls, 1):
        print(f"  {i}. {url}")
    
    print("\n✨ You can now use these URLs with the MCP ComfyUI generate_3_image_video tool!")
    
    return {
        "image_urls": image_urls,
        "images": selected_images,
        "search_tags": tags
    }

def main():
    """Main test function"""
    print("🧪 Testing 3-Image Video Generation with Eagle Images\n")
    
    try:
        # Test 1: Random images
        print("=" * 60)
        result1 = test_random_3_image_video()
        
        # Test 2: Search by tags
        print("\n" + "=" * 60)
        test_tags = ["anime", "character"]
        result2 = test_tagged_3_image_video(test_tags)
        
        # Test 3: Different tags
        print("\n" + "=" * 60)
        test_tags2 = ["nature", "landscape"]
        result3 = test_tagged_3_image_video(test_tags2)
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed!")
        print("\nNext steps:")
        print("1. Use the image URLs above with mcp__comfyui__generate_3_image_video")
        print("2. Test the existing scripts:")
        print("   - python scripts/create_3_image_video_request.py --help")
        print("   - python scripts/generate_3_image_videos.py --help")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()