#!/usr/bin/env python3
"""
Eagle Image Pool Manager

Handles loading and selecting images from the Eagle database for advanced automations.
Ensures no duplicate images are used in single remix operations while allowing
reuse across different iterations.
"""

import os
import sys
import random
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Add src to path to import supabase client directly
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'core'))

try:
    from supabase_client import get_supabase_client
except ImportError:
    print("Error: supabase client not found. Check project structure.")
    sys.exit(1)

# Load environment variables
load_dotenv()


class EagleImagePool:
    """
    Manages a pool of Eagle images for use in automation workflows.
    Provides intelligent selection methods to ensure diversity and avoid duplicates.
    """
    
    def __init__(self, tags: List[str] = None, folder: str = None, min_pool_size: int = 4):
        """
        Initialize the Eagle image pool.
        
        Args:
            tags: List of tags to filter images by (optional if folder provided)
            folder: Eagle folder name to filter by (optional if tags provided)
            min_pool_size: Minimum number of images required in pool
        """
        if not tags and not folder:
            raise ValueError("Either tags or folder must be provided")
        
        self.tags = tags if tags and isinstance(tags, list) else (tags.split() if tags else [])
        self.folder = folder
        self.min_pool_size = min_pool_size
        self.images = []
        self.supabase = self._get_supabase_client()
        self.load_images()
    
    def _get_supabase_client(self):
        """Get Supabase client using proper client factory."""
        return get_supabase_client()
    
    def load_images(self) -> None:
        """
        Load images from the Eagle database that match the specified tags or folder.
        Uses PostgreSQL array overlap operator to find images with any of the tags,
        or exact folder name matching for folder filtering.
        """
        try:
            query = self.supabase.table("eagle_images").select(
                "id, eagle_id, image_url, title, tags, width, height, storage_path, folders"
            )
            
            if self.folder:
                # Filter by folder name (folders is an array, check if it contains the folder)
                response = query.contains("folders", [self.folder]).execute()
                filter_description = f"folder: {self.folder}"
            else:
                # Format tags as PostgreSQL array literal for overlap query
                tags_formatted = "{" + ",".join(f'"{tag}"' for tag in self.tags) + "}"
                # Query for images that have any of the specified tags
                response = query.filter("tags", "ov", tags_formatted).execute()
                filter_description = f"tags: {', '.join(self.tags)}"
            
            # Filter out images without image_url
            self.images = [
                img for img in response.data 
                if img.get('image_url') and img['image_url'].strip()
            ]
            
            print(f"✓ Loaded {len(self.images)} images matching {filter_description}")
            
            if len(self.images) < self.min_pool_size:
                raise ValueError(
                    f"Insufficient images found. Got {len(self.images)}, need at least {self.min_pool_size}. "
                    f"Try broadening your selection or adding more images to the database."
                )
                
        except Exception as e:
            raise Exception(f"Failed to load Eagle images: {e}")
    
    def select_2_different_random_images(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Select 2 different random images from the pool.
        Ensures no duplicate images are selected for a single remix operation.
        
        Returns:
            Tuple of two different image dictionaries
            
        Raises:
            ValueError: If pool has fewer than 2 images
        """
        if len(self.images) < 2:
            raise ValueError(
                f"Cannot select 2 different images. Pool only has {len(self.images)} images. "
                f"Need at least 2 images for remix operations."
            )
        
        # Select first image
        img1 = random.choice(self.images)
        
        # Select second image, ensuring it's different from first
        remaining = [img for img in self.images if img['id'] != img1['id']]
        img2 = random.choice(remaining)
        
        return img1, img2
    
    def get_random_image(self) -> Dict[str, Any]:
        """
        Get a single random image from the pool.
        
        Returns:
            Random image dictionary
            
        Raises:
            ValueError: If pool is empty
        """
        if not self.images:
            raise ValueError("Cannot select image from empty pool")
        
        return random.choice(self.images)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current image pool.
        
        Returns:
            Dictionary with pool statistics
        """
        if not self.images:
            return {
                "total_images": 0,
                "tags": self.tags,
                "dimensions": {},
                "file_types": {}
            }
        
        # Calculate dimension distribution
        dimensions = {}
        file_types = {}
        
        for img in self.images:
            # Dimension stats
            dim_key = f"{img.get('width', 'unknown')}x{img.get('height', 'unknown')}"
            dimensions[dim_key] = dimensions.get(dim_key, 0) + 1
            
            # File type stats
            if img.get('storage_path'):
                ext = img['storage_path'].split('.')[-1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
        
        return {
            "total_images": len(self.images),
            "tags": self.tags,
            "folder": self.folder,
            "dimensions": dimensions,
            "file_types": file_types,
            "sample_titles": [img.get('title', 'Untitled') for img in self.images[:5]]
        }
    
    def refresh_pool(self) -> None:
        """Reload the image pool from the database."""
        print("🔄 Refreshing image pool...")
        self.load_images()
    
    def __len__(self) -> int:
        """Return the number of images in the pool."""
        return len(self.images)
    
    def __bool__(self) -> bool:
        """Return True if pool has images."""
        return len(self.images) > 0