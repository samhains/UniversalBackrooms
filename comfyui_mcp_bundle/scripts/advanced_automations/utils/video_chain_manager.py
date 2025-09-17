#!/usr/bin/env python3
"""
Video Chain Manager

Handles database operations and session management for chained video generation.
Provides resume capability, progress tracking, and storage of intermediate results.
"""

import os
import sys
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
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


class VideoChainManager:
    """
    Manages database operations and session state for video chain generation.
    Handles storing intermediate results and provides resume capability.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize the video chain manager.
        
        Args:
            session_id: Optional existing session ID for resume capability
        """
        self.supabase = self._get_supabase_client()
        self.session_id = session_id or self._generate_session_id()
        self.session_data = None
        
        if session_id:
            self.load_session()
    
    def _get_supabase_client(self):
        """Get Supabase client using proper client factory."""
        return get_supabase_client()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"chain_video_{timestamp}_{short_uuid}"
    
    def create_session(self, parameters: Dict[str, Any]) -> str:
        """
        Create a new session for tracking. Uses existing tables for storage.
        
        Args:
            parameters: Session parameters (tags, iterations, dimensions, etc.)
            
        Returns:
            Session ID
        """
        print(f"✓ Created session: {self.session_id}")
        print(f"  Parameters: {parameters}")
        return self.session_id
    
    def load_session(self) -> Optional[Dict[str, Any]]:
        """
        Load an existing session by finding existing segments in database.
        
        Returns:
            Session summary if found, None otherwise
        """
        try:
            # Check for existing video segments with this session_id
            response = self.supabase.table("video_generation_results").select(
                "id, video_url, generation_settings"
            ).like("source_report", f"%{self.session_id}%").execute()
            
            if response.data:
                print(f"✓ Found existing session with {len(response.data)} segments")
                return {
                    "session_id": self.session_id,
                    "segments_count": len(response.data),
                    "segments": response.data
                }
            else:
                print(f"No existing segments found for session: {self.session_id}")
                return None
                
        except Exception as e:
            print(f"Warning: Failed to load session {self.session_id}: {e}")
            return None
    
    def update_session_progress(self, iteration: int, completed_segments: List[str], 
                              current_start_image: Optional[str] = None) -> None:
        """
        Update session progress. Progress is tracked through database records.
        
        Args:
            iteration: Current iteration number
            completed_segments: List of completed video segment URLs
            current_start_image: Current start image URL for next iteration
        """
        print(f"✓ Session progress: iteration {iteration}, {len(completed_segments)} segments completed")
    
    def store_remix_image(self, image_url: str, source_images: List[str], 
                         iteration: int, role: str, source_titles: List[str] = None) -> int:
        """
        Store a remix image in the database.
        
        Args:
            image_url: URL of the generated remix image
            source_images: URLs of the source images used
            iteration: Current iteration number
            role: Role of the image ('start' or 'end')
            source_titles: Optional titles of source images for better description
            
        Returns:
            Database record ID
        """
        try:
            # Create descriptive prompt text
            if source_titles and len(source_titles) >= 2:
                prompt_text = f"Remix of '{source_titles[0]}' and '{source_titles[1]}' - Chain video {role} image for iteration {iteration}"
            else:
                prompt_text = f"Remix of two Eagle images - Chain video {role} image for iteration {iteration}"
            
            image_data = {
                "image_url": image_url,
                "model_used": "flux-redux",
                "prompt_text": prompt_text,
                "generation_status": "success",
                "generation_settings": json.dumps({
                    "type": "remix",
                    "source_images": source_images,
                    "session_id": self.session_id,
                    "iteration": iteration,
                    "role": role,
                    "source_titles": source_titles or []
                }),
                "source_report": f"chain_video_{self.session_id}"
            }
            
            response = self.supabase.table("image_generation_results").insert(image_data).execute()
            
            if response.data:
                record_id = response.data[0]["id"]
                print(f"✓ Stored {role} remix image for iteration {iteration} (ID: {record_id})")
                return record_id
            else:
                raise Exception("No data returned from insert")
                
        except Exception as e:
            raise Exception(f"Failed to store remix image: {e}")
    
    def store_f2f_video(self, video_url: str, start_image: str, end_image: str, 
                       iteration: int, start_title: str = "start image", end_title: str = "end image") -> int:
        """
        Store an f2f video segment in the database.
        
        Args:
            video_url: URL of the generated f2f video
            start_image: Start image URL
            end_image: End image URL  
            iteration: Current iteration number
            start_title: Description of start image
            end_title: Description of end image
            
        Returns:
            Database record ID
        """
        try:
            video_data = {
                "video_url": video_url,
                "prompt_text": f"F2F transition from {start_title} to {end_title} - Chain video segment {iteration}",
                "generation_status": "success",
                "source_report": f"chain_video_{self.session_id}",
                "generation_settings": json.dumps({
                    "type": "f2f",
                    "start_image": start_image,
                    "end_image": end_image,
                    "session_id": self.session_id,
                    "iteration": iteration,
                    "start_title": start_title,
                    "end_title": end_title
                })
            }
            
            response = self.supabase.table("video_generation_results").insert(video_data).execute()
            
            if response.data:
                record_id = response.data[0]["id"]
                print(f"✓ Stored f2f video segment {iteration} (ID: {record_id})")
                return record_id
            else:
                raise Exception("No data returned from insert")
                
        except Exception as e:
            raise Exception(f"Failed to store f2f video: {e}")
    
    def store_final_video(self, video_url: str, segment_urls: List[str], 
                         parameters: Dict[str, Any]) -> int:
        """
        Store the final compiled video in the database.
        
        Args:
            video_url: URL of the final compiled video
            segment_urls: List of segment video URLs that were stitched
            parameters: Original generation parameters
            
        Returns:
            Database record ID
        """
        try:
            # Store final compilation in video_generation_results
            video_data = {
                "video_url": video_url,
                "prompt_text": f"Chain video compilation - {len(segment_urls)} segments",
                "generation_status": "success",
                "source_report": f"chain_video_final_{self.session_id}",
                "generation_settings": json.dumps({
                    "type": "compilation",
                    "segment_count": len(segment_urls),
                    "segment_urls": segment_urls,
                    "session_id": self.session_id,
                    "parameters": parameters
                })
            }
            
            response = self.supabase.table("video_generation_results").insert(video_data).execute()
            
            if response.data:
                record_id = response.data[0]["id"]
                print(f"✅ Stored final compiled video (ID: {record_id})")
                return record_id
            else:
                raise Exception("No data returned from insert")
                
        except Exception as e:
            raise Exception(f"Failed to store final video: {e}")
    
    def complete_session(self) -> None:
        """Mark the session as completed."""
        print(f"✅ Session {self.session_id} completed successfully")
    
    def fail_session(self, error_message: str) -> None:
        """Mark the session as failed."""
        print(f"❌ Session {self.session_id} failed: {error_message}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session from existing database records."""
        try:
            # Get video segments for this session
            video_response = self.supabase.table("video_generation_results").select(
                "id, video_url, generation_settings, created_at"
            ).like("source_report", f"%{self.session_id}%").execute()
            
            # Get image generations for this session
            image_response = self.supabase.table("image_generation_results").select(
                "id, image_url, generation_settings, created_at"
            ).like("source_report", f"%{self.session_id}%").execute()
            
            return {
                "session_id": self.session_id,
                "video_segments": len(video_response.data),
                "remix_images": len(image_response.data),
                "latest_activity": max(
                    [v.get("created_at", "") for v in video_response.data] +
                    [i.get("created_at", "") for i in image_response.data],
                    default="unknown"
                )
            }
        except Exception as e:
            return {"session_id": self.session_id, "status": "error", "error": str(e)}