#!/usr/bin/env python3
"""
Eagle Images Semantic Search

Simple script to search your eagle_images database using semantic similarity.
Converts text queries to OpenAI embeddings and finds similar images.

Usage:
    python search_eagle_images.py "slotmachine cruise ship" --limit 10
    python search_eagle_images.py "sunset beach photography" --limit 5 --min-similarity 0.7
"""

import os
import argparse
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import openai
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()


def get_supabase_client() -> Client:
    """Initialize Supabase client using environment variables.

    Requires SUPABASE_URL and one of SUPABASE_KEY, SUPABASE_ANON_KEY, or SUPABASE_SERVICE_ROLE_KEY.
    """
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )
    if not url or not key:
        raise ValueError(
            "Missing Supabase configuration: set SUPABASE_URL and SUPABASE_KEY (or SUPABASE_ANON_KEY/SUPABASE_SERVICE_ROLE_KEY) in .env"
        )
    return create_client(url, key)


def get_openai_client():
    """Initialize OpenAI client using environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return openai.OpenAI(api_key=api_key)


def text_to_embedding(text: str, client) -> List[float]:
    """Convert text to OpenAI embedding vector."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        raise Exception(f"Failed to generate embedding: {e}")


def search_images_semantic(
    query: str,
    limit: int = 20,
    min_similarity: float = 0.0,
    folders: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Search eagle_images using semantic similarity.

    Args:
        query: Text query to search for
        limit: Maximum number of results to return
        min_similarity: Minimum similarity score (0.0 to 1.0)
        folders: Optional list of folders to filter by

    Returns:
        List of image records with similarity scores
    """

    # Initialize clients
    supabase = get_supabase_client()
    openai_client = get_openai_client()

    # Generate embedding for query
    print(f"🔍 Generating embedding for: '{query}'")
    embedding = text_to_embedding(query, openai_client)

    # Call the database function
    print(f"🔎 Searching database...")
    try:
        result = supabase.rpc(
            'search_images_with_embedding',
            {
                'query_embedding': embedding,
                'match_limit': limit,
                'folder_filter': folders,
                'similarity_threshold': min_similarity
            }
        ).execute()

        images = result.data
        print(f"✅ Found {len(images)} results")

        return images

    except Exception as e:
        raise Exception(f"Database search failed: {e}")


def format_results(images: List[Dict[str, Any]], show_urls: bool = True) -> None:
    """Pretty print search results."""
    if not images:
        print("❌ No results found. Try lowering --min-similarity or using different keywords.")
        return

    print(f"\n📸 Found {len(images)} images:\n")

    for i, img in enumerate(images, 1):
        title = img.get('title') or 'Untitled'
        similarity = img.get('similarity_score', 0)
        eagle_id = img.get('eagle_id')
        tags = img.get('tags', [])
        folders = img.get('folders', [])

        print(f"{i:2d}. {title}")
        print(f"    📊 Similarity: {similarity:.3f}")
        print(f"    🏷️  Tags: {', '.join(tags[:5])}{' ...' if len(tags) > 5 else ''}")

        if folders:
            print(f"    📁 Folders: {', '.join(folders)}")

        if show_urls and img.get('image_url'):
            print(f"    🔗 URL: {img['image_url']}")
        elif show_urls and img.get('storage_path'):
            base_url = "https://idyoveanwiuwcvgxijtz.supabase.co/storage/v1/object/public/eagle-images/"
            print(f"    🔗 URL: {base_url}{img['storage_path']}")

        print(f"    🆔 ID: {eagle_id}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Search eagle_images database using semantic similarity")
    parser.add_argument("query", help="Text query to search for")
    parser.add_argument("--limit", "-l", type=int, default=10,
                       help="Maximum number of results (default: 10)")
    parser.add_argument("--min-similarity", "-s", type=float, default=0.0,
                       help="Minimum similarity score 0.0-1.0 (default: 0.0)")
    parser.add_argument("--folders", "-f", nargs="+",
                       help="Filter by specific folders")
    parser.add_argument("--no-urls", action="store_true",
                       help="Don't display image URLs")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON")

    args = parser.parse_args()

    try:
        # Perform search
        results = search_images_semantic(
            query=args.query,
            limit=args.limit,
            min_similarity=args.min_similarity,
            folders=args.folders
        )

        # Output results
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            format_results(results, show_urls=not args.no_urls)

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
