#!/usr/bin/env python3
"""
Eagle Images Semantic Search

Search the eagle_images database using either text prompts or example images.
Text search uses OpenAI embeddings; image search uses Google Vertex multimodal
embeddings that are stored in the image_embedding column.

Usage:
    python search_eagle_images.py "slotmachine cruise ship" --limit 10
    python search_eagle_images.py --image-url https://example.com/sample.jpg --limit 5
"""

import argparse
import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import openai
import requests
from dotenv import load_dotenv
from supabase import Client, create_client

from scripts.eagle.utils import row_to_image_url, storage_path_from_url

try:  # Vertex AI is optional until image search is requested
    import vertexai
    from vertexai.preview.vision_models import Image as VertexImage
    from vertexai.preview.vision_models import MultiModalEmbeddingModel
except ImportError:  # pragma: no cover - handled at runtime when needed
    vertexai = None
    VertexImage = None
    MultiModalEmbeddingModel = None

# Load environment variables from .env file
load_dotenv()


VERTEX_PROJECT = os.getenv("VERTEX_PROJECT")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION")
VERTEX_IMAGE_EMBED_MODEL = os.getenv("VERTEX_IMAGE_EMBED_MODEL", "multimodalembedding")


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


@lru_cache(maxsize=1)
def _get_vertex_embedding_model() -> "MultiModalEmbeddingModel":
    """Initialise and cache the Vertex multimodal embedding model."""
    if vertexai is None or MultiModalEmbeddingModel is None or VertexImage is None:
        raise ImportError(
            "google-cloud-aiplatform is required for image similarity search. "
            "Install it and ensure Vertex AI credentials are configured."
        )
    if not VERTEX_PROJECT or not VERTEX_LOCATION:
        raise ValueError(
            "Set VERTEX_PROJECT and VERTEX_LOCATION environment variables for Vertex AI."
        )
    vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    model_name = VERTEX_IMAGE_EMBED_MODEL or "multimodalembedding"
    return MultiModalEmbeddingModel.from_pretrained(model_name)


def _download_image_bytes(image_url: str) -> bytes:
    """Download raw image bytes from a URL."""
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Failed to download image from '{image_url}': {exc}") from exc
    return response.content


def _lookup_eagle_row_by_url(
    supabase: Client, image_url: str
) -> Optional[Dict[str, Any]]:
    """Find a matching eagle_images row for the provided URL."""
    # Direct image_url match first
    try:
        direct = supabase.table("eagle_images").select(
            "id, image_embedding, image_url, storage_path"
        ).eq("image_url", image_url).limit(1).execute()
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Supabase lookup failed: {exc}") from exc

    rows = direct.data or []
    if rows:
        return rows[0]

    # Try matching via storage path if the URL points at the public bucket
    storage_path = storage_path_from_url(image_url)
    if storage_path:
        try:
            path_match = supabase.table("eagle_images").select(
                "id, image_embedding, image_url, storage_path"
            ).eq("storage_path", storage_path).limit(1).execute()
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"Supabase lookup failed: {exc}") from exc

        rows = path_match.data or []
        if rows:
            return rows[0]

    return None


def get_image_embedding_for_url(
    image_url: str,
    *,
    supabase: Optional[Client] = None,
) -> Tuple[List[float], Optional[Dict[str, Any]], bool]:
    """Return (embedding, matched_row, was_cached) for an image URL."""
    if not image_url or not image_url.strip():
        raise ValueError("Image URL cannot be empty")

    sb = supabase or get_supabase_client()
    existing_row = _lookup_eagle_row_by_url(sb, image_url)
    if existing_row:
        embedding = existing_row.get("image_embedding")
        if isinstance(embedding, list) and embedding:
            return embedding, existing_row, True

    # No cached embedding – generate via Vertex AI
    model = _get_vertex_embedding_model()
    image_bytes = _download_image_bytes(image_url)
    embeddings = model.get_embeddings(image=VertexImage(image_bytes=image_bytes))
    embedding = list(embeddings.image_embedding or [])
    if not embedding:
        raise RuntimeError("Vertex AI returned an empty embedding for the supplied image")

    # Persist embedding if we matched an existing row
    if existing_row and existing_row.get("id"):
        try:
            sb.table("eagle_images").update({"image_embedding": embedding}).eq(
                "id", existing_row["id"]
            ).execute()
        except Exception:
            # Don't fail the search just because the write-back failed.
            pass

    return embedding, existing_row, False


def search_images_by_image_embedding(
    embedding: List[float],
    *,
    limit: int = 20,
    min_similarity: float = 0.3,
    folders: Optional[List[str]] = None,
    supabase: Optional[Client] = None,
    exclude_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Run the image-to-image similarity RPC using a prepared embedding."""
    if not embedding:
        raise ValueError("Embedding cannot be empty for image similarity search")

    sb = supabase or get_supabase_client()
    try:
        result = sb.rpc(
            "search_images_by_image_embedding",
            {
                "query_embedding": embedding,
                "match_limit": limit,
                "folder_filter": folders,
                "similarity_threshold": min_similarity,
                "excluded_ids": exclude_ids,
            },
        ).execute()
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Database image search failed: {exc}") from exc

    return result.data or []


def search_images_by_image_url(
    image_url: str,
    *,
    limit: int = 20,
    min_similarity: float = 0.3,
    folders: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Search for visually similar Eagle images using an input image URL."""
    sb = get_supabase_client()
    embedding, matched_row, cached = get_image_embedding_for_url(
        image_url, supabase=sb
    )
    exclude_ids = None
    if matched_row and matched_row.get("id"):
        try:
            exclude_ids = [int(matched_row["id"])]
        except (TypeError, ValueError):
            exclude_ids = None
    results = search_images_by_image_embedding(
        embedding,
        limit=limit,
        min_similarity=min_similarity,
        folders=folders,
        supabase=sb,
        exclude_ids=exclude_ids,
    )
    metadata = {
        "embedding_source": "cache" if cached else "vertex",
        "matched_row_id": matched_row.get("id") if isinstance(matched_row, dict) else None,
    }
    return results, metadata


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

        if show_urls:
            resolved_url = row_to_image_url(img)
            if resolved_url:
                print(f"    🔗 URL: {resolved_url}")
            elif img.get('storage_path'):
                print(f"    📁 Storage path: {img['storage_path']}")

        print(f"    🆔 ID: {eagle_id}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Search eagle_images database using semantic or image similarity"
    )
    parser.add_argument("query", nargs="?", help="Text query to search for")
    parser.add_argument(
        "--image-url",
        help="Search by providing an example image URL instead of text",
    )
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
        if bool(args.query) == bool(args.image_url):
            parser.error("Provide either a text query or --image-url (but not both)")

        if args.image_url:
            print(f"🖼️ Preparing embedding for: {args.image_url}")
            results, metadata = search_images_by_image_url(
                args.image_url,
                limit=args.limit,
                min_similarity=args.min_similarity,
                folders=args.folders,
            )
            print(
                "✅ Using "
                + ("cached embedding" if metadata.get("embedding_source") == "cache" else "Vertex AI embedding")
            )
        else:
            # Perform text search
            results = search_images_semantic(
                query=args.query,
                limit=args.limit,
                min_similarity=args.min_similarity,
                folders=args.folders,
            )
            metadata = None

        if args.json:
            payload = {"results": results}
            if metadata is not None:
                payload["metadata"] = metadata
            print(json.dumps(payload, indent=2, default=str))
        else:
            format_results(results, show_urls=not args.no_urls)

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
