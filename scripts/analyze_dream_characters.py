#!/usr/bin/env python3
"""
Analyze dreams to identify recurring characters.

Environment:
  - SUPABASE_URL: e.g. https://<project>.supabase.co
  - SUPABASE_ANON_KEY (preferred for read) or SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY

Usage:
  python scripts/analyze_dream_characters.py --output recurring_characters.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import re
from collections import defaultdict
from typing import List, Dict, Any
from pathlib import Path
import requests


def _load_env():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key, value)


def _env_keys():
    _load_env()
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
    )
    if not url or not key:
        sys.exit(
            "Missing SUPABASE_URL and/or SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY)."
        )
    return url.rstrip("/"), key


def _headers(key: str):
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def fetch_all_my_dreams(url: str, key: str) -> List[Dict[str, Any]]:
    """Fetch all dreams with source='mine'"""
    endpoint = f"{url}/rest/v1/dreams"
    params = {
        "select": "*",
        "source": "eq.mine",
        "order": "date.desc",
    }

    all_dreams = []
    offset = 0
    limit = 1000

    while True:
        params["limit"] = str(limit)
        params["offset"] = str(offset)

        r = requests.get(endpoint, headers=_headers(key), params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()

        if not batch:
            break

        all_dreams.extend(batch)
        offset += limit

        print(f"Fetched {len(all_dreams)} dreams so far...", file=sys.stderr)

        if len(batch) < limit:
            break

    return all_dreams


def extract_character_mentions(content: str) -> List[str]:
    """
    Extract potential character mentions from dream content.
    This looks for:
    - Proper names (capitalized words that aren't at start of sentences)
    - Relationship terms (my mom, my friend, etc.)
    """
    if not content:
        return []

    # Skip words that are commonly capitalized but not names
    skip_words = {
        'i', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'it', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'could',
        'there', 'here', 'where', 'when', 'why', 'how', 'what', 'which', 'who',
        'if', 'because', 'as', 'while', 'after', 'before', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'very', 'so',
        'just', 'don', 'now', 'no', 'yes', 'not', 'only', 'own', 'same', 'such',
        'than', 'too', 'out', 'back', 'down', 'over', 'off', 'some', 'all', 'any',
        'each', 'every', 'both', 'few', 'more', 'most', 'other', 'another',
        'many', 'much', 'several', 'me', 'my', 'we', 'us', 'our',
        # Common place names and non-character words
        'im', 'instagram', 'youtube', 'brooklyn', 'new', 'york', 'europe',
        'american', 'australian', 'asian', 'dvds', 'id', 'vce', 'dmt', 'bbc'
    }

    mentions = []

    # Pattern 1: Relationship terms (my mom, my friend, the guy, etc.)
    relationship_pattern = r'\b(?:my|the|a|an)\s+(mom|dad|mother|father|brother|sister|sibling|friend|boyfriend|girlfriend|husband|wife|spouse|son|daughter|child|kid|cousin|uncle|aunt|grandma|grandpa|grandmother|grandfather|boss|manager|teacher|professor|coworker|colleague|neighbor|neighbour|partner|ex|roommate|classmate|doctor|therapist|guy|girl|man|woman|person|people|stranger|student|employee)\b'
    for match in re.finditer(relationship_pattern, content, re.IGNORECASE):
        mentions.append(match.group(1).lower())

    # Pattern 2: Proper names - capitalized words that appear mid-sentence
    # Split into sentences first
    sentences = re.split(r'[.!?]+\s+', content)
    for sentence in sentences:
        # Find capitalized words that aren't at the start
        words = sentence.split()
        for i, word in enumerate(words):
            # Clean punctuation
            clean_word = re.sub(r'[^\w\s-]', '', word)
            if not clean_word:
                continue

            # Skip if it's the first word of the sentence or a common word
            if i == 0 or clean_word.lower() in skip_words:
                continue

            # Check if it's capitalized (likely a proper name)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                # Could be a name - add it
                mentions.append(clean_word)

    return mentions


def analyze_characters(dreams: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze dreams to find recurring characters.
    Returns a dictionary with character statistics and dream appearances.
    """
    character_occurrences = defaultdict(lambda: {
        "count": 0,
        "dreams": [],
        "dream_ids": set(),
        "contexts": []
    })

    for dream in dreams:
        dream_id = dream.get("id") or dream.get("uuid")
        content = dream.get("content", "")
        date = dream.get("date") or dream.get("created_at")

        # Extract character mentions
        mentions = extract_character_mentions(content)

        # Process mentions
        seen_in_dream = set()
        for mention in mentions:
            mention_normalized = mention.lower().strip()

            # Skip common words that aren't really characters
            if mention_normalized in ["the", "a", "my", "i", "me", "we", "us"]:
                continue

            # Track unique mentions per dream
            if mention_normalized not in seen_in_dream:
                character_occurrences[mention_normalized]["count"] += 1
                character_occurrences[mention_normalized]["dream_ids"].add(dream_id)
                character_occurrences[mention_normalized]["dreams"].append({
                    "dream_id": dream_id,
                    "date": date,
                    "snippet": content[:200] if content else ""
                })
                seen_in_dream.add(mention_normalized)

            # Collect context around mention
            mention_pattern = re.compile(re.escape(mention), re.IGNORECASE)
            for match in mention_pattern.finditer(content):
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end].strip()
                if context and len(character_occurrences[mention_normalized]["contexts"]) < 10:
                    character_occurrences[mention_normalized]["contexts"].append(context)

    # Convert to final format
    result = {
        "total_dreams_analyzed": len(dreams),
        "recurring_characters": []
    }

    # Sort by frequency
    sorted_characters = sorted(
        character_occurrences.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )

    for char_name, data in sorted_characters:
        # Only include characters that appear in 2+ dreams
        if data["count"] >= 2:
            result["recurring_characters"].append({
                "name": char_name,
                "appearance_count": data["count"],
                "unique_dreams": len(data["dream_ids"]),
                "dreams": sorted(data["dreams"], key=lambda x: x.get("date") or "", reverse=True),
                "sample_contexts": data["contexts"][:5]  # Top 5 contexts
            })

    return result


def main():
    ap = argparse.ArgumentParser(
        description="Analyze dreams to identify recurring characters"
    )
    ap.add_argument(
        "--output",
        "-o",
        default="recurring_characters.json",
        help="Output JSON file (default: recurring_characters.json)",
    )
    ap.add_argument(
        "--min-occurrences",
        type=int,
        default=2,
        help="Minimum occurrences to include character (default: 2)",
    )
    args = ap.parse_args()

    url, key = _env_keys()

    print("Fetching all dreams with source='mine'...", file=sys.stderr)
    dreams = fetch_all_my_dreams(url, key)
    print(f"Fetched {len(dreams)} total dreams", file=sys.stderr)

    print("Analyzing characters...", file=sys.stderr)
    analysis = analyze_characters(dreams)

    # Filter by min occurrences
    analysis["recurring_characters"] = [
        char for char in analysis["recurring_characters"]
        if char["appearance_count"] >= args.min_occurrences
    ]

    # Convert dream_ids sets to lists for JSON serialization
    print(f"Found {len(analysis['recurring_characters'])} recurring characters", file=sys.stderr)

    # Write output
    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults written to {args.output}", file=sys.stderr)
    print(f"\nTop 10 recurring characters:", file=sys.stderr)
    for i, char in enumerate(analysis["recurring_characters"][:10], 1):
        print(f"{i}. {char['name']}: {char['appearance_count']} appearances across {char['unique_dreams']} dreams", file=sys.stderr)


if __name__ == "__main__":
    main()
