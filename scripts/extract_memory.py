#!/usr/bin/env python3
"""
Memory extraction script for dream simulations.

This script takes a completed conversation log and extracts significant
information to update the persistent memory system.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure repository root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_conversation_log(log_path: Path) -> str:
    """Read the conversation log file and return its content."""
    try:
        with log_path.open("r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading log file {log_path}: {e}")
        return ""


def read_memory_file(memory_path: Path) -> Dict[str, Any]:
    """Read the current memory.json file."""
    try:
        if memory_path.exists():
            with memory_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # Return default structure if file doesn't exist
            return {
                "sessions": [],
                "significant_events": [],
                "recurring_themes": [],
                "character_developments": [],
                "insights": [],
                "last_updated": None,
                "session_count": 0
            }
    except Exception as e:
        print(f"Error reading memory file {memory_path}: {e}")
        return {}


def call_memory_extractor_llm(
    conversation: str,
    current_memory: Dict[str, Any],
    template_path: Path,
    model: str = "sonnet4"
) -> Optional[Dict[str, Any]]:
    """Call the LLM to extract and update memory."""

    # Prepare the system prompt
    system_prompt_path = template_path / "memory_extractor.system.md"
    try:
        with system_prompt_path.open("r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception as e:
        print(f"Error reading system prompt: {e}")
        return None

    # Prepare the user message
    user_message = f"""## Current Memory
{json.dumps(current_memory, indent=2)}

## Conversation Transcript
{conversation}

Please analyze this conversation and update the memory.json structure."""

    # Call backrooms.py with memory extraction setup
    try:
        # Create a temporary template for memory extraction
        temp_template_path = template_path / "temp_memory_extraction"
        temp_template_path.mkdir(exist_ok=True)

        # Write temporary template files
        template_json = {
            "agents": [
                {"system": str(system_prompt_path.relative_to(template_path))}
            ]
        }

        with (temp_template_path / "template.json").open("w") as f:
            json.dump(template_json, f, indent=2)

        # Write the system prompt to temp location
        temp_system = temp_template_path / "memory_extractor.system.md"
        with temp_system.open("w") as f:
            f.write(system_prompt)

        # Call backrooms.py
        cmd = [
            sys.executable,
            str(ROOT / "backrooms.py"),
            "--lm", model,
            "--template", str(temp_template_path.relative_to(ROOT / "templates")),
            "--max-turns", "1",
            "--query", user_message
        ]

        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Memory extraction LLM call failed: {result.stderr}")
            return None

        # Parse the response to extract JSON
        output = result.stdout.strip()
        # Look for JSON in the output
        try:
            # Try to find JSON block in the output
            start = output.find('{')
            end = output.rfind('}') + 1
            if start != -1 and end > start:
                json_str = output[start:end]
                return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing LLM response as JSON: {e}")
            print(f"Raw output: {output}")
            return None

    except Exception as e:
        print(f"Error calling memory extraction LLM: {e}")
        return None
    finally:
        # Clean up temporary files
        try:
            if temp_template_path.exists():
                for file in temp_template_path.iterdir():
                    file.unlink()
                temp_template_path.rmdir()
        except Exception:
            pass

    return None


def update_memory_file(memory_path: Path, updated_memory: Dict[str, Any]) -> bool:
    """Write the updated memory back to the file."""
    try:
        # Add metadata
        import datetime
        updated_memory["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        updated_memory["session_count"] = updated_memory.get("session_count", 0) + 1

        with memory_path.open("w", encoding="utf-8") as f:
            json.dump(updated_memory, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(f"Updated memory file: {memory_path}")
        return True
    except Exception as e:
        print(f"Error writing memory file {memory_path}: {e}")
        return False


def extract_memory_for_log(
    log_path: Path,
    template: str,
    model: str = "sonnet4"
) -> bool:
    """Extract memory for a specific log file."""

    # Determine template path
    template_path = ROOT / "templates" / template
    if not template_path.exists():
        print(f"Template not found: {template_path}")
        return False

    # Check if this template has memory support
    memory_path = template_path / "memory.json"
    if not (template_path / "memory_extractor.system.md").exists():
        print(f"Template {template} does not support memory extraction")
        return False

    # Read conversation log
    conversation = read_conversation_log(log_path)
    if not conversation:
        print(f"No conversation content found in {log_path}")
        return False

    # Read current memory
    current_memory = read_memory_file(memory_path)

    # Extract and update memory
    updated_memory = call_memory_extractor_llm(
        conversation,
        current_memory,
        template_path,
        model
    )

    if updated_memory is None:
        print("Memory extraction failed")
        return False

    # Write updated memory
    return update_memory_file(memory_path, updated_memory)


def main():
    parser = argparse.ArgumentParser(description="Extract memory from dream simulation logs")
    parser.add_argument("--log", required=True, help="Path to conversation log file")
    parser.add_argument("--template", required=True, help="Template name")
    parser.add_argument("--model", default="sonnet4", help="Model to use for extraction")

    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    success = extract_memory_for_log(log_path, args.template, args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()