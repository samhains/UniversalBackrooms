import asyncio
import json
import logging
import os
from typing import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
from mcp.server.fastmcp import FastMCP
from comfyui_client import ComfyUIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCP_Server")

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in config file {config_path}")
        raise

def load_tools():
    """Load tool definitions from separate tools file"""
    config = load_config()
    tools_file = config.get("tools_file", "tools.json")
    tools_path = os.path.join(os.path.dirname(__file__), tools_file)
    try:
        with open(tools_path, 'r') as f:
            tools_data = json.load(f)
            return tools_data.get("tools", {})
    except FileNotFoundError:
        logger.error(f"Tools file not found at {tools_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in tools file {tools_path}")
        raise

config = load_config()
tools = load_tools()

# Global ComfyUI client using config
comfyui_client = ComfyUIClient(config["server"]["comfyui_url"])

# Define application context (for future use)
class AppContext:
    def __init__(self, comfyui_client: ComfyUIClient):
        self.comfyui_client = comfyui_client

# Lifespan management (placeholder for future context support)
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle"""
    logger.info("Starting MCP server lifecycle...")
    try:
        # Startup: Could add ComfyUI health check here in the future
        logger.info("ComfyUI client initialized globally")
        yield AppContext(comfyui_client=comfyui_client)
    finally:
        # Shutdown: Cleanup (if needed)
        logger.info("Shutting down MCP server")

# Initialize FastMCP with lifespan
mcp = FastMCP("ComfyUI_MCP_Server", lifespan=app_lifespan)

# Define the image generation tool
@mcp.tool()
def generate_image(params: dict) -> dict:
    """Generate an image using ComfyUI
    
    Args:
        params: Dictionary containing:
            - prompt (required): Text description of the image to generate
    
    Returns:
        dict: Contains 'image_url' on success or 'error' on failure
        
    Example params: {"prompt": "anime girl in armor"}
    """
    logger.info(f"Received request with params: {params}")
    try:
        param_dict = params
        prompt = param_dict["prompt"]
        
        # Get settings from tools definition and config
        tool_config = tools["generate_image"]
        width = config["resolutions"]["image_generation"]["width"]
        height = config["resolutions"]["image_generation"]["height"]
        workflow_id = tool_config["workflow_id"]
        # Model is optional; some workflows (e.g., Qwen Image) do not use CheckpointLoaderSimple
        model = tool_config.get("model")

        # Use global comfyui_client (since mcp.context isn't available)
        image_url = comfyui_client.generate_image(
            prompt=prompt,
            width=width,
            height=height,
            workflow_id=workflow_id,
            model=model
        )
        logger.info(f"Returning image URL: {image_url}")
        return {"image_url": image_url}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

# Define the video generation tool
@mcp.tool()
def generate_video(params: dict) -> dict:
    """Generate a video using ComfyUI with WAN 2.2 T2V model
    
    Args:
        params: Dictionary containing:
            - prompt (required): Text description of the video to generate
            - audio_prompt (optional): Text description of the audio/sound to generate
            - frame_length (optional): Number of frames for the video
            - width (optional): Video width in pixels
            - height (optional): Video height in pixels
    
    Returns:
        dict: Contains 'video_url' on success or 'error' on failure
        
    Example params: {"prompt": "a cat walking in a garden", "width": 1920, "height": 1080}
    """
    logger.info(f"Received video request with params: {params}")
    try:
        param_dict = params
        prompt = param_dict["prompt"]
        audio_prompt = param_dict.get("audio_prompt")
        frame_length = param_dict.get("frame_length")
        
        # Get settings from tools definition and config
        tool_config = tools["generate_video"]
        workflow_id = tool_config["workflow_id"]
        
        # Use provided resolution or fall back to config values
        width = param_dict.get("width", config["resolutions"]["video_generation"]["width"])
        height = param_dict.get("height", config["resolutions"]["video_generation"]["height"])
        
        # Use default frame length from config if not provided
        if frame_length is None:
            frame_length = config.get("video_generation", {}).get("default_frame_length")

        # Use global comfyui_client
        video_url = comfyui_client.generate_video(
            prompt=prompt,
            width=width,
            height=height,
            audio_prompt=audio_prompt,
            frame_length=frame_length,
            workflow_id=workflow_id
        )
        logger.info(f"Returning video URL: {video_url}")
        return {"video_url": video_url}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

# Define the remix image tool
@mcp.tool()
def remix_image(params: dict) -> dict:
    """Generate an image using 2 input images with Flux Redux through ComfyUI
    
    Args:
        params: Dictionary containing:
            - image1_url (required): URL to the first reference image for style conditioning
            - image2_url (required): URL to the second reference image for style conditioning
            - width (optional): Image width in pixels
            - height (optional): Image height in pixels
    
    Returns:
        dict: Contains 'image_url' on success or 'error' on failure
        
    Example params: {"image1_url": "https://storage.supabase.co/bucket/img1.jpg", "image2_url": "https://storage.supabase.co/bucket/img2.jpg", "width": 720, "height": 720}
    """
    logger.info(f"Received remix image request with params: {params}")
    try:
        param_dict = params
        
        # Required parameters
        image1_url = param_dict["image1_url"]
        image2_url = param_dict["image2_url"]
        
        # Optional parameters
        width = param_dict.get("width")
        height = param_dict.get("height")
        
        # Get settings from tools definition
        tool_config = tools["remix_image"]
        
        # Use global comfyui_client
        image_url = comfyui_client.remix_image(
            image1_url=image1_url,
            image2_url=image2_url,
            width=width,
            height=height
        )
        
        logger.info(f"Returning remix image URL: {image_url}")
        return {"image_url": image_url}
        
    except KeyError as e:
        missing_param = str(e).strip("'")
        error_msg = f"Missing required parameter: {missing_param}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

# Define the image edit tool (Qwen Image Edit)
@mcp.tool()
def edit_image(params: dict) -> dict:
    """Edit an image using Qwen Image Edit workflow through ComfyUI

    Args:
        params: Dictionary containing:
            - image_url (required): URL to the source image
            - prompt (required): Edit instruction text
            - width (optional): Output image width in pixels
            - height (optional): Output image height in pixels

    Returns:
        dict: Contains 'image_url' on success or 'error' on failure

    Example params: {"image_url": "https://.../image.jpg", "prompt": "turn the sky green", "width": 1024, "height": 1024}
    """
    logger.info(f"Received image edit request with params: {params}")
    try:
        param_dict = params
        
        # Required parameters
        image_url = param_dict["image_url"]
        prompt = param_dict["prompt"]

        # Optional parameters
        width = param_dict.get("width")
        height = param_dict.get("height")

        # Use global comfyui_client
        edited_image_url = comfyui_client.edit_image(
            image_url=image_url,
            prompt=prompt,
            width=width,
            height=height,
        )

        logger.info(f"Returning edited image URL: {edited_image_url}")
        return {"image_url": edited_image_url}
        
    except KeyError as e:
        missing_param = str(e).strip("'")
        error_msg = f"Missing required parameter: {missing_param}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

# Define the 3-image video generation tool
@mcp.tool()
def generate_3_image_video(params: dict) -> dict:
    """Generate a video using 3 input images with Flux Redux and WAN 2.2 I2V workflow through ComfyUI
    
    Args:
        params: Dictionary containing:
            - image1_url (required): URL to the first reference image for style conditioning
            - image2_url (required): URL to the second reference image for style conditioning  
            - image3_url (required): URL to the third reference image for style conditioning
            - frame_length (optional): Number of frames for the video
            - width (optional): Video width in pixels  
            - height (optional): Video height in pixels
    
    Returns:
        dict: Contains 'video_url' on success or 'error' on failure
        
    Example params: {"image1_url": "https://storage.supabase.co/bucket/img1.jpg", "image2_url": "https://storage.supabase.co/bucket/img2.jpg", "image3_url": "https://storage.supabase.co/bucket/img3.jpg", "width": 1920, "height": 1080}
    """
    logger.info(f"Received 3-image video request with params: {params}")
    try:
        param_dict = params
        
        # Required parameters
        image1_url = param_dict["image1_url"]
        image2_url = param_dict["image2_url"]  
        image3_url = param_dict["image3_url"]
        
        # Optional parameters
        frame_length = param_dict.get("frame_length")
        
        # Get settings from tools definition and config
        tool_config = tools["generate_3_image_video"]
        
        # Use provided resolution or fall back to workflow defaults
        width = param_dict.get("width")
        height = param_dict.get("height")
        
        # Use global comfyui_client
        video_url = comfyui_client.generate_3_image_video(
            image1_url=image1_url,
            image2_url=image2_url,
            image3_url=image3_url,
            width=width,
            height=height,
            frame_length=frame_length
        )
        
        logger.info(f"Returning 3-image video URL: {video_url}")
        return {"video_url": video_url}
        
    except KeyError as e:
        missing_param = str(e).strip("'")
        error_msg = f"Missing required parameter: {missing_param}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

@mcp.tool()
def generate_f2f_video(params: dict) -> dict:
    """Generate a video using frame-to-frame animation between 2 input images with WAN 2.2 I2V workflow through ComfyUI
    
    Args:
        params: Dictionary containing:
            - image1_url (required): URL to the first/starting frame image
            - image2_url (required): URL to the second/ending frame image
            - width (optional): Video width in pixels
            - height (optional): Video height in pixels
            - frame_length (optional): Number of frames for the video (minimum 81 frames)
            - prompt (optional): Custom prompt for motion description
    
    Returns:
        dict: Contains 'video_url' on success or 'error' on failure
        
    Example params: {"image1_url": "https://storage.supabase.co/bucket/start.jpg", "image2_url": "https://storage.supabase.co/bucket/end.jpg", "width": 720, "height": 720, "frame_length": 81, "prompt": "smooth transition"}
    """
    logger.info(f"Received frame-to-frame video request with params: {params}")
    try:
        param_dict = params
        
        # Required parameters
        image1_url = param_dict["image1_url"]
        image2_url = param_dict["image2_url"]
        
        # Optional parameters
        width = param_dict.get("width")
        height = param_dict.get("height")
        frame_length = param_dict.get("frame_length")
        prompt = param_dict.get("prompt")
        
        # Use global comfyui_client
        video_url = comfyui_client.generate_f2f_video(
            image1_url=image1_url,
            image2_url=image2_url,
            width=width,
            height=height,
            frame_length=frame_length,
            prompt=prompt
        )
        
        logger.info(f"Returning frame-to-frame video URL: {video_url}")
        return {"video_url": video_url}
        
    except KeyError as e:
        missing_param = str(e).strip("'")
        error_msg = f"Missing required parameter: {missing_param}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

# Create FastAPI app for HTTP endpoints
app = FastAPI(title="ComfyUI MCP Server")

@app.post("/generate_image")
async def generate_image_http(params: dict):
    """HTTP endpoint for image generation"""
    logger.info(f"Received HTTP request with params: {params}")
    try:
        result = generate_image(params)
        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

@app.post("/generate_image_stream")
async def generate_image_stream(params: dict):
    """SSE endpoint for streaming image generation progress"""
    logger.info(f"Received SSE request with params: {params}")
    
    async def event_stream():
        try:
            # Send initial status
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Initializing image generation...'})}\n\n"
            
            # Generate image (this will still use polling internally)
            result = generate_image(params)
            
            # Send progress updates (can be enhanced to show actual ComfyUI progress)
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Generating image...'})}\n\n"
            
            # Send final result
            yield f"data: {json.dumps({'status': 'complete', 'result': result})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.post("/generate_video")
async def generate_video_http(params: dict):
    """HTTP endpoint for video generation"""
    logger.info(f"Received HTTP video request with params: {params}")
    try:
        result = generate_video(params)
        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

@app.post("/remix_image")
async def remix_image_http(params: dict):
    """HTTP endpoint for remix image generation"""
    logger.info(f"Received HTTP remix image request with params: {params}")
    try:
        result = remix_image(params)
        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

@app.post("/edit_image")
async def edit_image_http(params: dict):
    """HTTP endpoint for image edit generation"""
    logger.info(f"Received HTTP image edit request with params: {params}")
    try:
        result = edit_image(params)
        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

@app.post("/generate_3_image_video")
async def generate_3_image_video_http(params: dict):
    """HTTP endpoint for 3-image video generation"""
    logger.info(f"Received HTTP 3-image video request with params: {params}")
    try:
        result = generate_3_image_video(params)
        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

@app.post("/generate_f2f_video")
async def generate_f2f_video_http(params: dict):
    """HTTP endpoint for frame-to-frame video generation"""
    logger.info(f"Received HTTP frame-to-frame video request with params: {params}")
    try:
        result = generate_f2f_video(params)
        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

@app.post("/generate_video_stream")
async def generate_video_stream(params: dict):
    """SSE endpoint for streaming video generation progress"""
    logger.info(f"Received SSE video request with params: {params}")
    
    async def event_stream():
        try:
            # Send initial status
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Initializing video generation...'})}\n\n"
            
            # Generate video (this will still use polling internally)
            result = generate_video(params)
            
            # Send progress updates
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Generating video with WAN 2.2...'})}\n\n"
            
            # Send final result
            yield f"data: {json.dumps({'status': 'complete', 'result': result})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in video stream: {e}")
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ComfyUI MCP Server"}

if __name__ == "__main__":
    import sys
    if "--http" in sys.argv:
        logger.info("Starting HTTP MCP server on http://0.0.0.0:9000...")
        uvicorn.run(app, host="0.0.0.0", port=9000)
    else:
        logger.info("Starting MCP server...")
        mcp.run()
