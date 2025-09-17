import requests
import json
import time
import logging
import os
import uuid
import tempfile
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComfyUIClient")

# Single source of truth: per-workflow param mapping and output config
WORKFLOW_SPECS = {
    "basic_api_test": {
        "params": {"prompt": ("6", "text"), "width": ("5", "width"), "height": ("5", "height"), "model": ("4", "ckpt_name")},
        "output": {"type": "image"}
    },
    "image_qwen_image": {
        "params": {"prompt": ("78", "value"), "width": ("75", "value"), "height": ("76", "value")},
        "output": {"type": "image", "node_id": "60"}
    },
    "image_qwen_image_edit": {
        "params": {
            "prompt": ("102", "value"),
            "image_url": ("103", "url_or_path"),
            "width": ("106", "value"),
            "height": ("107", "value")
        },
        "output": {"type": "image", "node_id": "60"}
    },
    "flux-dev-workflow": {
        "params": {"prompt": ("6", "text"), "width": ("27", "width"), "height": ("27", "height"), "model": ("30", "ckpt_name")},
        "output": {"type": "image"}
    },
    "wan-2.2-t2v-api": {
        "params": {"prompt": ("6", "text"), "width": ("59", "width"), "height": ("59", "height")},
        "output": {"type": "video", "node_id": "125"}
    },
    "mmaudio-workflow": {
        "params": {"prompt": ("6", "text"), "audio_prompt": ("83:75", "prompt"), "frame_length": ("59", "length"), "width": ("59", "width"), "height": ("59", "height")},
        "output": {"type": "video", "node_id": "125"}
    },
    "wan-2.2-t2v-hq": {
        "params": {"prompt": ("141", "value"), "audio_prompt": ("147", "value"), "width": ("143", "value"), "height": ("144", "value"), "frame_length": ("145", "value")},
        "output": {"type": "video", "node_id": "125"}
    },
    "wan2.2-t2v-sd": {
        "params": {"prompt": ("141", "value"), "audio_prompt": ("147", "value"), "width": ("143", "value"), "height": ("144", "value"), "frame_length": ("145", "value")},
        "output": {"type": "video", "node_id": "125"}
    },
    "flux-3-redux-wan2.2-i2v-sd": {
        "params": {"image1_url": ("243", "url_or_path"), "image2_url": ("248", "url_or_path"), "image3_url": ("249", "url_or_path"), "width": ("143", "value"), "height": ("144", "value"), "frame_length": ("145", "value")},
        "output": {"type": "video", "node_id": "125"}
    },
    "flux-2-redux": {
        "params": {"image1_url": ("69", "url_or_path"), "image2_url": ("70", "url_or_path"), "width": ("62", "value"), "height": ("65", "value")},
        "output": {"type": "image", "node_id": "9"}
    },
    "wan2.2-f2f-loop": {
        "params": {"image1_url": ("278", "url_or_path"), "image2_url": ("280", "url_or_path"), "width": ("143", "value"), "height": ("144", "value"), "frame_length": ("145", "value"), "prompt": ("141", "value")},
        "output": {"type": "video", "node_id": "125"}
    }
}

# Backward-compatible alias expected by some code paths
WORKFLOW_MAPPINGS = {wid: spec["params"] for wid, spec in WORKFLOW_SPECS.items()}
DEFAULT_MAPPING = WORKFLOW_MAPPINGS["basic_api_test"]

def _apply_params_to_workflow(workflow: dict, workflow_id: str, params: dict):
    mapping = WORKFLOW_SPECS.get(workflow_id, {}).get("params", DEFAULT_MAPPING)
    logger.info(f"Using mapping for workflow {workflow_id}: {mapping}")
    for param_key, value in params.items():
        if param_key in mapping:
            node_id, input_key = mapping[param_key]
            if node_id not in workflow:
                raise Exception(f"Node {node_id} not found in workflow {workflow_id}")
            workflow[node_id]["inputs"][input_key] = value

def _extract_output_url(base_url: str, outputs: dict, workflow_id: str):
    spec = WORKFLOW_SPECS.get(workflow_id, {})
    out_spec = spec.get("output", {})
    node_id = out_spec.get("node_id")
    output_type = out_spec.get("type", "image")

    if node_id:
        if node_id not in outputs:
            raise Exception(f"Output node {node_id} not found in outputs: {outputs}")
        node_out = outputs[node_id]
        if output_type == "image" and "images" in node_out:
            data = node_out["images"][0]
        elif output_type == "video":
            if "images" in node_out:
                data = node_out["images"][0]
            elif "gifs" in node_out:
                data = node_out["gifs"][0]
            elif "filenames" in node_out:
                data = node_out["filenames"][0]
            else:
                raise Exception(f"No video output found in node {node_id}: {node_out}")
        else:
            key = next((k for k in ("images", "gifs", "filenames") if k in node_out), None)
            if not key:
                raise Exception(f"No recognizable outputs in node {node_id}: {node_out}")
            data = node_out[key][0]
    else:
        # Fallback: first node with images
        image_node = next((nid for nid, out in outputs.items() if "images" in out), None)
        if not image_node:
            raise Exception(f"No output node with images found: {outputs}")
        data = outputs[image_node]["images"][0]

    filename = data["filename"]
    subfolder = data.get("subfolder", "")
    return f"{base_url}/view?filename={filename}&subfolder={subfolder}&type=output"

class ComfyUIClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.available_models = self._get_available_models()

    def _get_available_models(self):
        """Fetch list of available checkpoint models from ComfyUI"""
        try:
            response = requests.get(f"{self.base_url}/object_info/CheckpointLoaderSimple")
            if response.status_code != 200:
                logger.warning("Failed to fetch model list; using default handling")
                return []
            data = response.json()
            models = data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            logger.info(f"Available models: {models}")
            return models
        except Exception as e:
            logger.warning(f"Error fetching models: {e}")
            return []

    def generate_image(self, prompt, width, height, workflow_id="basic_api_test", model=None):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            workflow_file = os.path.join(script_dir, "workflows", f"{workflow_id}.json")
            with open(workflow_file, "r") as f:
                workflow = json.load(f)

            params = {"prompt": prompt, "width": width, "height": height}
            if model:
                # Validate or correct model name
                if model.endswith("'"):  # Strip accidental quote
                    model = model.rstrip("'")
                    logger.info(f"Corrected model name: {model}")
                if self.available_models and model not in self.available_models:
                    raise Exception(f"Model '{model}' not in available models: {self.available_models}")
                params["model"] = model

            _apply_params_to_workflow(workflow, workflow_id, params)

            logger.info(f"Submitting workflow {workflow_id} to ComfyUI...")
            response = requests.post(f"{self.base_url}/prompt", json={"prompt": workflow})
            if response.status_code != 200:
                raise Exception(f"Failed to queue workflow: {response.status_code} - {response.text}")

            prompt_id = response.json()["prompt_id"]
            logger.info(f"Queued workflow with prompt_id: {prompt_id}")

            while True:
                history = requests.get(f"{self.base_url}/history/{prompt_id}").json()
                if history.get(prompt_id):
                    outputs = history[prompt_id]["outputs"]
                    logger.info("Workflow outputs: %s", json.dumps(outputs, indent=2))
                    image_url = _extract_output_url(self.base_url, outputs, workflow_id)
                    logger.info(f"Generated image URL: {image_url}")
                    return image_url
                time.sleep(1)

        except FileNotFoundError:
            raise Exception(f"Workflow file '{workflow_file}' not found")
        except KeyError as e:
            raise Exception(f"Workflow error - invalid node or input: {e}")
        except requests.RequestException as e:
            raise Exception(f"ComfyUI API error: {e}")

    def generate_f2f_video(self, image1_url, image2_url, width=None, height=None, frame_length=None, prompt=None):
        """Generate frame-to-frame animation video between two images using WAN 2.2 I2V workflow
        
        Args:
            image1_url (str): URL to the first/starting frame image
            image2_url (str): URL to the second/ending frame image  
            width (int, optional): Video width in pixels. Defaults to 720.
            height (int, optional): Video height in pixels. Defaults to 720.
            frame_length (int, optional): Number of frames for the video. Defaults to 81.
            prompt (str, optional): Custom prompt for motion description. Defaults to "".
            
        Returns:
            str: URL to the generated video file
            
        Raises:
            Exception: If workflow execution fails
        """
        try:
            workflow_id = "wan2.2-f2f-loop"
            
            # Load workflow
            script_dir = os.path.dirname(os.path.abspath(__file__))
            workflow_file = os.path.join(script_dir, "workflows", f"{workflow_id}.json")
            with open(workflow_file, "r") as f:
                workflow = json.load(f)

            # Prepare parameters with defaults
            params = {
                "image1_url": image1_url,
                "image2_url": image2_url,
                "width": width or 720,
                "height": height or 720,
                "frame_length": frame_length or 81,
                "prompt": prompt or ""
            }
            _apply_params_to_workflow(workflow, workflow_id, params)

            # Submit workflow and wait for results
            response = requests.post(f"{self.base_url}/prompt", json={"prompt": workflow})
            if response.status_code != 200:
                raise Exception(f"Failed to queue workflow: {response.status_code} - {response.text}")

            prompt_id = response.json()["prompt_id"]
            logger.info(f"Queued frame-to-frame video workflow with prompt_id: {prompt_id}")

            # Wait for completion and get results using HARDCODED output node
            while True:
                history = requests.get(f"{self.base_url}/history/{prompt_id}").json()
                if history.get(prompt_id):
                    outputs = history[prompt_id]["outputs"]
                    logger.info("Frame-to-frame video workflow outputs: %s", json.dumps(outputs, indent=2))
                    video_url = _extract_output_url(self.base_url, outputs, workflow_id)
                    logger.info(f"Generated frame-to-frame video URL: {video_url}")
                    return video_url
                    
                time.sleep(1)

        except Exception as e:
            raise Exception(f"Error generating frame-to-frame video: {e}")

    def _download_image_from_url(self, url, filename_prefix="temp_image"):
        """Download an image from URL and save it locally with a unique filename"""
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {url}")
            
            logger.info(f"Downloading image from URL: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file extension from URL or content type
            content_type = response.headers.get('content-type', '').lower()
            if 'image/jpeg' in content_type or 'image/jpg' in content_type:
                ext = '.jpg'
            elif 'image/png' in content_type:
                ext = '.png'
            elif 'image/webp' in content_type:
                ext = '.webp'
            elif 'image/gif' in content_type:
                ext = '.gif'
            else:
                # Try to get extension from URL path
                path = parsed_url.path.lower()
                if path.endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                    ext = os.path.splitext(path)[1]
                else:
                    ext = '.jpg'  # Default fallback
            
            # Generate unique filename
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{filename_prefix}_{unique_id}{ext}"
            
            # Save to temp directory
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, filename)
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded image to: {file_path}")
            return filename  # Return just the filename, not the full path
            
        except requests.RequestException as e:
            raise Exception(f"Failed to download image from {url}: {e}")
        except Exception as e:
            raise Exception(f"Error processing image download from {url}: {e}")

    def remix_image(self, image1_url, image2_url, width=None, height=None):
        """Generate image using 2 input images with Flux Redux style conditioning"""
        try:
            workflow_id = "flux-2-redux"
            
            # Load workflow
            script_dir = os.path.dirname(os.path.abspath(__file__))
            workflow_file = os.path.join(script_dir, "workflows", f"{workflow_id}.json")
            with open(workflow_file, "r") as f:
                workflow = json.load(f)

            # Prepare parameters with defaults - try passing URLs directly to LoadImage nodes
            params = {
                "image1_url": image1_url,
                "image2_url": image2_url,
                "width": width or 720,
                "height": height or 720
            }
            _apply_params_to_workflow(workflow, workflow_id, params)

            logger.info(f"Submitting remix image workflow {workflow_id} to ComfyUI...")
            response = requests.post(f"{self.base_url}/prompt", json={"prompt": workflow})
            if response.status_code != 200:
                raise Exception(f"Failed to queue workflow: {response.status_code} - {response.text}")

            prompt_id = response.json()["prompt_id"]
            logger.info(f"Queued remix image workflow with prompt_id: {prompt_id}")

            # Wait for completion and get results
            while True:
                history = requests.get(f"{self.base_url}/history/{prompt_id}").json()
                if history.get(prompt_id):
                    outputs = history[prompt_id]["outputs"]
                    logger.info("Remix image workflow outputs: %s", json.dumps(outputs, indent=2))
                    image_url = _extract_output_url(self.base_url, outputs, workflow_id)
                    
                    logger.info(f"Generated remix image URL: {image_url}")
                    return image_url
                    
                time.sleep(1)

        except FileNotFoundError:
            raise Exception(f"Workflow file '{workflow_file}' not found")
        except KeyError as e:
            raise Exception(f"Workflow error - invalid node or input: {e}")
        except requests.RequestException as e:
            raise Exception(f"Failed to communicate with ComfyUI server: {e}")
        except Exception as e:
            raise Exception(f"Error generating remix image: {e}")

    def generate_3_image_video(self, image1_url, image2_url, image3_url, width=None, height=None, frame_length=None):
        """Generate video using 3 input images with Flux Redux and WAN 2.2 I2V workflow"""
        try:
            workflow_id = "flux-3-redux-wan2.2-i2v-sd"
            
            # Load workflow
            script_dir = os.path.dirname(os.path.abspath(__file__))
            workflow_file = os.path.join(script_dir, "workflows", f"{workflow_id}.json")
            with open(workflow_file, "r") as f:
                workflow = json.load(f)

            # Prepare parameters - pass URLs directly to LoadImageFromUrlOrPath nodes
            params = {
                "image1_url": image1_url,
                "image2_url": image2_url, 
                "image3_url": image3_url
            }
            
            # Add optional parameters if provided
            if width is not None:
                params["width"] = width
            if height is not None:
                params["height"] = height
            if frame_length is not None:
                params["frame_length"] = frame_length

            _apply_params_to_workflow(workflow, workflow_id, params)

            logger.info(f"Submitting 3-image video workflow {workflow_id} to ComfyUI...")
            response = requests.post(f"{self.base_url}/prompt", json={"prompt": workflow})
            if response.status_code != 200:
                raise Exception(f"Failed to queue workflow: {response.status_code} - {response.text}")

            prompt_id = response.json()["prompt_id"]
            logger.info(f"Queued 3-image video workflow with prompt_id: {prompt_id}")

            # Wait for completion and get results
            while True:
                history = requests.get(f"{self.base_url}/history/{prompt_id}").json()
                if history.get(prompt_id):
                    outputs = history[prompt_id]["outputs"]
                    logger.info("3-image video workflow outputs: %s", json.dumps(outputs, indent=2))
                    video_url = _extract_output_url(self.base_url, outputs, workflow_id)
                    logger.info(f"Generated 3-image video URL: {video_url}")
                    return video_url
                    
                time.sleep(1)

        except FileNotFoundError:
            raise Exception(f"Workflow file '{workflow_file}' not found")
        except KeyError as e:
            raise Exception(f"Workflow error - invalid node or input: {e}")
        except requests.RequestException as e:
            raise Exception(f"ComfyUI API error: {e}")

    

    def generate_video(self, prompt, width=None, height=None, audio_prompt=None, frame_length=None, workflow_id=None):
        try:
            # Load config to get default workflow if none specified
            if workflow_id is None:
                config_path = os.path.join(os.path.dirname(__file__), "config.json")
                with open(config_path, 'r') as f:
                    config = json.load(f)
                workflow_id = config["video_generation"]["workflow"]
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            workflow_file = os.path.join(script_dir, "workflows", f"{workflow_id}.json")
            with open(workflow_file, "r") as f:
                workflow = json.load(f)

            params = {"prompt": prompt}
            if width is not None:
                params["width"] = width
            if height is not None:
                params["height"] = height
            if audio_prompt:
                params["audio_prompt"] = audio_prompt
            if frame_length:
                params["frame_length"] = frame_length
            _apply_params_to_workflow(workflow, workflow_id, params)
            
            # Calculate dynamic audio duration for mmaudio workflow
            if workflow_id == "mmaudio-workflow":
                try:
                    # Get video parameters from workflow nodes
                    frame_count = workflow.get("59", {}).get("inputs", {}).get("length", 181)
                    
                    # Use base generation rate (16fps) for duration calculation
                    # With 32fps output and 2x RIFE interpolation, this maintains original timing
                    base_fps = 16
                    duration = frame_count / base_fps
                    
                    logger.info(f"Calculated audio duration: {frame_count} frames ÷ {base_fps} fps (base rate) = {duration} seconds")
                    
                    # Update audio duration in MMAudioSampler node
                    if "83:75" in workflow:
                        workflow["83:75"]["inputs"]["duration"] = duration
                        logger.info(f"Set MMAudioSampler duration to {duration} seconds")
                except Exception as e:
                    logger.warning(f"Failed to calculate dynamic duration, using default: {e}")

            logger.info(f"Submitting video workflow {workflow_id} to ComfyUI...")
            response = requests.post(f"{self.base_url}/prompt", json={"prompt": workflow})
            if response.status_code != 200:
                raise Exception(f"Failed to queue workflow: {response.status_code} - {response.text}")

            prompt_id = response.json()["prompt_id"]
            logger.info(f"Queued video workflow with prompt_id: {prompt_id}")

            while True:
                history = requests.get(f"{self.base_url}/history/{prompt_id}").json()
                if history.get(prompt_id):
                    outputs = history[prompt_id]["outputs"]
                    logger.info("Video workflow outputs: %s", json.dumps(outputs, indent=2))
                    video_url = _extract_output_url(self.base_url, outputs, workflow_id)
                    logger.info(f"Generated video URL: {video_url}")
                    return video_url
                time.sleep(1)

        except FileNotFoundError:
            raise Exception(f"Workflow file '{workflow_file}' not found")
        except KeyError as e:
            raise Exception(f"Workflow error - invalid node or input: {e}")
        except requests.RequestException as e:
            raise Exception(f"ComfyUI API error: {e}")

    def edit_image(self, image_url: str, prompt: str, width: int | None = None, height: int | None = None):
        """Edit an image using the Qwen Image Edit workflow.

        Parameters map to nodes:
        - prompt -> node 102:value (PrimitiveStringMultiline)
        - image_url -> node 103:url_or_path (LoadImageFromUrlOrPath)
        - width -> node 106:value (PrimitiveInt)
        - height -> node 107:value (PrimitiveInt)
        Output node: 60 (SaveImage)
        """
        try:
            workflow_id = "image_qwen_image_edit"

            # Load workflow
            script_dir = os.path.dirname(os.path.abspath(__file__))
            workflow_file = os.path.join(script_dir, "workflows", f"{workflow_id}.json")
            with open(workflow_file, "r") as f:
                workflow = json.load(f)

            # Prepare parameters
            params = {
                "prompt": prompt,
                "image_url": image_url,
            }
            if width is not None:
                params["width"] = width
            if height is not None:
                params["height"] = height

            _apply_params_to_workflow(workflow, workflow_id, params)

            logger.info(f"Submitting image edit workflow {workflow_id} to ComfyUI...")
            response = requests.post(f"{self.base_url}/prompt", json={"prompt": workflow})
            if response.status_code != 200:
                raise Exception(f"Failed to queue workflow: {response.status_code} - {response.text}")

            prompt_id = response.json()["prompt_id"]
            logger.info(f"Queued image edit workflow with prompt_id: {prompt_id}")

            # Wait for completion and get results
            while True:
                history = requests.get(f"{self.base_url}/history/{prompt_id}").json()
                if history.get(prompt_id):
                    outputs = history[prompt_id]["outputs"]
                    logger.info("Image edit workflow outputs: %s", json.dumps(outputs, indent=2))
                    image_url = _extract_output_url(self.base_url, outputs, workflow_id)
                    logger.info(f"Generated edited image URL: {image_url}")
                    return image_url
                time.sleep(1)

        except FileNotFoundError:
            raise Exception(f"Workflow file '{workflow_file}' not found")
        except KeyError as e:
            raise Exception(f"Workflow error - invalid node or input: {e}")
        except requests.RequestException as e:
            raise Exception(f"ComfyUI API error: {e}")
