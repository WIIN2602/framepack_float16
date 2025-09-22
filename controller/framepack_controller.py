import os
import time
import math
import uuid
import argparse
import traceback
import asyncio
from typing import Optional, List
import uvicorn
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from services import *
import logging
from pydantic import BaseModel

# ---------- Create folder ----------
# Service Directory
SERVICE_DIR = "userdata"
VID_SRC_DIR = "sources"
VID_SCN_DIR = "scenes"
VID_SUM_DIR = "results"
SCRIPT_DIR = "scripts"
CACHE_DIR = "cache"
SCRIPT_SUM_DIR = "script_sum"
KEY_SUM_DIR = "key_sum"
VOICE_DIR = "voice"
IMAGE_DIR = "images"
FRAMEPACK_DIR = "framepack_outputs"

# List of directories to create inside SERVICE_DIR
folders = [
    VID_SRC_DIR,
    VID_SCN_DIR,
    VID_SUM_DIR,
    CACHE_DIR,
    SCRIPT_DIR,
    VOICE_DIR,
    IMAGE_DIR,
    FRAMEPACK_DIR,
]

# Create the base service directory
os.makedirs(SERVICE_DIR, exist_ok=True)

# Create subdirectories
for folder in folders:
    os.makedirs(os.path.join(SERVICE_DIR, folder), exist_ok=True)
# ----------------------------

# ---------- Logger ----------
log_file_path = "server.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _truncate(s: str, limit: int = 1000) -> str:
    if s is None:
        return ""
    return (s[:limit] + "... [truncated]") if len(s) > limit else s
# ----------------------------

# ---------- Model ----------
class imgPTRequest(BaseModel):
    session_id: str
    all_content: str
    summarize_text: str

class createIMGRequest(BaseModel):
    sence_index: int
    session_id: str
    image_prompt: str
    provider: Optional[str]

class frampackPTRequest(BaseModel):
    session_id: str
    summarize_text: str
    image_path: str
    
class createVDORequest(BaseModel):
    session_id: str
    video_prompt: str
    image_name: str
    total_second_length: int
    latent_window_size: int 
# ----------------------------


# -------------------------------------------
# Create a FastAPI router – this file acts as a "controller"
# -------------------------------------------
FramePackStream = APIRouter()

# -------------------------------------------
# Endpoint: hello (for testing)
# -------------------------------------------
@FramePackStream.get("/hello")
async def root():
    logger.info("Message form root path in float16.cloud")
    return JSONResponse(content={"message": "Message form root path in float16.cloud"})

# ===================================================================
# Endpoint: /generate-image-prompt
# ===================================================================
@FramePackStream.post("/generate-image-prompt")
async def generate_imgprompt(
    request: imgPTRequest
):
    try:
        img_prompt = gen_ImgPrompt(request.session_id, request.all_content, request.summarize_text)

        logger.info(f"[{request.session_id}] Image prompt generated successfully")
        return JSONResponse(
            content={
                "message": "Generate prompt for generate image successfully",
                "image_prompt": img_prompt
            }
        )

    except ValueError as e:
        logger.error(f"[{request.session_id}] generate_imgprompt failed: {str(e)}")
        return JSONResponse(status_code=404, content={"error": str(e)})

    except Exception as e:
        logger.exception(f"[{request.session_id}] Unexpected error in generate_imgprompt")  # includes stack trace
        return JSONResponse(status_code=500, content={"error": "Unexpected error during image prompt generation"})


# ===================================================================
# Endpoint: /generate_image
# ===================================================================
DEFAULT_PROVIDER = "sdxl"
ALLOWED_PROVIDERS = {"sdxl", "gemini", "dalle3"}

@FramePackStream.post("/generate_image")
async def generate_image(
    request: createIMGRequest
):
    try:

        norm_provider = (request.provider or "").strip().lower() or DEFAULT_PROVIDER
        if norm_provider not in ALLOWED_PROVIDERS:
            logger.warning(f"[{request.session_id}] Invalid provider '{request.provider}'")
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid provider '{request.provider}'. Allowed: {sorted(ALLOWED_PROVIDERS)}"}
            )

        image_name, image_path = gen_Image(
            sence_index=request.sence_index,
            session_id=request.session_id,
            image_prompt=request.image_prompt,
            provider=norm_provider,
            size="1024x1024",
        )

        if image_path:
            logger.info(f"[{request.session_id}] Image generated: {image_name}")
            return FileResponse(image_path, media_type="image/jpeg", filename=image_name)

        logger.error(f"[{request.session_id}] Image generation returned no result")
        return JSONResponse(status_code=500, content={"error": "Image generation returned no result"})

    except Exception as e:
        logger.exception(f"[{request.session_id}] Internal error in generate_image")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error while generating image", "details": str(e)},
        )


# ===================================================================
# Endpoint: /generate-framepack-prompt
# ===================================================================
@FramePackStream.post("/generate-framepack-prompt")
async def generate_FPprompt(
    request: frampackPTRequest
):
    try:
        fp_prompt = gen_FramePackPrompt(request.session_id, request.summarize_text, request.image_path)

        logger.info(f"[{request.session_id}] Framepack prompt generated successfully")
        return JSONResponse(
            content={
                "message": "Generate prompt for generate framepack successfully",
                "framepack_prompt": fp_prompt
            }
        )
    except ValueError as e:
        logger.error(f"[{request.session_id}] generate_FPprompt failed: {str(e)}")
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        logger.exception(f"[{request.session_id}] Unexpected error in generate_FPprompt")
        return JSONResponse(status_code=500, content={"error": "Unexpected error during framepack prompt generation"})  


# ===================================================================
# Endpoint: /generate_video
# ===================================================================
@FramePackStream.post("/generate_video")
async def generate_video(
    request: createVDORequest
):
    try:
        run_fp = run_Framepack(
            session_id=request.session_id,
            video_prompt=request.video_prompt,
            image_name=request.image_name,
            total_second_length=request.total_second_length,
            latent_window_size=request.latent_window_size
        )

        if run_fp:
            logger.info(f"[{request.session_id}] Video generated successfully: {run_fp}")
            return FileResponse(run_fp, media_type="video/mp4", filename=os.path.basename(run_fp))

        logger.error(f"[{request.session_id}] Framepack returned no result")
        return JSONResponse(status_code=500, content={"error": "Framepack returned no result"})

    except ValueError as e:
        logger.error(f"[{request.session_id}] generate_video failed: {str(e)}")
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        logger.exception(f"[{request.session_id}] Unexpected error in generate_video")
        return JSONResponse(status_code=500, content={"error": str(e)})

