import base64
import logging
import os
import uuid
from io import BytesIO
from typing import List, Optional, Tuple
from warnings import filterwarnings
import einops
import numpy as np
import openai
import requests
import safetensors.torch as sf
import torch
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    LlamaModel,
    LlamaTokenizerFast,
    SiglipImageProcessor,
    SiglipVisionModel
)
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.hunyuan import (
    encode_prompt_conds,
    vae_decode,
    vae_decode_fake,
    vae_encode
)
from diffusers_helper.memory import (
    DynamicSwapInstaller,
    cpu,
    fake_diffusers_current_device,
    get_cuda_free_memory_gb,
    gpu,
    load_model_as_complete,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    unload_complete_models
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.utils import (
    crop_or_pad_yield_mask,
    generate_timestamp,
    resize_and_center_crop,
    save_bcthw_as_mp4,
    soft_append_bcthw,
    state_dict_offset_merge,
    state_dict_weighted_merge
)

filterwarnings("ignore")

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

# Load environment variables
load_dotenv()  

# Provider clients
clientOpenAI = openai.OpenAI(api_key=os.getenv("GPT_TOKEN"))
clientGemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
clientHF = InferenceClient(api_key=os.environ.get("HF_TOKEN"))


# ======================================================================
# Function: gen_ImgPrompt
# ======================================================================
def gen_ImgPrompt(session_id: str, all_content: str, summarize_text: str):
    try:
        if not summarize_text:
            logger.error("gen_ImgPrompt: summarize_text is empty (session_id=%s)", session_id)
            raise ValueError("summarize_text is required")

        logger.debug("gen_ImgPrompt: session_id=%s summarize_text=%s", session_id, _truncate(summarize_text, 500))
        logger.debug("gen_ImgPrompt: all_content=%s", _truncate(all_content, 2000))
        logger.debug("gen_ImgPrompt: calling Gemini API...")

        try:
            model = "gemini-2.5-flash"
            prompt_text = (
                "You will silently reason in two stages, then output only a single, final line.\n"
                "Stage 1 — Environment (from Full Story Summary): infer a consistent, uniform scene setting "
                "(time/place, mood, lighting/weather, background elements, color palette, art/camera style) that will frame the image.\n"
                "Stage 2 — Focus (from Short Message): extract the primary subject and action; place them within the Stage 1 environment, "
                "specifying composition/POV, key objects, and critical visual details.\n\n"
                f"Full Story Summary (use ONLY to define the environment):\n{all_content}\n\n"
                f"Short Message (use to define the main subject/action):\n{summarize_text}\n\n"
                "Output: one concise English text to image prompt on a single line that merges the uniform environment and the focused subject; "
                "be vivid and specific (setting, subject, action, composition, style/camera cues), avoid meta language like 'prompt:' or 'image of', "
                "and do not include your reasoning steps or any extra text."
            )

            stream = clientGemini.models.generate_content_stream(
                model=model,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])],
                config=types.GenerateContentConfig(),
            )

            chunks = []
            for ch in stream:
                if getattr(ch, "text", None):
                    chunks.append(ch.text)

            img_prompt = " ".join("".join(chunks).strip().split())

        except Exception:
            logger.exception("gen_ImgPrompt: Gemini API call failed (session_id=%s)", session_id)
            raise

        logger.info("gen_ImgPrompt: generated image prompt (session_id=%s): %s", session_id, _truncate(img_prompt, 500))
        return img_prompt

    except ValueError:
        logger.exception("gen_ImgPrompt: validation error (session_id=%s)", session_id)
        raise
    except Exception:
        logger.exception("gen_ImgPrompt: unexpected error (session_id=%s)", session_id)
        raise


# Allowed providers
ALLOWED_PROVIDERS = {"dalle3", "gemini", "sdxl"}


# ======================================================================
# Utility: save raw image bytes as JPEG
# ======================================================================
def _save_image_bytes_as_jpeg(image_bytes: bytes, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with Image.open(BytesIO(image_bytes)) as im:
            im = im.convert("RGB")
            im.save(out_path, format="JPEG", quality=95)
            logger.debug("_save_image_bytes_as_jpeg: saved at %s", out_path)
    except Exception as pil_err:
        # fallback raw write
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        logger.warning("_save_image_bytes_as_jpeg: PIL decode failed, wrote raw bytes. err=%s", pil_err)


# ======================================================================
# Provider-specific generator: OpenAI DALL·E 3
# ======================================================================
def _gen_with_dalle3(prompt: str, size: str) -> bytes:
    response = clientOpenAI.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size or "1024x1024",
        quality="standard",
        n=1
    )

    if not response or not getattr(response, "data", None):
        raise ValueError("_gen_with_dalle3: empty response from OpenAI")

    image_url = getattr(response.data[0], "url", None)
    if not image_url:
        raise ValueError("_gen_with_dalle3: missing image URL in response")

    dl = requests.get(image_url, timeout=60)
    if dl.status_code != 200:
        raise ValueError(f"_gen_with_dalle3: download failed (HTTP {dl.status_code})")

    img_bytes = dl.content
    if not img_bytes:
        raise ValueError("_gen_with_dalle3: downloaded image is empty")

    logger.debug("_gen_with_dalle3: downloaded image bytes (%s)", len(img_bytes))
    return img_bytes


# ======================================================================
# Provider-specific generator: Gemini (Imagen 4.0)
# ======================================================================
def _gen_with_gemini(prompt: str) -> bytes:
    resp = clientGemini.models.generate_images(
        model="imagen-4.0-generate-001",
        prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=1),
    )
    if not resp or not getattr(resp, "generated_images", None):
        logger.error("_gen_with_gemini: no images returned")
        raise ValueError("No image returned from Gemini")

    generated = resp.generated_images[0]
    if hasattr(generated, "image") and hasattr(generated.image, "bytes"):
        logger.debug("_gen_with_gemini: response contains image bytes")
        return generated.image.bytes
    if hasattr(generated, "bytes"):
        logger.debug("_gen_with_gemini: response contains raw bytes")
        return generated.bytes

    raise ValueError("Gemini: cannot locate image bytes in response")


# ======================================================================
# Provider-specific generator: SDXL (Stable Diffusion XL)
# ======================================================================
def _gen_with_sdxl(prompt: str) -> bytes:
    pil_img = clientHF.text_to_image(
        prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0",
    )
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    logger.debug("_gen_with_sdxl: generated image with SDXL (bytes=%s)", len(buf.getvalue()))
    return buf.getvalue()


# ======================================================================
# Main function: gen_Image
# ======================================================================
def gen_Image(
    sence_index: int,
    session_id: str,
    image_prompt: str,
    *,
    size: str = "1024x1024",
    provider: str,
) -> Tuple[Optional[str], Optional[str]]:
    img_name, img_path = None, None
    try:
        pv = (provider or "").lower().strip()
        if pv not in ALLOWED_PROVIDERS:
            raise ValueError(f"gen_Image: unsupported provider: {pv}")

        logger.debug("gen_Image: start (session_id=%s, provider=%s, prompt=%s)",
                     session_id, pv, _truncate(image_prompt, 200))

        if pv == "dalle3":
            image_bytes = _gen_with_dalle3(image_prompt, size=size)
        elif pv == "gemini":
            image_bytes = _gen_with_gemini(image_prompt)
        else:
            image_bytes = _gen_with_sdxl(image_prompt)

        img_folder = os.path.join(SERVICE_DIR, IMAGE_DIR, session_id)
        os.makedirs(img_folder, exist_ok=True)
        img_name = f"{session_id}_{sence_index}.jpg"
        img_path = os.path.join(img_folder, img_name)
        _save_image_bytes_as_jpeg(image_bytes, img_path)

        logger.info("gen_Image: saved image (session_id=%s) at %s", session_id, img_path)

    except Exception:
        logger.exception("gen_Image: failed (session_id=%s)", session_id)
        return None, None

    return img_name, img_path


def gen_FramePackPrompt(session_id: str, summarize_text: str, image_path: str) -> Optional[str]:
    try:
        if not summarize_text:
            raise ValueError("gen_FramePackPrompt: missing summarize_text")

        if not os.path.exists(image_path):
            raise ValueError(f"gen_FramePackPrompt: image path not found: {image_path}")

        img_folder = os.path.join(SERVICE_DIR, IMAGE_DIR, session_id)
        img_name = image_path
        img_path = os.path.join(img_folder, img_name)
        logger.debug("gen_FramePackPrompt: using image %s", img_path)

        image = Image.open(img_path)

        control_prompt = (
            "You will silently reason in two stages, but output only the final result.\n\n"
            "Stage 1 — Environment (from Full Story Summary): infer a consistent, cinematic environment "
            "(time/place, mood, weather/lighting, background elements, color palette, art/camera style).\n"
            "Stage 2 — Focus (from Short Message + Image): extract the main subject and action, "
            "then place it within the Stage 1 environment. Define composition, POV, objects, and style cues.\n\n"
            f"Full Story Summary (for environment only):\n{summarize_text}\n\n"
            "Short Message (for subject/action):\n"
            f"{summarize_text}\n\n"
            "Rules:\n"
            "- Only output one final vivid Image-to-Video prompt in English.\n"
            "- Do not include reasoning steps or the words 'Stage 1'/'Stage 2'.\n"
            "- Make it cinematic, specific, and natural — not meta."
        )

        response = clientGemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, control_prompt],
            config=types.GenerateContentConfig()
        )

        if response and response.text:
            final_prompt = " ".join(response.text.strip().split())
        else:
            raise ValueError("gen_FramePackPrompt: empty response from Gemini")

        logger.info("gen_FramePackPrompt: generated (session_id=%s): %s", session_id, _truncate(final_prompt, 500))
        return final_prompt

    except Exception:
        logger.exception("gen_FramePackPrompt: failed (session_id=%s)", session_id)
        return None


@torch.no_grad()
def run_Framepack(
    session_id: str,
    video_prompt: str,
    image_name: str,
    total_second_length: int,
    latent_window_size: int,
):
    result = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(
            os.path.join(os.path.dirname(__file__), './hf_download')
        ))

        # Load reference image
        try:
            img_path = os.path.join(SERVICE_DIR, IMAGE_DIR, session_id, image_name)
            with open(img_path, "rb") as f:
                read_input_image = Image.open(f).convert("RGB")
            logger.debug("run_Framepack: loaded image %s", img_path)
        except Exception:
            logger.exception("run_Framepack: failed to load image (session_id=%s)", session_id)
            return None

        # Prepare parameters
        seed = 313447
        steps = 10
        cfg = 1
        gs = 10
        rs = 0.0
        gpu_memory_preservation = 10
        use_teacache = "y"
        mp4_crf = 16
        n_prompt = ""
        prompt = video_prompt
        input_image_path = img_path

        # Device
        gpu_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug("run_Framepack: using device %s", gpu_dev)

        random_uid = str(uuid.uuid4())
        try:
            output_dir = os.path.join(SERVICE_DIR, "framepack_outputs", session_id)
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            logger.exception("run_Framepack: failed to create output_dir")
            raise

        # Load models (แต่ละบล็อคแยก try เพื่อ log ให้ตรง)
        try:
            text_encoder = LlamaModel.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='text_encoder',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).cpu().eval()
        except Exception:
            logger.exception("run_Framepack: load text_encoder failed")
            raise
        try:
            text_encoder_2 = CLIPTextModel.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='text_encoder_2',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).cpu().eval()
        except Exception:
            logger.exception("run_Framepack: load text_encoder_2 failed")
            raise
        try:
            tokenizer = LlamaTokenizerFast.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='tokenizer'
            )
        except Exception:
            logger.exception("run_Framepack: load LlamaTokenizerFast failed")
            raise
        try:
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='tokenizer_2'
            )
        except Exception:
            logger.exception("run_Framepack: load CLIPTokenizer failed")
            raise
        try:
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='vae',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).cpu().eval()
            vae.enable_slicing()
            vae.enable_tiling()
        except Exception:
            logger.exception("run_Framepack: load VAE failed")
            raise
        try:
            feature_extractor = SiglipImageProcessor.from_pretrained(
                "lllyasviel/flux_redux_bfl",
                subfolder='feature_extractor'
            )
        except Exception:
            logger.exception("run_Framepack: load feature_extractor failed")
            raise
        try:
            image_encoder = SiglipVisionModel.from_pretrained(
                "lllyasviel/flux_redux_bfl",
                subfolder='image_encoder',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).cpu().eval()
        except Exception:
            logger.exception("run_Framepack: load image_encoder failed")
            raise
        try:
            transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                'lllyasviel/FramePackI2V_HY',
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            ).cpu().eval()
        except Exception:
            logger.exception("run_Framepack: load transformer failed")
            raise

        try:
            DynamicSwapInstaller.install_model(transformer, device=device)
        except Exception:
            logger.exception("run_Framepack: install transformer to device failed")
            raise
        try:
            DynamicSwapInstaller.install_model(text_encoder, device=device)
        except Exception:
            logger.exception("run_Framepack: install text_encoder to device failed")
            raise
        try:
            transformer.high_quality_fp32_output_for_inference = True
        except Exception:
            logger.exception("run_Framepack: set FP32 output failed")
            raise
        try:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        except Exception:
            logger.exception("run_Framepack: unload models failed")
            raise

        # text encoding
        try:
            fake_diffusers_current_device(text_encoder, gpu_dev)
            load_model_as_complete(text_encoder_2, target_device=gpu_dev)

            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        except Exception:
            logger.exception("run_Framepack: text encoding failed")
            raise

        # input image encoding
        try:
            image_id = uuid.uuid4()
            input_image = read_input_image
            W, H = input_image.size
            height, width = find_nearest_bucket(H, W, resolution=640)
            input_image_np = resize_and_center_crop(input_image_path, target_width=width, target_height=height)

            Image.fromarray(input_image_np).save(os.path.join(output_dir, f'{str(image_id)[:8]}.png'))

            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        except Exception:
            logger.exception("run_Framepack: input image encoding failed")
            raise

        # VAE encoding
        try:
            load_model_as_complete(vae, target_device=gpu_dev)
            start_latent = vae_encode(input_image_pt, vae)

            load_model_as_complete(image_encoder, target_device=gpu_dev)
            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        except Exception:
            logger.exception("run_Framepack: VAE/vision encoding failed")
            raise

        # Dtype
        try:
            llama_vec = llama_vec.to(transformer.dtype)
            llama_vec_n = llama_vec_n.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        except Exception:
            logger.exception("run_Framepack: dtype cast failed")
            raise

        # Sampling
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8),
            dtype=torch.float32
        ).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            logger.debug("run_Framepack: latent_padding_size=%s, is_last_section=%s",
                         latent_padding_size, is_last_section)

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            (
                clean_latent_indices_pre,
                blank_indices,
                latent_indices,
                clean_latent_indices_post,
                clean_latent_2x_indices,
                clean_latent_4x_indices
            ) = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)

            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            unload_complete_models()
            move_model_to_device_with_memory_preservation(
                transformer,
                target_device=gpu_dev,
                preserved_memory_gb=gpu_memory_preservation
            )

            transformer.initialize_teacache(
                enable_teacache=use_teacache,
                num_steps=steps
            )

            def callback(d):
                try:
                    preview = vae_decode_fake(d['denoised'])
                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    # ไม่ log รูป preview เพื่อลด I/O
                except Exception:
                    logger.debug("run_Framepack: preview callback step=%s failed", d.get('i'))

            unload_complete_models()
            torch.cuda.empty_cache()

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu_dev,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat(
                    [start_latent.to(generated_latents), generated_latents],
                    dim=2
                )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            offload_model_from_device_for_memory_preservation(
                transformer,
                target_device=gpu_dev,
                preserved_memory_gb=8
            )
            load_model_as_complete(vae, target_device=gpu_dev)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3
                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
            unload_complete_models()

            logger.debug("run_Framepack: decoded; latent=%s, pixels=%s",
                         tuple(real_history_latents.shape), tuple(history_pixels.shape))
            if is_last_section:
                output_filename = os.path.join(output_dir, f'{random_uid}.mp4')
                save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
                break

        # เซฟภาพอ้างอิง (บรรทัดเดิมใช้ Image.save ซึ่งไม่ถูกต้อง)
        try:
            read_input_image.save(os.path.join(output_dir, "reference_image.jpg"))
        except Exception:
            logger.debug("run_Framepack: save reference image failed", exc_info=True)

        result = output_filename
        logger.info("run_Framepack: finished output=%s (session_id=%s)", result, session_id)

        # Clear memory
        import gc
        def clear_gpu_memory():
            try:
                torch.cuda.empty_cache()
                gc.collect()
                logger.debug("run_Framepack: GPU memory cleared")
            except Exception:
                logger.debug("run_Framepack: GPU memory clear failed", exc_info=True)
        clear_gpu_memory()

    except ValueError:
        logger.exception("run_Framepack: ValueError (session_id=%s)", session_id)
    except Exception:
        logger.exception("run_Framepack: unexpected error (session_id=%s)", session_id)

    return result
