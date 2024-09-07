import torch
from typing import cast
from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
from src.shared.hf_login import login_to_hf
from src.shared.pipe_classes import StableDiffusionPipeObject
from src.shared.device import DEVICE_CUDA

MODEL_NAME = "Stable Diffusion 3"
MODEL_FILE = "https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium_incl_clips_t5xxlfp8.safetensors"


def get_pipe_object(to_cuda: bool = True) -> StableDiffusionPipeObject:
    login_to_hf()

    text2img = StableDiffusion3Pipeline.from_single_file(
        MODEL_FILE,
        torch_dtype=torch.float16,
        add_watermark=False,
        safety_checker=None,
    )

    img2img = None
    if to_cuda:
        text2img = text2img.to(DEVICE_CUDA)
        img2img = StableDiffusion3Img2ImgPipeline(**text2img.components)
    else:
        del text2img
        text2img = None

    return StableDiffusionPipeObject(
        text2img=cast(StableDiffusion3Pipeline, text2img),
        img2img=img2img,
    )


if __name__ == "__main__":
    get_pipe_object(to_cuda=False)
