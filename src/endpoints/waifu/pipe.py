import torch
from typing import cast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from src.shared.classes import StableDiffusionPipeObject
from src.shared.constants import DEVICE_CUDA

MODEL_NAME = "Waifu Diffusion"
MODEL_ID = "hakurei/waifu-diffusion"
VARIANT = "fp16"
DEFAULT_PROMPT_PREFIX = "masterpiece, best quality, high quality"
DEFAULT_NEGATIVE_PROMPT_PREFIX = "worst quality, low quality, deleted, nsfw, blurry"
test = "true"


def get_pipe_object(to_cuda: bool = True) -> StableDiffusionPipeObject:
    text2img = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        add_watermark=False,
        safety_checker=None,
        variant=VARIANT,
    )

    img2img = None
    if to_cuda:
        text2img = text2img.to(DEVICE_CUDA)
        img2img = StableDiffusionPipeline(**text2img.components)
    else:
        del text2img
        text2img = None

    return StableDiffusionPipeObject(
        text2img=cast(StableDiffusionPipeline, text2img),
        img2img=cast(StableDiffusionImg2ImgPipeline, img2img),
    )


if __name__ == "__main__":
    get_pipe_object(to_cuda=False)
