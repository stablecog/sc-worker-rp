import torch
from .constants import MODEL_ID, REFINER_ID
from src.shared.classes import StableDiffusionPipeObject
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from src.shared.constants import DEVICE_CUDA
from typing import cast


def get_pipe_object(to_cuda: bool = True) -> StableDiffusionPipeObject | None:
    text2img = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        add_watermark=False,
        safety_checker=None,
    )
    if to_cuda:
        text2img = text2img.to(DEVICE_CUDA)
    else:
        del text2img
        text2img = None

    img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        add_watermark=False,
        safety_checker=None,
    )
    if to_cuda:
        img2img = img2img.to(DEVICE_CUDA)
    else:
        del img2img
        img2img = None

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        REFINER_ID,
        torch_dtype=torch.float16,
        add_watermark=False,
    )
    if to_cuda:
        refiner = refiner.to(DEVICE_CUDA)
    else:
        del refiner
        img2img = None

    inpaint = None

    return StableDiffusionPipeObject(
        text2img=cast(StableDiffusionXLPipeline, text2img),
        img2img=cast(StableDiffusionXLImg2ImgPipeline, img2img),
        refiner=cast(StableDiffusionXLImg2ImgPipeline, refiner),
        inpaint=inpaint,
    )
