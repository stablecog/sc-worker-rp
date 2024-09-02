import torch
from src.endpoints.ssd_1b.constants import MODEL_ID, REFINER_ID
from src.shared.classes import StableDiffusionPipeObject
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

from src.shared.constants import DEVICE_CUDA
from typing import cast


def get_pipe_object(to_cuda: bool = True) -> StableDiffusionPipeObject:
    text2img = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        add_watermark=False,
        safety_checker=None,
    )
    img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        add_watermark=False,
        safety_checker=None,
    )
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        REFINER_ID,
        torch_dtype=torch.float16,
        add_watermark=False,
    )
    inpaint = None

    if to_cuda:
        text2img = text2img.to(DEVICE_CUDA)
        img2img = img2img.to(DEVICE_CUDA)
        refiner = refiner.to(DEVICE_CUDA)

    return StableDiffusionPipeObject(
        text2img=cast(StableDiffusionXLPipeline, text2img),
        img2img=cast(StableDiffusionXLImg2ImgPipeline, img2img),
        refiner=cast(StableDiffusionXLImg2ImgPipeline, refiner),
        inpaint=inpaint,
    )
