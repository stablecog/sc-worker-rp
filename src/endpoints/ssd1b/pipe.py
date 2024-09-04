import torch
from typing import cast
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from src.shared.device import DEVICE_CUDA
from src.shared.pipe_classes import StableDiffusionPipeObject

MODEL_NAME = "SSD-1B"
MODEL_ID = "segmind/SSD-1B"
REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
VARIANT = "fp16"


def get_pipe_object(to_cuda: bool = True) -> StableDiffusionPipeObject:
    text2img = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        add_watermark=False,
        safety_checker=None,
        variant=VARIANT,
    )
    print("test")

    img2img = None
    if to_cuda:
        text2img = text2img.to(DEVICE_CUDA)
        img2img = StableDiffusionXLImg2ImgPipeline(**text2img.components)
    else:
        del text2img
        text2img = None

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        REFINER_ID,
        torch_dtype=torch.float16,
        add_watermark=False,
        variant=VARIANT,
    )
    if to_cuda:
        refiner = refiner.to(DEVICE_CUDA)
    else:
        del refiner
        refiner = None

    return StableDiffusionPipeObject(
        text2img=cast(StableDiffusionXLPipeline, text2img),
        img2img=cast(StableDiffusionXLImg2ImgPipeline, img2img),
        refiner=cast(StableDiffusionXLImg2ImgPipeline, refiner),
    )


if __name__ == "__main__":
    get_pipe_object(to_cuda=False)
