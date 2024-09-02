import torch
from .constants import DEFAULT_LORA_ID, MODEL_ID, REFINER_ID, VARIANT
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
        variant=VARIANT,
    )
    text2img.load_lora_weights(
        MODEL_ID,
        weight_name=DEFAULT_LORA_ID,
    )

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

    inpaint = None

    return StableDiffusionPipeObject(
        text2img=cast(StableDiffusionXLPipeline, text2img),
        img2img=cast(StableDiffusionXLImg2ImgPipeline, img2img),
        refiner=cast(StableDiffusionXLImg2ImgPipeline, refiner),
        inpaint=inpaint,
    )


if __name__ == "__main__":
    get_pipe_object(to_cuda=False)
