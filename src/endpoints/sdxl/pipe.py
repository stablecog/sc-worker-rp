import torch
from typing import cast
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from src.shared.pipe_classes import StableDiffusionPipeObject
from src.shared.device import DEVICE_CUDA

MODEL_NAME = "SDXL"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
VARIANT = "fp16"
DEFAULT_LORA_ID = "sd_xl_offset_example-lora_1.0.safetensors"


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

    return StableDiffusionPipeObject(
        text2img=cast(StableDiffusionXLPipeline, text2img),
        img2img=img2img,
        refiner=refiner,
    )


if __name__ == "__main__":
    get_pipe_object(to_cuda=False)
