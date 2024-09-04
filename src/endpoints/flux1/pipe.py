import logging
import torch
from typing import cast
from diffusers import FluxPipeline
from src.shared.pipe_classes import Flux1PipeObject
from src.shared.device import DEVICE_CUDA
import os

MODEL_NAME = "FLUX.1"
MODEL_ID = "black-forest-labs/FLUX.1-schnell"


def get_pipe_object(to_cuda: bool = True) -> Flux1PipeObject:
    text2img = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        add_watermark=False,
        safety_checker=None,
    )

    if to_cuda:
        text2img = text2img.to(DEVICE_CUDA)
    else:
        del text2img
        text2img = None

    return Flux1PipeObject(
        text2img=cast(FluxPipeline, text2img),
    )


if __name__ == "__main__":
    """ get_pipe_object(to_cuda=False) """
    logging.info("🔵 Skip downloading model")
    folder = os.environ.get("HF_DATASETS_CACHE", None)
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
