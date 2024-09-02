import torch
from src.endpoints.kandinsky_22.constants import MODEL_ID, PRIOR_MODEL_ID
from src.shared.classes import Kandinsky22PipeObject
from diffusers import (
    KandinskyV22Pipeline,
    KandinskyV22PriorPipeline,
)

from src.shared.constants import DEVICE_CUDA
from typing import cast


def get_pipe_object(to_cuda: bool = True) -> Kandinsky22PipeObject:
    prior = KandinskyV22PriorPipeline.from_pretrained(
        PRIOR_MODEL_ID,
        torch_dtype=torch.float16,
    )
    text2img = KandinskyV22Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    inpaint = None

    if to_cuda:
        prior = prior.to(DEVICE_CUDA)
        text2img = text2img.to(DEVICE_CUDA)

    return Kandinsky22PipeObject(
        prior=cast(KandinskyV22PriorPipeline, prior),
        text2img=cast(KandinskyV22Pipeline, text2img),
        inpaint=inpaint,
    )
