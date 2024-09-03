import torch
from typing import cast
from diffusers import (
    KandinskyV22Pipeline,
    KandinskyV22PriorPipeline,
)
from src.shared.classes import Kandinsky22PipeObject
from src.shared.constants import DEVICE_CUDA

MODEL_NAME = "Kandinsky 2.2"
MODEL_ID = "kandinsky-community/kandinsky-2-2-decoder"
PRIOR_MODEL_ID = "kandinsky-community/kandinsky-2-2-prior"


def get_pipe_object(to_cuda: bool = True) -> Kandinsky22PipeObject:
    prior = KandinskyV22PriorPipeline.from_pretrained(
        PRIOR_MODEL_ID,
        torch_dtype=torch.float16,
    )
    if to_cuda:
        prior = prior.to(DEVICE_CUDA)
    else:
        del prior
        prior = None

    text2img = KandinskyV22Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    if to_cuda:
        text2img = text2img.to(DEVICE_CUDA)
    else:
        del text2img
        text2img = None

    return Kandinsky22PipeObject(
        prior=cast(KandinskyV22PriorPipeline, prior),
        text2img=cast(KandinskyV22Pipeline, text2img),
    )


if __name__ == "__main__":
    get_pipe_object(to_cuda=False)
