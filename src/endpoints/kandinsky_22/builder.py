import torch
from diffusers import (
    KandinskyV22Pipeline,
    KandinskyV22PriorPipeline,
)

from src.endpoints.kandinsky_22.constants import (
    KANDINSKY_22_DECODER_MODEL_ID,
    KANDINSKY_22_PRIOR_MODEL_ID,
)


def main():
    prior = KandinskyV22PriorPipeline.from_pretrained(
        KANDINSKY_22_PRIOR_MODEL_ID,
        torch_dtype=torch.float16,
    )
    text2img = KandinskyV22Pipeline.from_pretrained(
        KANDINSKY_22_DECODER_MODEL_ID,
        torch_dtype=torch.float16,
    )
    return prior, text2img


if __name__ == "__main__":
    main()
