import logging
import os
from typing import Any, List, cast
from PIL import Image
import torch
from src.shared.classes import GenerateFunctionProps, GenerateOutput
from src.shared.device import DEVICE_CUDA
from src.shared.helpers import log_gpu_memory
from src.shared.pipe_classes import Flux1PipeObject
import time

MAX_INFERENCE_STEPS = 4


def generate(
    props: GenerateFunctionProps[Flux1PipeObject],
) -> List[GenerateOutput]:
    # Props ------------------------------------
    input = props.input
    pipe_object = props.pipe_object
    model_name = props.model_name
    # ------------------------------------------

    inference_start = time.time()
    log_gpu_memory(message="Before inference")

    prompt = input.prompt
    seed = input.seed

    if seed is None:
        seed = int.from_bytes(os.urandom(3), "big")
        logging.info(f"Using seed: {seed}")
    generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed)

    # The process is: text2img
    pipe_selected = pipe_object.text2img

    output_images: List[Image.Image] = []

    for i in range(input.num_outputs):
        generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed + i)
        out = cast(
            Any,
            pipe_selected(
                prompt=prompt,
                generator=generator,
                guidance_scale=0,
                num_images_per_prompt=1,
                num_inference_steps=min(input.num_inference_steps, MAX_INFERENCE_STEPS),
                width=input.width,
                height=input.height,
            ),
        ).images[0]
        output_images.append(out)

    inference_end = time.time()
    logging.info(
        f"🔮 🟢 Inference | {model_name} | {input.num_outputs} image(s) | {round((inference_end - inference_start) * 1000)}ms"
    )
    log_gpu_memory(message="After inference")

    outputs: List[GenerateOutput] = []
    for image in output_images:
        outputs.append(GenerateOutput(image=image))

    return outputs
