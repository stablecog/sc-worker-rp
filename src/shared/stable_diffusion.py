import math
import os
from typing import Any, List, cast
import torch
from src.shared.classes import (
    GenerateFunctionProps,
    GenerateOutput,
    StableDiffusionPipeObject,
)
from src.shared.constants import DEVICE_CUDA
import time
from src.shared.helpers import (
    download_and_fit_image,
    log_gpu_memory,
)
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
)
import logging
from PIL import Image
from src.shared.schedulers import SD_SCHEDULERS


def get_scheduler(name, config):
    return SD_SCHEDULERS[name]["from_config"](config)


def generate(
    props: GenerateFunctionProps[StableDiffusionPipeObject],
) -> List[GenerateOutput]:
    # Props
    input = props.input
    pipe_object = props.pipe_object
    model_name = props.model_name
    default_prompt_prefix = props.default_prompt_prefix
    default_negative_prompt_prefix = props.default_negative_prompt_prefix
    dont_set_scheduler = props.dont_set_scheduler
    # ---------------------------------------------------------------------

    inference_start = time.time()
    prompt = input.prompt
    negative_prompt = input.negative_prompt
    seed = input.seed

    inference_start = time.time()

    if seed is None:
        seed = int.from_bytes(os.urandom(3), "big")
        logging.info(f"Using seed: {seed}")
    generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed)

    selected_prompt_prefix = (
        input.prompt_prefix
        if input.prompt_prefix is not None
        else default_prompt_prefix
    )
    if selected_prompt_prefix is not None:
        prompt = f"{selected_prompt_prefix} {prompt}"

    selected_negative_prompt_prefix = (
        input.negative_prompt_prefix
        if input.negative_prompt_prefix is not None
        else default_negative_prompt_prefix
    )

    if selected_negative_prompt_prefix is not None:
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = selected_negative_prompt_prefix
        else:
            negative_prompt = f"{selected_negative_prompt_prefix} {negative_prompt}"

    extra_kwargs = {}
    pipe_selected: (
        StableDiffusionPipeline
        | StableDiffusionImg2ImgPipeline
        | StableDiffusionImg2ImgPipeline
        | StableDiffusionXLPipeline
        | StableDiffusionXLImg2ImgPipeline
        | StableDiffusionInpaintPipeline
        | StableDiffusion3Pipeline
        | None
    ) = None

    if pipe_object.refiner is not None:
        extra_kwargs["output_type"] = "latent"

    if input.init_image_url is not None and (
        pipe_object.img2img is not None or pipe_object.inpaint is not None
    ):
        start_i = time.time()
        extra_kwargs["image"] = download_and_fit_image(
            url=input.init_image_url,
            width=input.width,
            height=input.height,
        )
        extra_kwargs["strength"] = input.prompt_strength
        end_i = time.time()
        logging.info(
            f"-- Downloaded and cropped init image in: {round((end_i - start_i) * 1000)}ms"
        )

        if pipe_object.inpaint is not None and input.mask_image_url is not None:
            # The process is: inpainting
            pipe_selected = cast(StableDiffusionInpaintPipeline, pipe_object.inpaint)
            start_i = time.time()
            extra_kwargs["mask_image"] = download_and_fit_image(
                url=input.mask_image_url,
                width=input.width,
                height=input.height,
            )
            extra_kwargs["strength"] = 0.99
            end_i = time.time()
            logging.info(
                f"-- Downloaded and cropped mask image in: {round((end_i - start_i) * 1000)}ms"
            )
        elif pipe_object.img2img is not None:
            # The process is: img2img
            pipe_selected = pipe_object.img2img
    else:
        # The process is: text2img
        pipe_selected = pipe_object.text2img
        extra_kwargs["width"] = input.width
        extra_kwargs["height"] = input.height

    if pipe_selected is None:
        raise ValueError("No pipeline selected")

    if dont_set_scheduler is False:
        pipe_selected.scheduler = get_scheduler(
            input.scheduler, pipe_selected.scheduler.config
        )

    output_images: List[Image.Image] = []

    for i in range(input.num_outputs):
        generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed + i)
        out = cast(
            Any,
            pipe_selected(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=input.guidance_scale,
                generator=generator,
                num_images_per_prompt=1,
                num_inference_steps=input.num_inference_steps,
                **extra_kwargs,
            ),
        ).images[0]
        output_images.append(out)

    log_gpu_memory(message="After inference")

    if pipe_object.refiner is not None:
        half_steps = math.floor(input.num_inference_steps / 2)
        refiner_steps = half_steps + (5 - half_steps % 5)
        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": input.guidance_scale,
            "num_inference_steps": refiner_steps,
        }

        s = time.time()
        for i in range(len(output_images)):
            generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed + i)
            image = output_images[i]
            out_image = cast(
                Any,
                pipe_object.refiner(
                    **args,
                    image=image,
                    generator=generator,
                    num_images_per_prompt=1,
                ),
            ).images[0]
            output_images[i] = out_image
        e = time.time()
        logging.info(
            f"🖌️ Refined {len(output_images)} image(s) in: {round((e - s) * 1000)}ms"
        )

    outputs: List[GenerateOutput] = []
    for image in output_images:
        outputs.append(GenerateOutput(image=image))

    inference_end = time.time()
    logging.info(
        f"🔮 🟢 Inference | {model_name} | {input.num_outputs} image(s) | {round((inference_end - inference_start) * 1000)}ms"
    )

    return outputs
