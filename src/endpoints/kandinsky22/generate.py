import os
import time
from PIL import Image
from diffusers import (
    KandinskyV22Pipeline,
    KandinskyV22Img2ImgPipeline,
    KandinskyV22InpaintPipeline,
)
from typing import Any, List, cast
import torch
import logging
from src.shared.classes import (
    GenerateFunctionProps,
    GenerateOutput,
)
from src.shared.device import DEVICE_CUDA
from src.shared.helpers import (
    crop_images,
    download_and_fit_image,
    download_and_fit_image_mask,
    log_gpu_memory,
    pad_image_mask_nd,
    pad_image_pil,
)
from src.shared.pipe_classes import Kandinsky22PipeObject
from src.shared.schedulers import KANDINSKY_22_SCHEDULERS

PRIOR_STEPS = 25
PRIOR_GUIDANCE_SCALE = 4.0


kandinsky_2_2_negative_prompt_prefix = "overexposed"


def get_scheduler(
    name: str,
    pipeline: (
        KandinskyV22Pipeline | KandinskyV22InpaintPipeline | KandinskyV22Img2ImgPipeline
    ),
):
    if "from_config" in KANDINSKY_22_SCHEDULERS[name]:
        return KANDINSKY_22_SCHEDULERS[name]["scheduler"].from_config(
            pipeline.scheduler.config
        )
    else:
        return KANDINSKY_22_SCHEDULERS[name]["scheduler"]()


def generate(
    props: GenerateFunctionProps[Kandinsky22PipeObject],
) -> List[GenerateOutput]:
    # Props
    input = props.input
    pipe_object = props.pipe_object
    model_name = props.model_name
    # ---------------------------------------------------------------------

    inference_start = time.time()
    log_gpu_memory(message="Before inference")

    prompt = input.prompt
    negative_prompt = input.negative_prompt
    seed = input.seed

    if seed is None:
        seed = int.from_bytes(os.urandom(3), "big")
        logging.info(f"Using seed: {seed}")
    generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed)

    if input.prompt_prefix is not None:
        prompt = f"{input.prompt_prefix} {input.prompt}"

    if input.negative_prompt_prefix is not None:
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = input.negative_prompt_prefix
        else:
            negative_prompt = f"{input.negative_prompt_prefix} {input.negative_prompt}"

    if negative_prompt is None or negative_prompt == "":
        negative_prompt = kandinsky_2_2_negative_prompt_prefix
    else:
        negative_prompt = f"{kandinsky_2_2_negative_prompt_prefix}, {negative_prompt}"

    logging.info(f"Negative prompt for Kandinsky 2.2: {negative_prompt}")

    output_images: List[Image.Image] = []

    if (
        input.init_image_url is not None
        and input.mask_image_url is not None
        and pipe_object.inpaint is not None
    ):
        pipe_object.inpaint.scheduler = get_scheduler(
            input.scheduler, pipe_object.inpaint
        )
        start = time.time()
        init_image = download_and_fit_image(
            input.init_image_url, input.width, input.height
        )
        init_image = pad_image_pil(init_image, 64)
        end = time.time()
        logging.info(
            f"-- Downloaded and cropped init image in: {round((end - start) * 1000)}ms"
        )
        start = time.time()
        mask_image = download_and_fit_image_mask(
            url=input.mask_image_url,
            width=input.width,
            height=input.height,
        )
        mask_image = pad_image_mask_nd(mask_image, 64, 0)
        end = time.time()
        logging.info(
            f"-- Downloaded and cropped mask image in: {round((end - start) * 1000)}ms"
        )
        for i in range(input.num_outputs):
            generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed + i)
            img_emb = cast(
                Any,
                pipe_object.prior(
                    prompt=prompt,
                    num_inference_steps=PRIOR_STEPS,
                    guidance_scale=PRIOR_GUIDANCE_SCALE,
                    num_images_per_prompt=1,
                    generator=generator,
                ),
            )
            neg_emb = cast(
                Any,
                pipe_object.prior(
                    prompt=negative_prompt,
                    num_inference_steps=PRIOR_STEPS,
                    guidance_scale=PRIOR_GUIDANCE_SCALE,
                    num_images_per_prompt=1,
                    generator=generator,
                ),
            )
            out = cast(
                Any,
                pipe_object.inpaint(
                    image=cast(Any, [init_image] * 1),
                    mask_image=cast(Any, [mask_image] * 1),
                    image_embeds=img_emb.image_embeds,
                    negative_image_embeds=neg_emb.image_embeds,
                    width=init_image.width,
                    height=init_image.height,
                    num_inference_steps=input.num_inference_steps,
                    guidance_scale=input.guidance_scale,
                    generator=generator,
                ),
            ).images[0]
            output_images.append(out)
    elif input.init_image_url is not None and input.prompt_strength is not None:
        pipe_object.text2img.scheduler = get_scheduler(
            input.scheduler, pipe_object.text2img
        )
        start = time.time()
        init_image = download_and_fit_image(
            input.init_image_url, input.width, input.height
        )
        end = time.time()
        logging.info(
            f"-- Downloaded and cropped init image in: {round((end - start) * 1000)}ms"
        )
        start = time.time()
        images_and_texts = [prompt, init_image]
        weights = [input.prompt_strength, 1 - input.prompt_strength]
        for i in range(input.num_outputs):
            generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed + i)
            prior_out = pipe_object.prior.interpolate(
                images_and_texts,
                weights,
                negative_prompt=negative_prompt,
                num_inference_steps=PRIOR_STEPS,
                guidance_scale=PRIOR_GUIDANCE_SCALE,
                num_images_per_prompt=1,
                generator=generator,
            )
            out = cast(
                Any,
                pipe_object.text2img(
                    **prior_out,
                    width=input.width,
                    height=input.height,
                    num_inference_steps=input.num_inference_steps,
                    guidance_scale=input.guidance_scale,
                    generator=generator,
                ),
            ).images[0]
            output_images.append(out)
    else:
        pipe_object.text2img.scheduler = get_scheduler(
            input.scheduler, pipe_object.text2img
        )
        for i in range(input.num_outputs):
            generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed + i)
            img_emb = cast(
                Any,
                pipe_object.prior(
                    prompt=prompt,
                    num_inference_steps=PRIOR_STEPS,
                    guidance_scale=PRIOR_GUIDANCE_SCALE,
                    num_images_per_prompt=1,
                    generator=generator,
                ),
            )
            neg_emb = cast(
                Any,
                pipe_object.prior(
                    prompt=negative_prompt,
                    num_inference_steps=PRIOR_STEPS,
                    guidance_scale=PRIOR_GUIDANCE_SCALE,
                    num_images_per_prompt=1,
                    generator=generator,
                ),
            )
            out = cast(
                Any,
                pipe_object.text2img(
                    num_inference_steps=input.num_inference_steps,
                    guidance_scale=input.guidance_scale,
                    width=input.width,
                    height=input.height,
                    generator=generator,
                    image_embeds=img_emb.image_embeds,
                    negative_image_embeds=neg_emb.image_embeds,
                ),
            ).images[0]
            output_images.append(out)

    output_images = crop_images(
        images=output_images, width=input.width, height=input.height
    )

    inference_end = time.time()
    logging.info(
        f"🔮 🟢 Inference | {model_name} | {input.num_outputs} image(s) | {round((inference_end - inference_start) * 1000)}ms"
    )
    log_gpu_memory(message="After inference")

    outputs: List[GenerateOutput] = []

    for image in output_images:
        outputs.append(GenerateOutput(image=image))

    return outputs
