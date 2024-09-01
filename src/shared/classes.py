from diffusers import (
    KandinskyV22InpaintPipeline,
    KandinskyV22Pipeline,
    KandinskyV22PriorPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    FluxPipeline,
    StableDiffusion3Pipeline,
)
from typing import Any, List
from pydantic import BaseModel, Field, validator

from src.shared.constants import SIZE_LIST
from src.shared.helpers import return_value_if_in_list
from src.shared.sd import SD_SCHEDULER_DEFAULT
from PIL import Image


class StableDiffusionPipeObject:
    def __init__(
        self,
        text2img: (
            StableDiffusionPipeline
            | StableDiffusionXLPipeline
            | StableDiffusion3Pipeline
        ),
        img2img: (
            StableDiffusionImg2ImgPipeline | StableDiffusionXLImg2ImgPipeline | None
        ),
        inpaint: StableDiffusionInpaintPipeline | None,
        refiner: StableDiffusionXLImg2ImgPipeline | None,
    ):
        self.text2img = text2img
        self.img2img = img2img
        self.inpaint = inpaint
        self.refiner = refiner


class Kandinsky22PipeObject:
    def __init__(
        self,
        prior: KandinskyV22PriorPipeline,
        text2img: KandinskyV22Pipeline,
        inpaint: KandinskyV22InpaintPipeline,
    ):
        self.prior = prior
        self.text2img = text2img
        self.inpaint = inpaint


class Flux1PipeObject:
    def __init__(
        self,
        text2img: FluxPipeline,
    ):
        self.text2img = text2img


class GenerateFunctionInput:
    def __init__(
        self,
        prompt: str,
        negative_prompt: str | None,
        prompt_prefix: str | None,
        negative_prompt_prefix: str | None,
        width: int,
        height: int,
        num_outputs: int,
        num_inference_steps: int,
        guidance_scale: float,
        init_image_url: str | None,
        mask_image_url: str | None,
        prompt_strength: float | None,
        scheduler: Any,
        seed: int,
    ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.prompt_prefix = prompt_prefix
        self.negative_prompt_prefix = negative_prompt_prefix
        self.width = width
        self.height = height
        self.num_outputs = num_outputs
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.init_image_url = init_image_url
        self.mask_image_url = mask_image_url
        self.prompt_strength = prompt_strength
        self.scheduler = scheduler
        self.seed = seed


class GenerateFunctionOutput:
    def __init__(self, image: Image.Image):
        self.image = image


class GenerateFunctionInputKandinsky22(GenerateFunctionInput):
    def __init__(self, *args, pipe_object: Kandinsky22PipeObject, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_object = pipe_object


class GenerateFunctionInputStableDiffusion(GenerateFunctionInput):
    def __init__(self, *args, pipe_object: StableDiffusionPipeObject, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_object = pipe_object


class GenerateFunctionInputFlux1(GenerateFunctionInput):
    def __init__(self, *args, pipe_object: Flux1PipeObject, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_object = pipe_object


class PredictGenerateInput(BaseModel):
    prompt: str = Field(description="Input prompt.", default="")
    negative_prompt: str = Field(description="Input negative prompt.", default="")
    num_outputs: int = Field(
        description="Number of images to output. If the NSFW filter is triggered, you may get fewer outputs than this.",
        ge=1,
        le=4,
        default=1,
    )
    init_image_url: str = Field(
        description="Init image url to be used with img2img.",
        default=None,
    )
    mask_image_url: str = Field(
        description="Mask image url to be used with img2img.",
        default=None,
    )
    prompt_strength: float = Field(
        description="The strength of the prompt when using img2img, between 0-1. When 1, it'll essentially ignore the image.",
        ge=0,
        le=1,
        default=None,
    )
    num_inference_steps: int = Field(
        description="Number of denoising steps", ge=1, le=500, default=30
    )
    guidance_scale: float = Field(
        description="Scale for classifier-free guidance.", ge=1, le=20, default=7.5
    )
    scheduler: str = Field(
        default=SD_SCHEDULER_DEFAULT,
        description=f'Choose a scheduler. Defaults to "{SD_SCHEDULER_DEFAULT}".',
    )
    seed: int = Field(
        description="Random seed. Leave blank to randomize the seed.", default=None
    )
    prompt_prefix: str = Field(description="Prompt prefix.", default=None)
    negative_prompt_prefix: str = Field(
        description="Negative prompt prefix.", default=None
    )
    output_image_extension: str = Field(
        description="Output type of the image. Can be 'png' or 'jpeg' or 'webp'.",
        default="jpeg",
    )
    output_image_quality: int = Field(
        description="Output quality of the image. Can be 1-100.", default=90
    )
    image_to_upscale: str = Field(
        description="Input image for the upscaler (AuraSR).", default=None
    )
    width: int = Field(
        description="Width of output image.",
        default=512,
    )
    height: int = Field(
        description="Height of output image.",
        default=512,
    )
    signed_urls: List[str] = Field(
        description="List of signed URLs for images to be uploaded to.", default=None
    )

    @validator("height")
    def validate_height(cls, v: int, values):
        return return_value_if_in_list(
            v,
            SIZE_LIST,
        )

    @validator("width")
    def validate_width(cls, v: int, values):
        return return_value_if_in_list(
            v,
            SIZE_LIST,
        )

    @validator("output_image_extension")
    def validate_output_image_extension(cls, v):
        return return_value_if_in_list(v, ["png", "jpeg", "webp"])


class UploadObject:
    def __init__(
        self,
        pil_image: Image.Image,
        signed_url: str,
        target_extension: str,
        target_quality: int,
    ):
        self.pil_image = pil_image
        self.signed_url = signed_url
        self.target_extension = target_extension
        self.target_quality = target_quality
