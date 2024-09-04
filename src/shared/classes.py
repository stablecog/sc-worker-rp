from typing import Any, Dict, Generic, List, Optional, TypeVar
from urllib.parse import urlparse
from pydantic import BaseModel, Field, validator

from PIL import Image
from .constants import SIZE_LIST

T = TypeVar("T")


def return_value_if_in_list(value: T, list_of_values: List[T]) -> T:
    if value not in list_of_values:
        raise ValueError(f'"{value}" is not in the list of choices')
    return value


T = TypeVar("T")


class GenerateInput:
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


class GenerateOutput:
    def __init__(self, image: Image.Image):
        self.image = image


class PredictionGenerateInput(BaseModel):
    prompt: str = Field(description="Input prompt.", default="")
    negative_prompt: str = Field(description="Input negative prompt.", default="")
    num_outputs: int = Field(
        description="Number of images to output. If the NSFW filter is triggered, you may get fewer outputs than this.",
        ge=1,
        le=4,
        default=1,
    )
    init_image_url: Optional[str] = Field(
        description="Init image url to be used with img2img.",
        default=None,
    )
    mask_image_url: Optional[str] = Field(
        description="Mask image url to be used with img2img.",
        default=None,
    )
    prompt_strength: Optional[float] = Field(
        description="The strength of the prompt when using img2img, between 0-1. When 1, it'll essentially ignore the image.",
        ge=0,
        le=1,
        default=None,
    )
    num_inference_steps: int = Field(
        description="Number of denoising steps", ge=1, le=500, default=25
    )
    guidance_scale: float = Field(
        description="Scale for classifier-free guidance.", ge=1, le=20, default=7
    )
    schedulers: List[str] = Field(
        description="List of available schedulers.",
    )
    default_scheduler: str = Field(
        description="Default scheduler.",
    )
    scheduler: str = Field(
        description=f"Choose a scheduler.",
        default=default_scheduler,
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
        description="Output quality of the image. Can be 1-100.",
        default=85,
        ge=1,
        le=100,
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
        description="List of signed URLs for images to be uploaded to."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        description="Optional metadata to be returned with the response.", default=None
    )

    @validator("height")
    def validate_height(cls, v: int, values):
        if v not in SIZE_LIST:
            raise ValueError(f'Invalid height: "{v}". Must be one of {SIZE_LIST}.')
        return v

    @validator("width")
    def validate_width(cls, v: int, values):
        if v not in SIZE_LIST:
            raise ValueError(f'Invalid width: "{v}". Must be one of {SIZE_LIST}.')
        return v

    @validator("output_image_extension")
    def validate_output_image_extension(cls, v):
        if v not in ["png", "jpeg", "webp"]:
            raise ValueError(
                f'Invalid output_image_extension: "{v}". Must be one of ["png", "jpeg", "webp"].'
            )
        return v

    @validator("prompt_strength", pre=True, always=True)
    def validate_prompt_strength(cls, v, values):
        if values.get("init_image_url") is not None and v is None:
            raise ValueError(
                "prompt_strength must be provided when init_image_url is set."
            )
        return v

    @validator("mask_image_url", pre=True, always=True)
    def validate_mask_image_url(cls, v, values):
        if v is not None and values.get("init_image_url") is None:
            raise ValueError(
                "init_image_url must be provided when mask_image_url is set."
            )
        return v

    @validator("signed_urls")
    def validate_signed_urls(cls, v, values):
        if not isinstance(v, list) or len(v) < values.get("num_outputs", 1):
            raise ValueError(
                "signed_urls must be a list with at least as many elements as num_outputs."
            )
        return v

    @validator("scheduler")
    def validate_scheduler(cls, v, values):
        if v not in values.get("schedulers", []):
            raise ValueError(
                f"Invalid scheduler: {v}. Must be one of {values['schedulers']}."
            )
        return v

    @validator("default_scheduler")
    def validate_default_scheduler(cls, v, values):
        if v not in values.get("schedulers", []):
            raise ValueError(
                f"Invalid default_scheduler: {v}. Must be one of {values['schedulers']}."
            )
        return v


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


def predict_input_to_generate_input(
    input: PredictionGenerateInput,
) -> GenerateInput:
    return GenerateInput(
        prompt=input.prompt,
        negative_prompt=input.negative_prompt,
        prompt_prefix=input.prompt_prefix,
        negative_prompt_prefix=input.negative_prompt_prefix,
        width=input.width,
        height=input.height,
        num_outputs=input.num_outputs,
        num_inference_steps=input.num_inference_steps,
        guidance_scale=input.guidance_scale,
        init_image_url=input.init_image_url,
        mask_image_url=input.mask_image_url,
        prompt_strength=input.prompt_strength,
        scheduler=input.scheduler,
        seed=input.seed,
    )


P = TypeVar("P")


class GenerateFunctionProps(Generic[P]):
    def __init__(
        self,
        input: GenerateInput,
        pipe_object: P,
        model_name: str,
        default_prompt_prefix: str | None = None,
        default_negative_prompt_prefix: str | None = None,
        dont_set_scheduler: bool = False,
    ):
        self.input = input
        self.pipe_object = pipe_object
        self.model_name = model_name
        self.default_prompt_prefix = default_prompt_prefix
        self.default_negative_prompt_prefix = default_negative_prompt_prefix
        self.dont_set_scheduler = dont_set_scheduler


class UpscaleInput:
    def __init__(
        self,
        images: List[str],
        scale: int,
    ):
        self.images = images
        self.scale = scale


U = TypeVar("U")


class UpscaleFunctionProps(Generic[U]):
    def __init__(
        self,
        input: UpscaleInput,
        pipe_object: U,
        model_name: str,
    ):
        self.input = input
        self.pipe_object = pipe_object
        self.model_name = model_name


class PredictionUpscaleInput(BaseModel):
    images: List[str] = Field(
        description="Image URLs to be upscaled.",
    )
    scale: int = Field(
        description="Upscale factor.",
        ge=1,
        le=4,
        default=4,
    )
    output_image_extension: str = Field(
        description="Output type of the image. Can be 'png' or 'jpeg' or 'webp'.",
        default="jpeg",
    )
    output_image_quality: int = Field(
        description="Output quality of the image. Can be 1-100.",
        default=85,
        ge=1,
        le=100,
    )
    signed_urls: List[str] = Field(
        description="List of signed URLs for images to be uploaded to."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        description="Optional metadata to be returned with the response.", default=None
    )

    @validator("output_image_extension")
    def validate_output_image_extension(cls, v):
        if v not in ["png", "jpeg", "webp"]:
            raise ValueError(
                f'Invalid output_image_extension: "{v}". Must be one of ["png", "jpeg", "webp"].'
            )
        return v

    @validator("images")
    def validate_image_urls(cls, v):
        for url in v:
            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValueError(f"Invalid URL: {url}")
            if parsed_url.scheme not in ["http", "https"]:
                raise ValueError(f"URL must use http or https scheme: {url}")
        return v

    @validator("signed_urls")
    def validate_signed_urls(cls, v, values):
        images = values.get("images", [])
        if not isinstance(v, list) or len(v) < len(images):
            raise ValueError(
                "signed_urls must be a list with at least as many elements as images."
            )
        return v


class UpscaleOutput:
    def __init__(self, image: Image.Image):
        self.image = image


def predict_input_to_upscale_input(
    input: PredictionUpscaleInput,
) -> UpscaleInput:
    return UpscaleInput(
        images=input.images,
        scale=input.scale,
    )
