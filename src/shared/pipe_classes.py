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
    StableDiffusion3Img2ImgPipeline,
)
from aura_sr import AuraSR


class StableDiffusionPipeObject:
    def __init__(
        self,
        text2img: (
            StableDiffusionPipeline
            | StableDiffusionXLPipeline
            | StableDiffusion3Pipeline
        ),
        img2img: (
            StableDiffusionImg2ImgPipeline
            | StableDiffusionXLImg2ImgPipeline
            | StableDiffusion3Img2ImgPipeline
            | None
        ),
        inpaint: StableDiffusionInpaintPipeline | None = None,
        refiner: StableDiffusionXLImg2ImgPipeline | None = None,
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
        inpaint: KandinskyV22InpaintPipeline | None = None,
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


class AuraSrPipeObject:
    def __init__(
        self,
        pipe: AuraSR,
    ):
        self.pipe = pipe
