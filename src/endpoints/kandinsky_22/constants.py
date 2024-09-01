from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)


KANDINSKY_22_DECODER_MODEL_ID = "kandinsky-community/kandinsky-2-2-decoder"
KANDINSKY_22_PRIOR_MODEL_ID = "kandinsky-community/kandinsky-2-2-prior"
KANDINSKY_22_DECODER_INPAINT_MODEL_ID = (
    "kandinsky-community/kandinsky-2-2-decoder-inpaint"
)
KANDINSKY_22_MODEL_NAME = "Kandinsky 2.2"

KANDINSKY_22_SCHEDULERS = {
    "DDPM": {
        "scheduler": DDPMScheduler,
        "from_config": True,
    },
    "DDIM": {"scheduler": DDIMScheduler},
    "DPM++_2M": {"scheduler": DPMSolverMultistepScheduler},
}
