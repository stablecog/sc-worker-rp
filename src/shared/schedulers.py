from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
    DDPMScheduler,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)

SD_SCHEDULERS = {
    "K_LMS": {"from_config": LMSDiscreteScheduler.from_config},
    "PNDM": {"from_config": PNDMScheduler.from_config},
    "DDIM": {"from_config": DDIMScheduler.from_config},
    "K_EULER": {"from_config": EulerDiscreteScheduler.from_config},
    "K_EULER_ANCESTRAL": {"from_config": EulerAncestralDiscreteScheduler.from_config},
    "HEUN": {"from_config": HeunDiscreteScheduler.from_config},
    "DPM++_2M": {"from_config": DPMSolverMultistepScheduler.from_config},
    "DPM++_2S": {"from_config": DPMSolverSinglestepScheduler.from_config},
    "DEIS": {"from_config": DEISMultistepScheduler.from_config},
    "UNI_PC": {"from_config": UniPCMultistepScheduler.from_config},
    "DDPM": {"from_config": DDPMScheduler.from_config},
}

SD_SCHEDULER_CHOICES = [*SD_SCHEDULERS.keys()]
SD_SCHEDULER_DEFAULT = SD_SCHEDULER_CHOICES[0]

KANDINSKY_22_SCHEDULERS = {
    "DDPM": {
        "scheduler": DDPMScheduler,
        "from_config": True,
    },
    "DDIM": {"scheduler": DDIMScheduler},
    "DPM++_2M": {"scheduler": DPMSolverMultistepScheduler},
}
