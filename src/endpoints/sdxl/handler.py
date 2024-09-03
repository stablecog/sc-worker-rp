import torch
import runpod
from src.shared.predict import create_predict_for_generate
from src.shared.schedulers import SD_SCHEDULER_CHOICES, SD_SCHEDULER_DEFAULT
from .constants import (
    MODEL_NAME,
)
from .pipe_object import get_pipe_object
from src.shared.stable_diffusion import generate

torch.cuda.empty_cache()


predict = create_predict_for_generate(
    model_name=MODEL_NAME,
    get_pipe_object=get_pipe_object,
    generate=generate,
    schedulers=SD_SCHEDULER_CHOICES,
    default_scheduler=SD_SCHEDULER_DEFAULT,
)

runpod.serverless.start({"handler": predict})
