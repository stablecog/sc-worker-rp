import torch
import runpod
from src.shared.predict import create_predict_for_generate
from src.shared.schedulers import SD_SCHEDULER_CHOICES, SD_SCHEDULER_DEFAULT
from src.shared.sd import generate
from .pipe import (
    DEFAULT_NEGATIVE_PROMPT_PREFIX,
    DEFAULT_PROMPT_PREFIX,
    get_pipe_object,
    MODEL_NAME,
)

torch.cuda.empty_cache()

predict = create_predict_for_generate(
    model_name=MODEL_NAME,
    get_pipe_object=get_pipe_object,
    generate=generate,
    schedulers=SD_SCHEDULER_CHOICES,
    default_scheduler=SD_SCHEDULER_DEFAULT,
    default_prompt_prefix=DEFAULT_PROMPT_PREFIX,
    default_negative_prompt_prefix=DEFAULT_NEGATIVE_PROMPT_PREFIX,
)

runpod.serverless.start({"handler": predict})
