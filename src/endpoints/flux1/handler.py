import torch
import runpod
from src.shared.predict import create_predict_for_generate
from src.shared.schedulers import (
    SD_SCHEDULER_CHOICES,
    SD_SCHEDULER_DEFAULT,
)
from .generate import generate
from .pipe import MODEL_NAME, get_pipe_object

torch.cuda.empty_cache()

predict = create_predict_for_generate(
    model_name=MODEL_NAME,
    get_pipe_object=get_pipe_object,
    generate=generate,
    schedulers=SD_SCHEDULER_CHOICES,
    default_scheduler=SD_SCHEDULER_DEFAULT,
    dont_set_scheduler=True,
)

runpod.serverless.start({"handler": predict})
