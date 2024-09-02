import torch
import runpod
from src.shared.predict import create_predict_func
from src.shared.schedulers import KANDINSKY_22_SCHEDULER_CHOICES
from .constants import MODEL_NAME
from .generate import generate
from .pipe_object import get_pipe_object

torch.cuda.empty_cache()

predict = create_predict_func(
    model_name=MODEL_NAME,
    get_pipe_object=get_pipe_object,
    generate=generate,
    schedulers=KANDINSKY_22_SCHEDULER_CHOICES,
)

runpod.serverless.start({"handler": predict})
