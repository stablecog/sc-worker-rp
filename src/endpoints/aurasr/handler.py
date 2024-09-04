import torch
import runpod
from src.shared.predict import create_predict_for_upscale
from .upscale import upscale
from .pipe import MODEL_NAME, get_pipe_object

torch.cuda.empty_cache()

predict = create_predict_for_upscale(
    model_name=MODEL_NAME,
    get_pipe_object=get_pipe_object,
    upscale=upscale,
)

runpod.serverless.start({"handler": predict})
