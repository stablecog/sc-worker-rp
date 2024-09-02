import torch
from pydantic import ValidationError
import runpod

from src.shared.response import get_generate_response
from .generate import generate
from .pipe_object import get_pipe_object
from src.shared.classes import (
    Kandinsky22PipeObject,
    PredictionGenerateInput,
    predict_input_to_generate_input,
)

torch.cuda.empty_cache()


class Model:
    def __init__(self):
        self.pipe_object: Kandinsky22PipeObject = get_pipe_object()


MODEL = Model()


@torch.inference_mode()
def predict(job):
    job_input = job["input"]
    validated_input: PredictionGenerateInput | None = None

    try:
        validated_input = PredictionGenerateInput(**job_input)
    except ValidationError as e:
        return {"error": {"code": "validation_failed", "message": e.json()}}

    generate_input = predict_input_to_generate_input(validated_input)
    outputs = generate(
        input=generate_input,
        pipe_object=MODEL.pipe_object,
    )

    response = get_generate_response(
        outputs=outputs,
        validated_input=validated_input,
        job_input=job_input,
    )
    return response


runpod.serverless.start({"handler": predict})
