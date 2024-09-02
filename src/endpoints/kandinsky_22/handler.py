import torch
from typing import List
from pydantic import ValidationError
import runpod
from .generate import generate
from .pipe_object import get_pipe_object
from src.shared.classes import (
    Kandinsky22PipeObject,
    PredictionGenerateInput,
    UploadObject,
    prediction_input_to_generate_input,
)
from src.shared.upload import upload_images

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

    generate_input = prediction_input_to_generate_input(validated_input)
    outputs = generate(
        input=generate_input,
        pipe_object=MODEL.pipe_object,
    )

    upload_objects: List[UploadObject] = []
    for i, output in enumerate(outputs):
        upload_objects.append(
            UploadObject(
                pil_image=output.image,
                signed_url=validated_input.signed_urls[i],
                target_extension=validated_input.output_image_extension,
                target_quality=validated_input.output_image_quality,
            )
        )
    upload_results = upload_images(
        upload_objects=upload_objects,
    )

    response = {"output": {"images": []}, "input": job_input}
    for upload_result in upload_results:
        response["output"]["images"].append(upload_result.image_url)

    return response


runpod.serverless.start({"handler": predict})
