import torch
from typing import List
from pydantic import ValidationError
import runpod
from src.endpoints.ssd_1b.constants import (
    MODEL_NAME,
)
from src.endpoints.ssd_1b.pipe_object import get_pipe_object
from src.stable_diffusion.generate import generate
from src.shared.classes import (
    PredictionGenerateInput,
    StableDiffusionPipeObject,
    UploadObject,
    prediction_input_to_generate_input,
)
from src.shared.upload import upload_images

torch.cuda.empty_cache()


class Model:
    def __init__(self):
        self.pipe_object: StableDiffusionPipeObject = get_pipe_object()


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
        model_name=MODEL_NAME,
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
