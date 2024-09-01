import torch
from typing import List
from diffusers import (
    KandinskyV22Pipeline,
    KandinskyV22PriorPipeline,
)
from pydantic import ValidationError
import runpod
from src.endpoints.kandinsky_22.constants import (
    KANDINSKY_22_DECODER_MODEL_ID,
    KANDINSKY_22_PRIOR_MODEL_ID,
)
from src.endpoints.kandinsky_22.generate import generate
from src.shared.classes import (
    Kandinsky22PipeObject,
    PredictionGenerateInput,
    UploadObject,
)
from src.shared.constants import DEVICE_CUDA
from src.shared.helpers import prediction_input_to_generate_input
from src.shared.upload import upload_images

torch.cuda.empty_cache()


class Model:
    def __init__(self):
        self.pipe_object: Kandinsky22PipeObject = Kandinsky22PipeObject(
            prior=KandinskyV22PriorPipeline.from_pretrained(
                KANDINSKY_22_PRIOR_MODEL_ID,
                torch_dtype=torch.float16,
            ).to(DEVICE_CUDA),
            text2img=KandinskyV22Pipeline.from_pretrained(
                KANDINSKY_22_DECODER_MODEL_ID,
                torch_dtype=torch.float16,
            ).to(DEVICE_CUDA),
            inpaint=None,
        )


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
