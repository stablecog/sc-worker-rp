import concurrent.futures
from typing import List

import torch
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
    GenerateFunctionInputKandinsky22,
    Kandinsky22PipeObject,
    PredictionGenerateInput,
    UploadObject,
)
from src.shared.constants import DEVICE_CUDA
from src.shared.upload import upload_images

torch.cuda.empty_cache()


class Handler:
    def __init__(self):
        self.pipe_object: Kandinsky22PipeObject
        self.load_pipes()

    def load_prior(self):
        prior = KandinskyV22PriorPipeline.from_pretrained(
            KANDINSKY_22_PRIOR_MODEL_ID,
            torch_dtype=torch.float16,
        ).to(DEVICE_CUDA)
        return prior

    def load_text2img(self):
        text2img = KandinskyV22Pipeline.from_pretrained(
            KANDINSKY_22_DECODER_MODEL_ID,
            torch_dtype=torch.float16,
        ).to(DEVICE_CUDA)
        return text2img

    def load_inpaint(self):
        return None

    def load_pipes(self):
        self.pipe_object = Kandinsky22PipeObject(
            prior=None,
            text2img=None,
            inpaint=None,
        )
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_prior = executor.submit(self.load_prior)
            future_text2img = executor.submit(self.load_text2img)
            future_inpaint = executor.submit(self.load_inpaint)

            self.pipe_object.prior = future_prior.result()
            self.pipe_object.text2img = future_text2img.result()
            self.pipe_object.inpaint = future_inpaint.result()


handler = Handler()


@torch.inference_mode()
def predict(job):
    job_input = job["input"]
    validated_input: PredictionGenerateInput = None

    try:
        validated_input = PredictionGenerateInput(**job_input)
    except ValidationError as e:
        return {"error": {"code": "validation_failed", "message": e.json()}}

    generate_input = GenerateFunctionInputKandinsky22(
        **validated_input.dict(), pipe_object=handler.pipe_object
    )
    outputs = generate(generate_input)
    upload_objects: List[UploadObject] = {}
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
        upload_path_prefix=validated_input["upload_prefix"],
    )

    response = {"output": {"images": []}}
    for upload_result in upload_results:
        response["output"]["images"].append(upload_result.image_url)

    return response


runpod.serverless.start({"handler": predict})
