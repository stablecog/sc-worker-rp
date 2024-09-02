import json
from typing import Callable, List, TypeVar
from pydantic import ValidationError
import torch

from src.shared.constants import WORKER_VERSION
from .upload import upload_images
from .classes import (
    GenerateFunctionProps,
    GenerateOutput,
    PredictionGenerateInput,
    UploadObject,
    predict_input_to_generate_input,
    GenerateFunctionProps,
)

T = TypeVar("T")


def create_predict_func(
    model_name: str,
    get_pipe_object: Callable[[bool], T],
    generate: Callable[[GenerateFunctionProps[T]], List[GenerateOutput]],
    schedulers: List[str],
    default_prompt_prefix: str | None = None,
    default_negative_prompt_prefix: str | None = None,
    dont_set_scheduler: bool = False,
):
    class Model:
        def __init__(self):
            self.pipe_object: T = get_pipe_object(True)

    MODEL = Model()

    def predict(job):
        job_input = job["input"]
        validated_input: PredictionGenerateInput | None = None

        try:
            validated_input = PredictionGenerateInput(
                **job_input, schedulers=schedulers
            )
        except ValidationError as e:
            print(f"Validation error: {e}")
            return {
                "error": {
                    "code": "validation_error",
                    "message": e.json(),
                },
                "input": job_input,
                "metadata": {
                    "worker_version": WORKER_VERSION,
                },
            }

        generate_input = predict_input_to_generate_input(validated_input)
        outputs = generate(
            GenerateFunctionProps(
                input=generate_input,
                pipe_object=MODEL.pipe_object,
                model_name=model_name,
                default_prompt_prefix=default_prompt_prefix,
                default_negative_prompt_prefix=default_negative_prompt_prefix,
                dont_set_scheduler=dont_set_scheduler,
            )
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

        response = {
            "output": {
                "images": [],
            },
            "input": job_input,
            "metadata": {
                "worker_version": WORKER_VERSION,
            },
        }
        for upload_result in upload_results:
            response["output"]["images"].append(upload_result.image_url)
        return response

    return predict
