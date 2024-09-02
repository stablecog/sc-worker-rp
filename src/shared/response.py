from typing import Any, List

from .upload import upload_images
from .classes import GenerateOutput, PredictionGenerateInput, UploadObject


def get_generate_response(
    outputs: List[GenerateOutput],
    validated_input: PredictionGenerateInput,
    job_input: Any,
):
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
