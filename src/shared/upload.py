from io import BytesIO
import logging
import time
from typing import List
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests
from urllib.parse import urlparse
from concurrent.futures import Future, ThreadPoolExecutor
from .classes import UploadObject
from .helpers import parse_content_type, time_log


class UploadedImageResult:
    def __init__(self, image_url: str):
        self.image_url = image_url


def extract_s3_url_from_signed_url(signed_url: str) -> str:
    """Helper function to extract the key from the signed URL."""
    parsed_url = urlparse(signed_url)
    path = parsed_url.path.lstrip("/")
    return f"s3://{path}"


def convert_and_upload_image_to_signed_url(upload_object: UploadObject) -> str:
    """Convert an individual image to a target format and upload to the provided signed URL."""

    with time_log(
        f"📨 Converted image to {upload_object.target_extension}",
        ms=True,
        start_log=False,
        prefix=False,
    ):
        _pil_image = upload_object.pil_image
        if upload_object.target_extension == "jpeg":
            _pil_image = _pil_image.convert("RGB")
        img_format = upload_object.target_extension.upper()
        img_bytes = BytesIO()
        _pil_image.save(
            img_bytes, format=img_format, quality=upload_object.target_quality
        )
        file_bytes = img_bytes.getvalue()

    # Define the retry strategy
    retry_strategy = Retry(
        total=3,  # Total number of retries
        status_forcelist=[429, 500, 502, 503, 504]
        + list(range(400, 429))
        + list(range(431, 500))
        + list(range(501, 600)),  # List of status codes to retry on
        allowed_methods=["PUT"],  # HTTP methods to retry
        backoff_factor=1,  # A backoff factor to apply between attempts
    )

    # Create an HTTPAdapter with the retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)

    # Create a session and mount the adapter
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    with time_log(
        f"📨 Uploaded image to S3",
        ms=True,
        start_log=False,
        prefix=False,
    ):
        response = session.put(
            upload_object.signed_url,
            data=file_bytes,
            headers={
                "Content-Type": parse_content_type(upload_object.target_extension)
            },
        )

    if response.status_code != 200:
        # throw an exception if the upload fails
        logging.info(f"^^ Failed to upload image. Status code: {response.status_code}")
        response.raise_for_status()

    s3_url = extract_s3_url_from_signed_url(upload_object.signed_url)
    return s3_url


def upload_images(
    upload_objects: List[UploadObject],
) -> List[UploadedImageResult]:
    """Upload all images to S3 in parallel and return the S3 URLs"""
    logging.info(
        f"^^ 📤 🟡 Started - Convert and upload {len(upload_objects)} image(s) to S3 in parallel"
    )
    start = time.time()

    logging.info(
        f"^^ Target extension: {upload_objects[0].target_extension} - Target quality: {upload_objects[0].target_quality}"
    )

    # Run all uploads at same time in threadpool
    tasks: List[Future] = []
    with ThreadPoolExecutor(max_workers=len(upload_objects)) as executor:
        logging.info(f"^^ Submitting to thread")
        for i, upload_object in enumerate(upload_objects):
            tasks.append(
                executor.submit(
                    convert_and_upload_image_to_signed_url,
                    upload_object,
                )
            )

    # Get results
    results: List[UploadedImageResult] = []
    for i, task in enumerate(tasks):
        results.append(
            UploadedImageResult(
                image_url=task.result(),
            )
        )

    end = time.time()
    logging.info(
        f"^^ 📤 🟢 {len(upload_objects)} image(s) converted and uploaded to S3 in: {round((end - start) *1000)}ms"
    )

    return results
