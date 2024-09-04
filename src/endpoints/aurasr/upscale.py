import time
from PIL import Image
from typing import List
from urllib.parse import urlparse
import requests
from io import BytesIO
import logging


from src.shared.classes import UpscaleFunctionProps, UpscaleOutput
from src.shared.pipe_classes import AuraSrPipeObject


def upscale(props: UpscaleFunctionProps[AuraSrPipeObject]) -> List[UpscaleOutput]:
    # Props
    input = props.input
    pipe_object = props.pipe_object
    model_name = props.model_name
    # ---------------------------------------------------------------------

    output_images: List[Image.Image] = []

    for image_url in input.images:
        logging.info("🟡 Upscale | Started")
        s = time.time()
        logging.info("🟡 Upscale | Image is a URL, downloading...")
        image = load_image_from_url(image_url)
        e = time.time()
        logging.info(f"🟢 Upscale | Image downloaded | {round((e - s) * 1000)}ms")

        inf_start_time = time.time()
        upscaled_image = pipe_object.pipe.upscale_4x_overlapped(image)
        inf_end_time = time.time()
        logging.info(
            f"🔮 🟢 Upscale |  {model_name} | Scale: {input.scale} | {round((inf_end_time - inf_start_time) * 1000)}ms"
        )
        output_images.append(upscaled_image)

    outputs: List[UpscaleOutput] = []
    for image in output_images:
        outputs.append(UpscaleOutput(image=image))

    return outputs


def load_image_from_url(url, timeout=10, max_size=8 * 1024 * 1024):
    # Validate URL
    if not url or not urlparse(url).scheme:
        raise ValueError("Invalid URL")

    try:
        # Send request with timeout
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError(
                f"URL does not point to an image (Content-Type: {content_type})"
            )

        # Check file size
        if len(response.content) > max_size:
            raise ValueError(
                f"Image size ({len(response.content)} bytes) exceeds maximum allowed size ({max_size} bytes)"
            )

        # Load image
        image_data = BytesIO(response.content)
        image = Image.open(image_data)

        return image

    except requests.RequestException as e:
        logging.error(f"Error fetching image from URL: {e}")
        raise
    except (IOError, ValueError) as e:
        logging.error(f"Error processing image: {e}")
        raise
