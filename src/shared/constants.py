from dotenv import load_dotenv

load_dotenv()

from enum import Enum


DEVICE_CUDA = "cuda"
WORKER_VERSION = "v1.04"
SIZE_LIST = list(range(256, 1537, 8))


class TabulateLevels(Enum):
    PRIMARY = "simple_grid"
    SECONDARY = "simple"
