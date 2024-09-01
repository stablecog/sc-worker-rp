from dotenv import load_dotenv

load_dotenv()

from enum import Enum


DEVICE_CUDA = "cuda"
WORKER_VERSION = "v3.30"
SIZE_LIST = range(256, 1537, 8)


class TabulateLevels(Enum):
    PRIMARY = "simple_grid"
    SECONDARY = "simple"
