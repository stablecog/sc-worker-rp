from dotenv import load_dotenv

load_dotenv()

from enum import Enum


WORKER_VERSION = "v1.11"
SIZE_LIST = list(range(256, 1537, 8))


class TabulateLevels(Enum):
    PRIMARY = "simple_grid"
    SECONDARY = "simple"
