from src.shared.device import DEVICE_CPU, DEVICE_CUDA
from src.shared.pipe_classes import AuraSrPipeObject
from src.shared.aura_sr import AuraSR


MODEL_NAME = "AuraSR"
MODEL_ID = "fal/AuraSR-v2"


def get_pipe_object(to_cuda: bool = True) -> AuraSrPipeObject:
    device = DEVICE_CUDA if to_cuda else DEVICE_CPU
    pipe = AuraSR.from_pretrained(MODEL_ID, device=device)
    return AuraSrPipeObject(pipe=pipe)


if __name__ == "__main__":
    get_pipe_object(to_cuda=False)
