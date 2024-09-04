from src.shared.pipe_classes import AuraSrPipeObject
from aura_sr import AuraSR


MODEL_NAME = "AuraSR"
MODEL_ID = "fal/AuraSR-v2"


def get_pipe_object(to_cuda: bool = True) -> AuraSrPipeObject:
    pipe = AuraSR.from_pretrained(MODEL_ID)
    return AuraSrPipeObject(pipe=pipe)


if __name__ == "__main__":
    get_pipe_object(to_cuda=False)
