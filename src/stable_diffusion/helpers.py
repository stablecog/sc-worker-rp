from src.shared.schedulers import SD_SCHEDULERS


def get_scheduler(name, config):
    return SD_SCHEDULERS[name]["from_config"](config)
