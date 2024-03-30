import importlib


def get_config(stage: str, exp_name: str, model: str):
    engine_config = importlib.import_module('configs.' + stage)
    return engine_config.EngineConfig(exp_name, model)
