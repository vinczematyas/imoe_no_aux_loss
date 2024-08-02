from safetensors.torch import save_model, load_model
import numpy as np
import yaml
import os


class NestedNamespace:
    """Namespace that allows nested dictionaries to be accessed as attributes"""

    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return NestedNamespace(**entry)
        return entry

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, NestedNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))
            else: # this is the only addition
                setattr(self, key, val)

    def to_dict(self):
        result = {}
        for key, val in self.__dict__.items():
            if isinstance(val, NestedNamespace):
                result[key] = val.to_dict()
            elif isinstance(val, list):
                result[key] = [v.to_dict() if isinstance(v, NestedNamespace) else v for v in val]
            else:
                result[key] = val
        return result

    def to_yaml(self, file_path):
        with open(file_path, 'w') as file:
            yaml.dump(self.to_dict(), file, default_flow_style=False)

    def update(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key) and val is not None:
                setattr(self, key, val)


def init_cfg(path):
    """Initialize the configuration from a yaml file"""
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return NestedNamespace(**cfg)


def save_agent(agent, path, save_obs=False):
    """Save the agent and observations from the replay buffer"""
    os.makedirs(path, exist_ok=True)
    save_model(agent.actor, f"{path}/actor.safetensors")
    save_model(agent.qf1, f"{path}/qf1.safetensors")
    save_model(agent.qf2, f"{path}/qf2.safetensors")
    save_model(agent.qf1_target, f"{path}/qf1_target.safetensors")
    save_model(agent.qf2_target, f"{path}/qf2_target.safetensors")
    if save_obs == True:
        np.savez_compressed(f"{path}/observations.npz", array=agent.rb.observations)


def load_agent(agent, path):
    """Load the agent and observations from the replay buffer"""

    assert os.path.exists(path), f"Path {path} does not exist"
    load_model(agent.actor, f"{path}/actor.safetensors")
    load_model(agent.qf1, f"{path}/qf1.safetensors")
    load_model(agent.qf2, f"{path}/qf2.safetensors")
    load_model(agent.qf1_target, f"{path}/qf1_target.safetensors")
    load_model(agent.qf2_target, f"{path}/qf2_target.safetensors")
    if os.path.exists(f"{path}/observations.npz"):
        return agent, np.load(f"{path}/observations.npz")["array"]
    else:
        return agent, None
