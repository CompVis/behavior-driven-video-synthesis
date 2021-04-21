from experiments.vunet import Vunet
from experiments.behavior_net import BehaviorNet
from experiments.shape_and_pose_net import (
    ShapePoseNet,
)
from experiments.mt_vae import MTVAEModel

__experiments__ = {
    "vunet": Vunet,
    "behavior_net": BehaviorNet,
    "cvbae": ShapePoseNet,
    "mtvae": MTVAEModel
}


def select_experiment(config,dirs, device):
    experiment = config["general"]["experiment"]
    project_name = config["general"]["project_name"]
    if experiment not in __experiments__:
        raise NotImplementedError("No such experiment!")
    if config["general"]["restart"]:
        print(f"Restarting experiment \"{project_name}\" of type \"{experiment}\". Device: {device}")
    else:
        print(f"Running new experiment \"{project_name}\" of type \"{experiment}\". Device: {device}")
    return __experiments__[experiment](config, dirs, device)
