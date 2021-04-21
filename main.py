import argparse
from os import path, makedirs
import os
from experiments import select_experiment
import torch
import yaml
from glob import glob
from shutil import copy

def create_dir_structure(config):
    subdirs = ["ckpt", "config", "generated", "log"]
    structure = {subdir: path.join(config["base_dir"],config["experiment"],subdir,config["project_name"]) for subdir in subdirs}
    return structure

def load_parameters(config_name, restart,run_inference, debug, pretrained_model):
    with open(config_name,"r") as f:
        cdict = yaml.load(f,Loader=yaml.FullLoader)

    if 'DATAPATH' in os.environ:
        cdict['general']['base_dir'] = path.join(os.environ['DATAPATH'],cdict['general']['base_dir'][1:])
        cdict['data']['datapath'] = path.join(os.environ['DATAPATH'], cdict['data']['datapath'][1:])

    dir_structure = create_dir_structure(cdict["general"])
    saved_config = path.join(dir_structure["config"], "config.yaml")
    if restart:
        if path.isfile(saved_config):
            with open(saved_config,"r") as f:
                cdict = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError("No saved config file found but model is intended to be restarted. Aborting....")

    elif pretrained_model:
        pretrained_config = path.join(pretrained_model,'config.yaml')
        if path.isfile(pretrained_config):
            with open(pretrained_config,"r") as f:
                cdict = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError("No saved config file found but model is intended to be restarted. Aborting....")

        dir_structure = create_dir_structure(cdict["general"])
        [makedirs(dir_structure[d],exist_ok=True) for d in dir_structure]
        dump_path = path.join(dir_structure["config"], "config.yaml")
        with open(dump_path, "w") as f:
            yaml.dump(cdict, f, default_flow_style=False)

        model_ckpt = glob(path.join(pretrained_model, '*.pth'))
        [copy(c,dir_structure['ckpt']) for c in model_ckpt]

    else:
        [makedirs(dir_structure[d],exist_ok=True) for d in dir_structure]
        if path.isfile(saved_config) and not debug and not run_inference:
            print(f"\033[93m" + "WARNING: Model has been started somewhen earlier: Resume training (y/n)?" + "\033[0m")
            while True:
                answer = input()
                if answer == "y" or answer == "yes":
                    with open(saved_config,"r") as f:
                        cdict = yaml.load(f, Loader=yaml.FullLoader)

                    restart = True
                    break
                elif answer == "n" or answer == "no":
                    with open(saved_config, "w") as f:
                        yaml.dump(cdict, f, default_flow_style=False)
                    break
                else:
                    print(f"\033[93m" + "Invalid answer! Try again!(y/n)" + "\033[0m")
        else:
            print(f'Load config saved at "{saved_config}"')
            with open(saved_config, "w") as f:
                yaml.dump(cdict,f,default_flow_style=False)

    return cdict, dir_structure, restart



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        default="config/behavior_net.yaml",
                        help="Define config file")
    parser.add_argument("-r","--restart", action='store_true', default=False,help="Whether training should be resumed.")
    parser.add_argument("-f", "--flow", action='store_true', default=False, help="Whether to train only the flow model (this requires a pretrained cVAE model).")
    parser.add_argument("--gpu",default=[0], type=int,
                        nargs="+",help="GPU to use.")
    parser.add_argument("-m","--mode",default="train",type=str,choices=["train","infer"],help="Whether to start in train or infer mode?")
    parser.add_argument("-d", "--debug", action='store_true', default=False,
                        help="Whether model should be started in debug mode.")
    parser.add_argument("-v", "--visualization", action='store_true', default=False,
                        help="Whether generate visual results with the model (otherwise, the model will be evaluated quantitatively.")
    parser.add_argument("-s", "--synth_model", type=str,
                        default="/export/data/ablattma/neural_pose_behavior/saved_runs/image_model/CVBAE-H36m-bs13-s256-nkl-nps-subpix-norm_l1-3dpos-imax1000-2020-05-15_23:37:44/config.yaml",
                        help="Config file for the pretrained image synthesis model, when intending to apply our full pipeline to synthesize videos.")
    parser.add_argument("-p", "--pretrained_model", type=str,
                        default=None,
                        help="The path to the hyperparameter file and checkpoint for a pretrained model for evaluation. If this is not None, the script will search for pretrained models under under the path.")
    args = parser.parse_args()

    if args.pretrained_model == args.restart:
        raise ValueError(f'Pretrained model for evaluation should not be restarted as this would have no effect. Select only one of both options....')

    infer = args.mode == "infer"
    config, structure, restart = load_parameters(args.config, args.restart or args.flow,infer, args.debug, args.pretrained_model)
    config["general"]["restart"] = restart
    if args.pretrained_model:
        config['general']['mode'] = 'infer'
    else:
        config["general"]["mode"] = args.mode

    config['general']['debug'] = args.debug
    config['training']['only_flow']=args.flow
    config['logging']['visualization'] = args.visualization
    config['logging']['synth_params'] = args.synth_model

    if len(args.gpu) == 1:
        gpus = torch.device(
            f"cuda:{int(args.gpu[0])}"
            if torch.cuda.is_available() and int(args.gpu[0]) >= 0
            else "cpu"
        )
    else:
        gpus = [int(id) for id in args.gpu]

    experiment = select_experiment(config,structure, gpus)

    # start selected experiment
    mode = config["general"]["mode"]
    if  mode == "train":
        experiment.run_training()
    elif mode == "infer":
        experiment.run_inference()
    else:
        raise ValueError(f"\"mode\"-parameter should be either \"train\" or \"infer\" but is actually {mode}")



