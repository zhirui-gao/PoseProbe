import os

class EnvironmentSettings:
    def __init__(self, data_root='', debug=False):
        self.workspace_dir = './logs'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = './logs'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir    # Directory for saving other models pre-trained networks
        self.eval_dir = ''    # Base directory for saving the evaluations.
        self.llff = ''
        self.dtu = '/home/gzr/data/sparf-dtu/dtu_dataset/rs_dtu_4/DTU'
        self.toy = '/home/gzr/data/toy_desk/processed'
        self.custom = '/media/gzr/ZX3 512G/transfer/mov'
        self.blender = '/home/zxn/workspace_gzr/pose_defrom'
        self.dtu_depth = ''
        self.dtu_mask = ''
        self.replica = ''
        self.log_dir = './logs'


