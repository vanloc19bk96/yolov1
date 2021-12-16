import argparse
import collections
import os
from pathlib import Path

from utils import read_cfg


class ConfigParser:
    def __init__(self, config):
        self._config = config

    @property
    def config(self):
        return self._config

    def __getitem__(self, name):
        return self._config[name]

    @classmethod
    def from_args(cls, args):
        if not isinstance(args, tuple):
            args = args.parse_args()
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        msg_no_cfg = "Configuration file need to be specified. Add '-c config.json'."
        assert args.config is not None, msg_no_cfg
        cfg_fname = Path(args.config)

        config = read_cfg(cfg_fname)

        return cls(config)

    def init_layers(self, module, *args, **kwargs):
        layer_names = self.config.sections()
        layers = []
        for name in layer_names:
            layer_args = {k: int(v) for k, v in self.config.items(name)}

            assert all([k not in layer_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
            layer_args.update(kwargs)
            layer_name_position = 0
            name = name.split('-')[layer_name_position]
            layer = getattr(module, name)(*args, **layer_args)
            layers.append(layer)

        return layers
