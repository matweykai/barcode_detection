from typing import List, Union

from datetime import datetime
from omegaconf import OmegaConf
from pydantic import BaseModel


class YoloConfig(BaseModel):
    """Pydantic model class for storing YOLO model parameters"""
    path: str
    dataset_path: str
    epochs_num: int

    img_size: int
    batch_size: int
    cache: bool
    device: str | int
    workers: int

    optimizer: str
    close_mosaic: int
    amp: bool
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float
    warmup_epochs: int
    warmup_momentum: float
    warmup_bias_lr: float


class Config(BaseModel):
    """Class for storing training parameters"""
    project_name: str
    exp_name: str

    yolo_config: YoloConfig

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Loads config from yaml file

        Args:
            path (str): path to yaml config file

        Returns:
            Config: config object that stores all training settings
        """
        cfg: dict = OmegaConf.to_container(OmegaConf.load(path), resolve=True)

        if 'experiment_name' not in cfg:            
            cfg['experiment_name'] = f'exp_{datetime.strftime(datetime.now(), r"%y_%m_%d__%H_%M")}'

        return cls(**cfg)
