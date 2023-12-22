import click
from ultralytics import YOLO

from src.config import Config


@click.command()
@click.argument('config_path')
def main(config_path: str):
    config = Config.from_yaml(config_path)

    yolo_config = config.yolo_config

    model = YOLO(
        model=yolo_config.path,
        task='detect',
    )

    model.train(
        project=config.project_name,
        name=config.exp_name,
        data=yolo_config.dataset_path,
        epochs=yolo_config.epochs_num,
        imgsz=yolo_config.img_size,
        batch=yolo_config.batch_size,
        cache=yolo_config.cache,
        device=yolo_config.device,
        workers=yolo_config.workers,
        optimizer=yolo_config.optimizer,
        close_mosaic=yolo_config.close_mosaic,
        amp=yolo_config.amp,
        lr0=yolo_config.lr0,
        lrf=yolo_config.lrf,
        momentum=yolo_config.momentum,
        weight_decay=yolo_config.weight_decay,
        warmup_epochs=yolo_config.warmup_epochs,
        warmup_momentum=yolo_config.warmup_momentum,
        warmup_bias_lr=yolo_config.warmup_bias_lr,
        save_dir='runs',
    )

    model.val(
        data=yolo_config.dataset_path
    )


if __name__ == '__main__':
    main()
