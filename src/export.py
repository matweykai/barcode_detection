import click
from ultralytics import YOLO


@click.command()
@click.argument('model_path')
def main(model_path: str):
    model = YOLO(model_path)

    model.export(
        format='onnx',
        dynamic=False,
    )


if __name__ == '__main__':
    main()
