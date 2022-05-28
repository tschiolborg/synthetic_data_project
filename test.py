import os

import hydra
import torch
from torchmetrics.detection.map import MeanAveragePrecision

from src.config import ConfigTest
from src.engine import evalute_both_stages
from src.utils import Json_writer, collate_fn, load_data_test


def test_full_model():
    """Test full model with both detection and classification on test data"""

    with hydra.initialize(config_path="conf"):
        cfg: ConfigTest = hydra.compose(config_name="config_test.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # load data
    dataset_test = load_data_test(cfg)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.dataset.test.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collate_fn,
    )

    # load models
    model = torch.load(cfg.model_dir)
    classifier = torch.load(cfg.classifier_dir)

    metric_dec = MeanAveragePrecision()
    metric_cls = MeanAveragePrecision(iou_thresholds=[0.5, 0.75], class_metrics=True)

    model = model.to(device)

    # evaluate full model
    scores_dec, scores_cls = evalute_both_stages(
        model=model,
        classifier=classifier,
        data_loader=data_loader_test,
        device=device,
        metric_dec=metric_dec,
        metric_cls=metric_cls,
        cls_input_size=224,
        write=False,
    )

    print("Detection scores: ")
    print(scores_dec)

    print("Detection + classification scores: ")
    print(scores_cls)

    # log
    # os.makedirs(cfg.save_dir, exist_ok=True)
    # writer = Json_writer(os.path.join(cfg.save_dir, "result.json"))
    # writer.add_scalars("Test/detection/scores", scores_dec)
    # writer.add_scalars("Test/scores", scores_cls)
    # writer.flush()

    return


if __name__ == "__main__":
    test_full_model()
