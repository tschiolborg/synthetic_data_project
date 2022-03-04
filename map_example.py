import torch
from torchmetrics.detection.map import MeanAveragePrecision

# Preds should be a list of elements, where each element is a dict
# containing 3 keys: boxes, scores, labels
# boxes should be on Pascal VOC format
# (xmin-top left, ymin-top left, xmax-bottom right, ymax-bottom right).
# Implementation follows pycocotools (standard for mAP)

preds = [
    dict(
        # The boxes keyword should contain an [N,4] tensor,
        # where N is the number of detected boxes with boxes of the format
        # [xmin, ymin, xmax, ymax] in absolute image coordinates
        boxes=torch.Tensor([[258.0, 41.0, 606.0, 285.0]]),
        # The scores keyword should contain an [N,] tensor where
        # each element is confidence score between 0 and 1
        scores=torch.Tensor([0.536]),
        # The labels keyword should contain an [N,] tensor
        # with integers of the predicted classes
        labels=torch.IntTensor([0]),
    )
]

# Target should be a list of elements, where each element is a dict
# containing 2 keys: boxes and labels. Each keyword should be formatted
# similar to the preds argument. The number of elements in preds and
# target need to match.
# It only requires these two, so more can be added without getting an error
target = [
    dict(
        boxes=torch.Tensor([[214.0, 41.0, 562.0, 285.0]]),
        labels=torch.IntTensor([0]),
    )
]

if __name__ == "__main__":
    # Initialize metric
    metric = MeanAveragePrecision()

    # Update metric with predictions and respective ground truth
    metric.update(preds, target)

    # Compute the results
    result = metric.compute()
    print(result)

    print(target)
