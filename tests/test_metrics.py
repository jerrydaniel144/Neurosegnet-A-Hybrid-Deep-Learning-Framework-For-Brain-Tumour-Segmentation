import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric

def test_dice_metric():
    pred = torch.randint(0, 2, (2, 3, 128, 128, 128))
    label = torch.randint(0, 2, (2, 3, 128, 128, 128))

    metric = DiceMetric(include_background=True, reduction="mean")
    score = metric(pred, label)
    assert score.shape == (1,), "Dice metric output should be scalar."

def test_hd95_metric():
    pred = torch.randint(0, 2, (2, 3, 128, 128, 128))
    label = torch.randint(0, 2, (2, 3, 128, 128, 128))

    metric = HausdorffDistanceMetric(include_background=True, percentile=95.0, reduction="mean")
    score = metric(pred, label)
    assert score.shape == (1,), "HD95 metric output should be scalar."