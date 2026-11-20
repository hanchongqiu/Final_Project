"""SQ1-specific regression checks for day ordering and engineered pixel features."""

import numpy as np
import torch

from experiments.exp_sq1_inductive_bias import TemporalCNN, pixel_features


def test_temporal_channels_preserve_day_and_price_coordinates():
    chart = torch.arange(64 * 60, dtype=torch.float32).reshape(1, 1, 64, 60)
    seen = []
    model = TemporalCNN().eval()
    hook = model.layers[0].register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    with torch.no_grad():
        model(chart)
    hook.remove()
    for day in range(20):
        expected = chart[0, 0, :, 3 * day : 3 * day + 3].reshape(192)
        torch.testing.assert_close(seen[0][0, :, day], expected)


def test_summary_features_follow_five_day_boundaries():
    chart = torch.zeros(1, 1, 64, 60, dtype=torch.uint8)
    chart[:, :, 10:21, 1::3] = 255
    for day in range(20):
        height = 1 + day // 5
        chart[:, :, -height:, 3 * day + 1] = 255
    features = pixel_features(chart).reshape(4, 12)
    np.testing.assert_allclose(features[:, :7], np.tile([35, 0, 0, 40, 30, 10, 10], (4, 1)))
    np.testing.assert_array_equal(features[:, 7], [1, 2, 3, 4])
    np.testing.assert_array_equal(features[:, 10], 0)
