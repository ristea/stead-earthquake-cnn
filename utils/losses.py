import torch
import torch.nn


def mse_loss(pred_depth, pred_distance, pred_magnitude,
             labels_depth, labels_distance, labels_magnitude):
    mse = torch.nn.MSELoss()

    loss_depth = mse(pred_depth, labels_depth)
    loss_distance = mse(pred_distance, labels_distance)
    loss_magn = mse(pred_magnitude, labels_magnitude)

    return loss_depth + loss_distance + loss_magn
