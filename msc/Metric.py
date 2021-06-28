import torch
import sys


class Metric:
    def __init__(self, loss_metric):
        self.loss = loss_metric

    def measure(self, original, reconstructed, latent):
        # get the loss
        accuracy = self.loss(original, reconstructed)

        # get the latent bit rate
        size = sys.getsizeof(latent)
        dims = list(original[0].size())
        for dim in dims:
            size /= dim

        return accuracy.item(), size * 8
