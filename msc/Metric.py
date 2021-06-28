import torch
import sys


class Metric:
    def __init__(self, loss_metric):
        self.loss = loss_metric

    def measure(self, original, reconstructed, latent):
        # get the loss
        accuracy = self.loss(original, reconstructed)

        # get the latent bit rate
        size = sys.getsizeof(latent) * 8
        size = size / len(latent)

        return accuracy.item(), size
