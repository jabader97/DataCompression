import torch
import sys


class Metric:
    def __init__(self):
        pass

    def measure(self, original, reconstructed, latent):
        # get the loss
        loss = torch.nn.MSELoss()
        accuracy = loss(original, reconstructed)

        # get the latent bit rate
        size = sys.getsizeof(latent)

        return accuracy, size / sum(list(latent.size()))
