import numpy as np
import torch

class Decoder_Buffer():
    """A simple buffer for the decoder
    """

    def __init__(self, image_dim, latent_dim, buffer_size) -> None:
        self.images = torch.zeros([buffer_size] + image_dim)
        self.latents = torch.zeros([buffer_size, latent_dim])
        self.pointer = 0
        self.buffer_size = buffer_size
        self.full = False


    def add(self, image, latent):
        if self.pointer == self.buffer_size:
            self.full = True
            self.pointer = 0 # reset pointer
        self.images[self.pointer] = image
        self.latents[self.pointer] = latent
        self.pointer += 1

    def sample(self, size):
        upper_bound = self.buffer_size if self.full else self.pointer
        sample_index = np.random.randint(0, upper_bound, size=size)

        return self.images[sample_index], self.latents[sample_index]