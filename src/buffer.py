import numpy as np
import torch
import io

class Decoder_Buffer():
    """A simple buffer for the decoder
    """

    def __init__(self, image_dim, latent_dim, buffer_size, to_gray = False, flatten = False) -> None:
        if to_gray:
            image_dim = image_dim[1:] # if rgb to gray, rgb dim (0) gets dropped
        if flatten:
            image_dim = [int(np.prod(image_dim))] # a flat image is multiplication of all its layers
        self.images = torch.zeros([buffer_size] + image_dim)
        self.latents = torch.zeros([buffer_size, latent_dim])
        self.pointer = 0
        self.buffer_size = buffer_size
        self.full = False
        self.to_gray = to_gray
        self.flatten = flatten

        print(f"Created a buffer for images of shape {self.images.shape} and latents {self.latents.shape}")


    def add(self, image, latent):
        if self.pointer == self.buffer_size:
            self.full = True
            self.pointer = 0 # reset pointer

        # if it has one more dim for nsamples drop
        if latent.shape[0] == 1:
            latent = latent[0]       
        if image.shape[0] == 1: 
            image = image[0]
        # convert to grayscale    
        if self.to_gray: 
            image = self.rgb2gray(image)
        # flatten
        if self.flatten:  
            image = torch.flatten(image)
        self.images[self.pointer] = image
        self.latents[self.pointer] = latent
        self.pointer += 1

    def sample(self, size):
        upper_bound = self.buffer_size if self.full else self.pointer
        sample_index = np.random.randint(0, upper_bound, size=size)
        return self.images[sample_index], self.latents[sample_index]

    def rgb2gray(self, rgb):
        """RGB image to gray

        Args:
            rgb (array/tensor color*height*width): The input array

        Returns:
            array/tensor (height*width): The greyscale image
        """
        r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def get_image_dims(self):
        return list(self.images.shape[1:])


    def save(self, filepath):
        if filepath[-1] != "/":
            filepath += "/"
        torch.save(self.images, filepath + "images.pt")
        torch.save(self.latents, filepath + "latents.pt")
        torch.save(self.pointer, filepath + "pointer.pt" )

    def load(self, filepath):
        if filepath[-1] != "/":
            filepath += "/"
        with open(filepath + 'images.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())
        self.images = torch.load(buffer)
        with open(filepath + 'pointer.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())
        self.pointer = torch.load(buffer)
        with open(filepath + 'latents.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())
        self.latents = torch.load(buffer)
