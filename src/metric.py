import torch
from scipy.stats import entropy
from torch.nn import MSELoss
import numpy as np

@torch.no_grad()
def evaluate(encoder, decoder, buffer, flattened=True, n_digits = 0):
    """Evaluates the Final Encoder/Decoder.

    Args:
        encoder (feature_extractor): The feature extractor used to generate input -> latents.
        decoder (NN): The decoder Neural Network, latents -> predicted input
        buffer (Decoderbuffer): Buffer who contains the images to test on.
        flattened (bool, optional): Wether the buffer contains the images as a flattened representation. Defaults to True.
        n_digits (int, optional): The number of decimal digits to round to. Defaults to 0.

    Returns:
        [type]: [description]
    """
    test_images, test_latents = buffer.get_all()

    if flattened:
        test_images_encoder = torch.reshape(test_images, (test_images.shape[0], 3, 210, 160))
    else:
        test_images_encoder = test_images
    
    # forward pass of images
    latents = encoder(test_images_encoder)
    
    # rounding
    latents = torch.round(latents * 10**n_digits) / (10**n_digits)

    # calculate entropy per dimension:
    latents_entropy = [entropy(np.unique(latent_i, return_counts=True)[1]) for latent_i in latents.numpy().T]

    # take mean entropy
    entropy_mean = np.mean(latents_entropy)
    print(f"The mean entropy of each latent dim is: {entropy_mean}")

    # pass through decoder
    reconstructed = decoder(latents)

    # calculate L2 loss on images
    mse_loss = MSELoss()
    L2_loss = mse_loss(test_images, reconstructed)
    print(f"The L2-loss in image space is: {L2_loss}")

    # calculate agent loss
    if flattened:
        reconstructed_encoder = torch.reshape(reconstructed, (reconstructed.shape[0], 3, 210, 160))
    else:
        reconstructed_encoder = test_images
    new_latents = encoder(reconstructed_encoder)
    agent_loss = mse_loss(new_latents, latents)
    print(f"The L2-loss in agent space is: {agent_loss}")

    return entropy_mean, L2_loss.item(), agent_loss.item()
