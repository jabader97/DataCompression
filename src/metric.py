import torch
from scipy.stats import entropy
from torch.nn import MSELoss

@torch.no_grad()
def evaluate(encoder, decoder, buffer, flattened=True):
    test_images, test_latents = buffer.get_all()

    if flattened:
        test_images_encoder = torch.reshape(test_images, (test_images.shape[0], 3, 210, 160))
    else:
        test_images_encoder = test_images
    
    # forward pass of images
    latents = encoder(test_images_encoder)
    
    # optional rounding: will latents already be integers?
    latents = torch.round(latents)

    # count occurences of all values:
    _, counts = torch.unique(latents, return_counts=True)

    # calculate the entropy of the latents
    latents_entropy = entropy(counts)
    print(f"The entropy of the latents is: {latents_entropy}")

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

    return latents_entropy, L2_loss.item(), agent_loss.item()