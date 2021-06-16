import torch


class AE(torch.nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.decoder_hidden_layer = torch.nn.Linear(
            in_features=in_dims, out_features=out_dims
        )
        self.decoder_output_layer = torch.nn.Linear(
            in_features=out_dims, out_features=out_dims
        )

    def forward(self, latent):
        layer = self.decoder_hidden_layer(latent)
        layer = torch.relu(layer)
        layer = self.decoder_output_layer(layer)
        return torch.relu(layer)


# method to decode according to MSE
# takes inputs:
# orig_images = ndarray of the original images n x d (n = num images, d = dims of the images in vector form)
# lat_images = ndarray of the compressed images n x d (n = num images, d = dims of the images in vector form)
# optional: epochs = num times to train (default = 1000)
def decode(orig_images, lat_images, epochs=1000):
    # input as numpy arrays, change to tensors
    orig_images = torch.from_numpy(orig_images).float()
    lat_images = torch.from_numpy(lat_images).float()
    # train
    criterion = torch.nn.MSELoss()
    model = AE(in_dims=lat_images.size()[1], out_dims=orig_images.size()[1])
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        total_loss = 0
        for (orig, latent) in zip(orig_images, lat_images):

            # reset the gradients to zero
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(latent)

            # compute reconstruction loss
            loss = criterion(outputs, orig)

            # compute gradients
            loss.backward()

            # perform parameter update
            optimizer.step()

            # add this images's loss to epoch loss
            total_loss += loss.item()

        # compute the epoch training loss
        total_loss = total_loss / len(orig_images)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, total_loss))
