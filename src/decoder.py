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



class Decoder():
    def __init__(self, in_dims, out_dims):
        self.model = AE(in_dims=in_dims, out_dims=out_dims)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())


    def train(self, orig_images, lat_images, epochs=1000):
        # input as numpy arrays, change to tensors
        orig_images = torch.from_numpy(orig_images).float()
        lat_images = torch.from_numpy(lat_images).float()
        # train
        for epoch in range(epochs):
            total_loss = 0
            for (orig, latent) in zip(orig_images, lat_images):

                # reset the gradients to zero
                self.optimizer.zero_grad()

                # compute reconstructions
                outputs = self.model(latent)

                # compute reconstruction loss
                loss = self.criterion(outputs, orig)

                # compute gradients
                loss.backward()

                # perform parameter update
                self.optimizer.step()

                # add this images's loss to epoch loss
                total_loss += loss.item()

            # compute the epoch training loss
            total_loss = total_loss / len(orig_images)

            # display the epoch training loss
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, total_loss))


class BaseDecoder:
    """Basic idea for the training of the decoder
    """
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.decoder.parameters())

    def train(self, images, latents):
        # decoder pass
        reconstructed = self.decoder(latents)
        with torch.no_grad():
            latent_predict = self.encoder(reconstructed)
        loss = self.criterion(latents, latent_predict)


        # reset the gradients to zero
        self.optimizer.zero_grad()

        # compute gradients
        loss.backward()

        # perform parameter update
        self.optimizer.step()




