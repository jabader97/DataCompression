import torch
from tqdm import tqdm
import time


class AE(torch.nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.decoder_hidden_layer = torch.nn.Linear(
            in_features=in_dims, out_features=out_dims
        )

    def forward(self, latent):
        layer = self.decoder_hidden_layer(latent)
        layer = torch.relu(layer)
        return layer


class CNN_AE(torch.nn.Module):
    # in_dims and out_dims in shape CxHxW (can pass image.shape)
    def __init__(self, in_dims, out_dims):
        super().__init__()
        ks = (out_dims[1] - in_dims[1] + 1, out_dims[2] - in_dims[2] + 1)
        s = (1, 1)
        self.decoder_hidden_layer = torch.nn.ConvTranspose2d(
            in_channels=in_dims[0], out_channels=out_dims[0], kernel_size=ks, stride=s
        )

    def forward(self, latent):
        layer = self.decoder_hidden_layer(latent)
        layer = torch.relu(layer)
        return layer


class Decoder:
    # example:
    # model = AE(in_dims=in_dims, out_dims=out_dims)
    # loss = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(self.model.parameters())
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.criterion = loss
        self.optimizer = optimizer

    def train(self, buffer, epochs=1000, batch_size=56):
        print(f"Training decoder for {epochs} epochs")
        time.sleep(1) # to avoid printstream clashing with progressbar
        # input as numpy arrays, change to tensors
        # orig_images = torch.from_numpy(orig_images).float()
        # lat_images = torch.from_numpy(lat_images).float()
        
        # train
        epoch_losses = []
        for epoch in tqdm(range(epochs)):
            orig_images, lat_images = buffer.sample(batch_size) # TODO: replace by minibatch sampling
            total_loss = 0
            # for (orig, latent) in zip(orig_images, lat_images):

            # reset the gradients to zero
            self.optimizer.zero_grad()

            # compute reconstructions
            outputs = self.model(lat_images)

            # compute reconstruction loss
            loss = self.criterion(outputs, orig_images)

            # compute gradients
            loss.backward()

            # perform parameter update
            self.optimizer.step()

            # add this images's loss to epoch losses (loss per image normalized)
            epoch_losses.append(loss.item()/batch_size)

            # display the epoch training loss
            if epoch % 100 == 0:
                print("\n epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, epoch_losses[-1]))
                time.sleep(1) # to avoid printstream clashing with progressbar

        return epoch_losses

    def save(self, filepath):
        if filepath[-1] != "/":
            filepath += "/"
        torch.save(self.model.state_dict(), filepath + "decoder")

    def load(self, filepath):
        if filepath[-1] != "/":
            filepath += "/"
        self.model.load_state_dict(torch.load(filepath + "decoder"))


class BaseDecoder:
    """Basic idea for the training of the decoder
    """
    def __init__(self, decoder):
        self.decoder = decoder
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.decoder.parameters())

    def train(self, images, latents):
        # decoder pass
        reconstructed = self.decoder(latents) # decode latent

        # method 1
        # with torch.no_grad():
        #     latent_predict = self.encoder(reconstructed) # encode again (new latent)
        # loss = self.criterion(latents, latent_predict)

        # method 2
        loss = self.criterion(images, reconstructed)

        # reset the gradients to zero
        self.optimizer.zero_grad()

        # compute gradients
        loss.backward()

        # perform parameter update
        self.optimizer.step()




