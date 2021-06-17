import torch
from tqdm import tqdm


class AE(torch.nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.decoder_hidden_layer = torch.nn.Linear(
            in_features=in_dims, out_features=out_dims
        )
         #had to remove this layer, otherwise to expensive
        # self.decoder_output_layer = torch.nn.Linear(
        #     in_features=out_dims, out_features=out_dims
        # )

    def forward(self, latent):
        layer = self.decoder_hidden_layer(latent)
        layer = torch.relu(layer)
        # layer = self.decoder_output_layer(layer)
        # layer = torch.relu(layer)
        return layer


class CNNAE():
     # TODO: add class
    pass



class Decoder():
    def __init__(self, in_dims, out_dims): # TODO: add decoder class
        self.model = AE(in_dims=in_dims, out_dims=out_dims)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())


    def train(self, buffer, epochs=1000, batch_size=56):
        print(f"Starting decoder training for {epochs} epochs")
        # input as numpy arrays, change to tensors
        # orig_images = torch.from_numpy(orig_images).float()
        # lat_images = torch.from_numpy(lat_images).float()
        # train
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

            # add this images's loss to epoch loss
            total_loss += loss.item()

            # compute the epoch training loss
            total_loss = total_loss / len(orig_images)

            # display the epoch training loss
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, total_loss))




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




