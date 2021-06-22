import torch
from tqdm import tqdm
import time


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




