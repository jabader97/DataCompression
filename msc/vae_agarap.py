# modified from code found here: https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
import torch


class AE_network(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = torch.nn.Linear(
            in_features=kwargs["input_shape"], out_features=64
        )
        self.encoder_output_layer = torch.nn.Linear(
            in_features=64, out_features=64
        )
        self.decoder_hidden_layer = torch.nn.Linear(
            in_features=64, out_features=64
        )
        self.decoder_output_layer = torch.nn.Linear(
            in_features=64, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features.float())
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed, code

    
class CNN_AE_network(torch.nn.Module):
    # in_dims and out_dims in shape CxHxW (can pass image.shape)
    def __init__(self, in_dims, out_dims):
        super().__init__()
        ks = (in_dims[1] - out_dims[1] + 1, in_dims[2] - out_dims[2] + 1)
        s = (1, 1)
        self.decoder_hidden_layer = torch.nn.Conv2d(
            in_channels=in_dims[0], out_channels=out_dims[0], kernel_size=ks, stride=s
        )
        self.decoder_hidden_layer2 = torch.nn.ConvTranspose2d(
            in_channels=out_dims[0], out_channels=in_dims[0], kernel_size=ks, stride=s
        )

    def forward(self, original):
        latent = self.decoder_hidden_layer(original)
        latent = torch.relu(latent)
        output = self.decoder_hidden_layer2(latent)
        output = torch.relu(output)
        return output, latent

    
class AE:
    def __init__(self):
        self.model = None

    def train_ae(self, train_data, model, epochs=10):
        train_data = torch.from_numpy(train_data).float()
        self.model = model
        # exp: AE_network(input_shape=len(train_data[0]))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        for epoch in range(epochs):
            loss = 0
            for data in train_data:

                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()

                # compute reconstructions
                data = torch.unsqueeze(data, 0)  # can remove this for AE_network
                outputs, _ = self.model(data)

                # compute training reconstruction loss
                train_loss = criterion(outputs, data)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

            # compute the epoch training loss
            loss = loss / len(train_data)

            # display the epoch training loss
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    def get_latent(self, original_images):
        original_images = torch.from_numpy(original_images).float()
        latent_images = []
        for image in original_images:
            image = torch.unsqueeze(image, 0)  # can remove this for AE_network
            _, latent = self.model(image)
            latent_images.append(latent.tolist())
        return latent_images
