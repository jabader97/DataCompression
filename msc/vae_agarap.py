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


class AE:
    def __init__(self):
        self.model = None

    def train_ae(self, train_data, epochs=10):
        train_data = torch.from_numpy(train_data).float()

        self.model = AE_network(input_shape=len(train_data[0]))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        for epoch in range(epochs):
            loss = 0
            for data in train_data:

                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()

                # compute reconstructions
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
            _, latent = self.model(image)
            latent_images.append(latent.tolist())
        return latent_images
