from DataCompression.src.decoder_MSE import CNN_AE
from DataCompression.msc.vae_agarap import AE
import torch
import numpy as np


# import data
data = np.load('digits389.npy', allow_pickle=True).item()
x_test = data['Xtest']
x_train = data['Xtrain']
y_test = data['Ytest']
y_train = data['Ytrain']

# train the original vae to get the low dim spaces (this is slow because used single batches)
print("Original VAE loss:")
ae = AE()
ae.train_ae(x_train, epochs=2)
# get the lower dimensional space it produced
latent = ae.get_latent(x_train)

# reformat the data to fit the decoder method
latent = torch.transpose(torch.tensor(np.expand_dims(np.reshape(np.asarray(latent), (len(latent), 8, 8)), -1)), 1, 3).float()
x_train = torch.transpose(torch.tensor(np.expand_dims(np.reshape(x_train, (len(latent), 16, 16)), -1)), 1, 3).float()


# create the model
model = CNN_AE(in_dims=latent[0].shape, out_dims=x_train[0].shape)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 2

# train the model
epoch_losses = []
for epoch in range(epochs):
    total_loss = 0
    # reset the gradients to zero
    optimizer.zero_grad()

    # compute reconstructions
    outputs = model(latent)
    outputs = torch.squeeze(outputs, 0)

    # compute reconstruction loss
    loss = criterion(outputs, x_train)

    # compute gradients
    loss.backward()

    # perform parameter update
    optimizer.step()

    # add this images's loss to epoch losses (loss per image normalized)
    epoch_losses.append(loss.item() / len(y_train))

        # display the epoch training loss
    if epoch % 100 == 0:
        print("\n epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, epoch_losses[-1]))
