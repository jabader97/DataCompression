from pytorch_msssim import ms_ssim
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


# load data
example_dataset = dset.FakeData(transform=transforms.ToTensor())

first_image = torch.unsqueeze(example_dataset[0][0], 0)
second_image = torch.unsqueeze(example_dataset[100][0], 0)

# apply ms_ssim
print("Between image and itself")
print(1 - ms_ssim(first_image, first_image))
print("Between two images")
print(1 - ms_ssim(first_image, second_image))
print("Between two images (reversed, should be same as above)")
print(1 - ms_ssim(second_image, first_image))

