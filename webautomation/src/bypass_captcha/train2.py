import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from captcha_dataset import CaptchaDataset

torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

train_transform = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_dataset = CaptchaDataset('captcha_train_labels.txt', 'captcha_images/train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
image_shape = (3, 128, 128)
image_dim = int(np.prod(image_shape))
latent_dim = 1000

n_classes = 10000
embedding_dim = 1000


# custom weights initialization called on generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_conditioned_generator = nn.Sequential(nn.Embedding(n_classes, embedding_dim),
                                                         nn.Linear(embedding_dim, 16))

        self.latent = nn.Sequential(nn.Linear(latent_dim, 4 * 4 * 512),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(nn.ConvTranspose2d(513, 64 * 8, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64 * 8, momentum=0.1, eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64 * 4, momentum=0.1, eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64 * 2, momentum=0.1, eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64 * 2, 64 * 1, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64 * 1, momentum=0.1, eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64 * 1, 3, 4, 2, 1, bias=False),
                                   nn.Tanh())

    def forward(self, inputs):
        noise_vector, label = inputs
        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 1, 4, 4)
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 512, 4, 4)
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)
        # print(image.size())
        return image


generator = Generator().to(device)
generator.apply(weights_init)
print(generator)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_condition_disc = nn.Sequential(nn.Embedding(n_classes, embedding_dim),
                                                  nn.Linear(embedding_dim, 3 * 128 * 128))

        self.model = nn.Sequential(nn.Conv2d(6, 64, 4, 2, 1, bias=False),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64, 64 * 2, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64 * 2, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 2, 64 * 4, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64 * 4, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 4, 64 * 8, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64 * 8, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Flatten(),
                                   nn.Dropout(0.4),
                                   nn.Linear(4608, 1),
                                   nn.Sigmoid()
                                   )

    def forward(self, inputs):
        img, label = inputs
        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 3, 128, 128)
        concat = torch.cat((img, label_output), dim=1)
        # print(concat.size())
        output = self.model(concat)
        return output


discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
print(discriminator)

adversarial_loss = nn.BCELoss()


def generator_loss(fake_output, label):
    gen_loss = adversarial_loss(fake_output, label)
    # print(gen_loss)
    return gen_loss


def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss


learning_rate = 0.0002
G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

num_epochs = 30
for epoch in range(1, num_epochs + 1):

    D_loss_list, G_loss_list = [], []

    for index, (real_images, labels) in enumerate(train_loader):
        D_optimizer.zero_grad()
        real_images = real_images.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(1).long()

        real_target = Variable(torch.ones(real_images.size(0), 1).to(device))
        fake_target = Variable(torch.zeros(real_images.size(0), 1).to(device))

        D_real_loss = discriminator_loss(discriminator((real_images, labels)), real_target)
        # print(discriminator(real_images))
        # D_real_loss.backward()

        noise_vector = torch.randn(real_images.size(0), latent_dim, device=device)
        noise_vector = noise_vector.to(device)

        generated_image = generator((noise_vector, labels))
        output = discriminator((generated_image.detach(), labels))
        D_fake_loss = discriminator_loss(output, fake_target)

        # train with fake
        # D_fake_loss.backward()

        D_total_loss = (D_real_loss + D_fake_loss) / 2
        D_loss_list.append(D_total_loss)

        D_total_loss.backward()
        D_optimizer.step()

        # Train generator with real labels
        G_optimizer.zero_grad()
        G_loss = generator_loss(discriminator((generated_image, labels)), real_target)
        G_loss_list.append(G_loss)

        G_loss.backward()
        G_optimizer.step()

    print('Epoch: [%d/%d]: D_loss: %.3f, G_loss: %.3f' % (
        epoch, num_epochs, torch.mean(torch.FloatTensor(D_loss_list)).item(),
        torch.mean(torch.FloatTensor(G_loss_list)).item()))
    real_img = make_grid(real_images.detach().cpu(), padding=2, normalize=True)
    fake_img = make_grid(generated_image.detach().cpu(), padding=2, normalize=True)
    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(real_img, (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(fake_img, (1, 2, 0)))
    plt.show()
