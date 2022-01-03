from os import path as osp

import mmcv
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only
from torch import nn
from torchvision import transforms, utils
from torchvision.utils import save_image

from captcha_dataset import CaptchaDataset


class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=False):
        """
        Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


class UpSampleConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel=4,
            strides=2,
            padding=1,
            activation=True,
            batchnorm=False,
            dropout=False
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()

        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 128 x 64 x 64
            DownSampleConv(64, 128),  # bs x 256 x 32 x 32
            DownSampleConv(128, 256),  # bs x 512 x 16 x 16
            DownSampleConv(256, 512),  # bs x 512 x 8 x 8
            DownSampleConv(512, 512),  # bs x 512 x 4 x 4
            DownSampleConv(512, 512),  # bs x 512 x 2 x 2
            DownSampleConv(512, 512, batchnorm=False),  # bs x 512 x 1 x 1
        ]

        # decoder/upsample convs
        self.decoders = [
            UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
            UpSampleConv(1024, 256),  # bs x 512 x 16 x 16
            UpSampleConv(512, 128),  # bs x 256 x 32 x 32
            UpSampleConv(256, 64),  # bs x 128 x 64 x 64
        ]
        self.decoder_channels = [512, 512, 512, 512, 256, 128, 64]
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            # print(x.shape, skip.shape)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        # print(x.shape)
        x = self.final_conv(x)
        return self.tanh(x)


class PatchGAN(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        self.final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        return xn


class Classifier(nn.Module):
    def __init__(self, input_channels, n_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False),
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
                                   nn.Dropout(0.4))

        self.fc = [nn.Linear(4608, n_classes) for i in range(4)]

    def forward(self, x):
        x = self.bakcbone(x)
        cls = []
        for fc in self.fc:
            cls.append(fc(x))
        x = torch.stack(cls).permute(1, 0, 2).flatten(end_dim=1)
        return x


def encode_conditions(conditions, expand=True):
    encoded = torch.zeros(len(conditions), 4, 10).type_as(conditions).float()
    for i, v in enumerate(conditions):
        for j, d in enumerate(f'{v.item():04d}'):
            encoded[i][j][int(d)] = 1
    encoded = encoded.flatten(start_dim=1)
    return encoded if not expand else encoded.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 128, 128)


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def display_progress(epoch_idx, cond, fake, real, figsize=(10, 5)):
    real = utils.make_grid(real.detach().cpu(), padding=50, pad_value=1, normalize=True).permute(1, 2, 0)
    fake = utils.make_grid(fake.detach().cpu(), padding=50, pad_value=1, normalize=True).permute(1, 2, 0)
    cond = cond.detach().cpu()

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Epoch {epoch_idx}')
    ax[0].imshow(real)
    ax[0].set_title("Real Images")
    ax[0].axis("off")
    ax[1].imshow(fake)
    ax[1].set_title("Fake Images")
    ax[1].axis("off")
    for i in range(1, len(cond) + 1):
        ax[1].annotate(f'{cond[i - 1].item():04d}', xy=[50 * i + 128 * (i - 0.5), 50 / 2], ha='center')
    plt.show()


class Pix2Pix(pl.LightningModule):

    def __init__(self, condition_channels, image_channels, learning_rate=0.0002, lambda_cls=200, display_step=25):

        super().__init__()
        self.save_hyperparameters()

        self.display_step = display_step
        self.embedding = nn.Embedding(10, 10)
        self.gen = Generator(condition_channels, image_channels)
        self.patch_gan = PatchGAN(image_channels)
        self.cls = Classifier(image_channels, condition_channels)

        # intializing weights
        self.gen = self.gen.apply(_weights_init)
        self.patch_gan = self.patch_gan.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.cls_criterion = nn.BCEWithLogitsLoss()

    def _disc_step(self, real_images, conditions):
        fake_images = self.gen(self.embedding(conditions)).detach()
        fake_logits = self.patch_gan(fake_images)
        real_logits = self.patch_gan(real_images)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    def _cls_step(self, conditions):
        fake_images = self.gen(self.embedding(conditions))
        cls_logits = self.cls(fake_images)
        cls_loss = self.cls_criterion(cls_logits, conditions.flatten())
        return cls_loss

    def _gen_step(self, conditions):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        fake_images = self.gen(self.embedding(conditions))
        fake_logits = self.patch_gan(fake_images)
        adversarial_loss = self.adversarial_criterion(fake_logits, torch.ones_like(fake_logits))
        return adversarial_loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        if isinstance(lr, (float, int)):
            lr = (lr,) * 3
        disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=lr[0])
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr[1])
        cls_opt = torch.optim.Adam(self.cls.parameters(), lr=lr[2])
        return disc_opt, gen_opt, cls_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, conditions = batch  # note that this 'condition' is a random value not label of the 'real'

        loss = None
        if optimizer_idx == 0:
            loss = self._disc_step(real, conditions)
            self.log('Discriminator Loss', loss, prog_bar=True, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._gen_step(conditions)
            self.log('Generator Loss', loss, prog_bar=True, sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._cls_step(conditions)
            self.log('Classifier Loss', loss, prog_bar=True, sync_dist=True)

        if self.current_epoch % self.display_step == 0 and batch_idx == 0 and optimizer_idx == 1:
            fake = self.gen(self.embedding(conditions)).detach()
            display_progress(self.current_epoch, conditions[:4], fake[:4], real[:4])
        return loss


# %% Init
display_step = 1
lambda_cls = 1
lr = 0.0002
batch_size = 64
condition_channels = 40
image_channels = 3

# %%Main
train_transform = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_dataset = CaptchaDataset('captcha_images/train', 'captcha_train_labels.txt',transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

pix2pix = Pix2Pix(condition_channels, image_channels, learning_rate=lr, lambda_cls=lambda_cls,
                  display_step=display_step)
trainer = pl.Trainer(max_epochs=1000, gpus=1)
trainer.fit(pix2pix, train_loader)
