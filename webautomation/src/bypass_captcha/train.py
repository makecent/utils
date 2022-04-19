import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms, utils
from torchvision.datasets import MNIST


class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
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
            self.bn = nn.InstanceNorm2d(out_channels)

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
            batchnorm=True,
            dropout=False
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.InstanceNorm2d(out_channels)

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


class UnetGenerator(nn.Module):

    def __init__(self, in_channels, out_channels):
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


# class CA_NET(nn.Module):
#     # some code is modified from vae examples
#     # (https://github.com/pytorch/examples/blob/master/vae/main.py)
#     def __init__(self):
#         super(CA_NET, self).__init__()
#         self.t_dim = 100
#         self.c_dim = 100
#         self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
#         self.relu = nn.ReLU()
#
#     def encode(self, text_embedding):
#         x = self.relu(self.fc(text_embedding))
#         mu = x[:, :self.c_dim]
#         logvar = x[:, self.c_dim:]
#         return mu, logvar
#
#     def reparametrize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         eps = torch.cuda.FloatTensor(std.size()).normal_()
#         eps = Variable(eps)
#         return eps.mul(std).add_(mu)
#
#     def forward(self, text_embedding):
#         mu, logvar = self.encode(text_embedding)
#         c_code = self.reparametrize(mu, logvar)
#         return c_code, mu, logvar

class UpGenerator(nn.Module):

    @staticmethod
    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    @staticmethod
    def upBlock(in_planes, out_planes):
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            UpGenerator.conv3x3(in_planes, out_planes),
            nn.InstanceNorm2d(out_planes),
            nn.ReLU(True))
        return block

    def __init__(self, embedding_channels, n_objects, noise_channels, image_channels=3, latent_channels=128):
        super().__init__()
        self.embedding_channels = embedding_channels
        self.n_objects = n_objects
        self.noise_channels = noise_channels
        self.image_channels = image_channels
        self.latent_channels = latent_channels

        # -> ngf x 4 x 4
        self.embedding = nn.Embedding(10, embedding_channels)
        self.fc = nn.Sequential(
            nn.Linear(embedding_channels * n_objects + noise_channels, latent_channels * 4 * 4, bias=False),
            nn.BatchNorm1d(latent_channels * 7 * 7),
            nn.ReLU(True))

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = self.upBlock(latent_channels, latent_channels // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = self.upBlock(latent_channels // 2, latent_channels // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = self.upBlock(latent_channels // 4, latent_channels // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = self.upBlock(latent_channels // 8, latent_channels // 16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            UpGenerator.conv3x3(latent_channels // 16, image_channels),
            nn.Tanh())

    def forward(self, x):
        embeddings = self.embedding(x).flatten(start_dim=1)
        noises = torch.randn(embeddings.shape[0], self.noise_channels).type_as(embeddings)
        x = torch.cat((embeddings, noises), 1)
        x = self.fc(x)

        x = x.view(-1, self.latent_channels, 4, 4)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        # state size 3 x 64 x 64
        fake_img = self.img(x)
        return fake_img


# class PatchGAN(nn.Module):
#
#     def __init__(self, input_channels):
#         super().__init__()
#         self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
#         self.d2 = DownSampleConv(64, 128)
#         self.d3 = DownSampleConv(128, 256)
#         self.d4 = DownSampleConv(256, 512)
#         self.final = nn.Conv2d(512, 1, kernel_size=1)
#
#     def forward(self, x):
#         x = self.d1(x)
#         x = self.d2(x)
#         x = self.d3(x)
#         x = self.d4(x)
#         x = self.final(x)
#         return x


class Classifier(nn.Module):
    def __init__(self, input_channels, n_objects=4, n_classes=10):
        super().__init__()
        self.n_objects = n_objects
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        for i in range(n_objects):
            setattr(self, f'fc{i + 1}', nn.Linear(512, n_classes))

    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.pool(x).flatten(start_dim=1)
        cls = []
        for i in range(self.n_objects):
            cls.append(getattr(self, f'fc{i + 1}')(x))
        x = torch.stack(cls).permute(1, 0, 2).flatten(end_dim=1)
        return x


class Discriminator(Classifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.pool(x).flatten(start_dim=1)
        x = self.fc(x)
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


def decode_cond(cond):
    decoded = []
    for row in cond:
        decoded.append(int(''.join(map(str, row))))
    return decoded


def display_progress(epoch_idx, cond, fake, real, figsize=(10, 2), save_fig=False):
    img_size = 64
    padding = img_size // 2

    real = utils.make_grid(real.detach().cpu(), padding=padding, pad_value=0, normalize=True).permute(1, 2, 0)
    fake = utils.make_grid(fake.detach().cpu(), padding=padding, pad_value=0, normalize=True).permute(1, 2, 0)
    cond = decode_cond(cond.detach().cpu().numpy())

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Epoch {epoch_idx}')
    ax[0].imshow(real)
    ax[0].set_title("Real Images")
    ax[0].axis("off")
    ax[1].imshow(fake)
    ax[1].set_title("Fake Images")
    ax[1].axis("off")
    for i in range(1, len(cond) + 1):
        ax[1].annotate(f'{cond[i - 1]:04d}', xy=[padding * i + img_size * (i - 0.5), padding / 2], ha='center',
                       color='w')
    plt.show()
    fig.savefig(f'produced_images/epoch_{epoch_idx}.jpg')


class Pix2Pix(pl.LightningModule):

    def __init__(self,
                 embedding_channels,
                 noise_channels,
                 image_channels=1,
                 latent_channels=128,
                 n_objects=1,
                 n_classes=10,
                 learning_rate=(0.0002, 0.0002, 0.0002),
                 lambda_cls=200,
                 display_step=25):

        super().__init__()
        self.save_hyperparameters()

        self.display_step = display_step
        self.gen = UpGenerator(embedding_channels, n_objects, noise_channels, image_channels, latent_channels)
        self.dsc = Discriminator(image_channels)
        self.cls = Classifier(image_channels, n_objects, n_classes)

        # intializing weights
        self.gen = self.gen.apply(_weights_init)
        self.dsc = self.dsc.apply(_weights_init)
        self.cls = self.cls.apply(_weights_init)

        self.adversarial_criterion = nn.CrossEntropyLoss()
        self.cls_criterion = nn.CrossEntropyLoss()

    def _disc_step(self, real_images, conditions):
        fake_images = self.gen(conditions).detach()
        fake_logits = self.dsc(fake_images)
        real_logits = self.dsc(real_images)

        fake_loss = self.adversarial_criterion(fake_logits,
                                               torch.zeros(fake_logits.shape[0]).type_as(fake_logits).long())
        real_loss = self.adversarial_criterion(real_logits,
                                               torch.ones(real_logits.shape[0]).type_as(real_logits).long())
        return (real_loss + fake_loss) / 2

    def _cls_step(self, conditions):
        fake_images = self.gen(conditions)
        cls_logits = self.cls(fake_images)
        cls_loss = self.cls_criterion(cls_logits, conditions.flatten())
        return cls_loss

    def _gen_step(self, conditions):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        fake_images = self.gen(conditions)
        fake_logits = self.dsc(fake_images)
        adversarial_loss = self.adversarial_criterion(fake_logits,
                                                      torch.ones(fake_logits.shape[0]).type_as(fake_logits).long())
        return adversarial_loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        if isinstance(lr, (float, int)):
            lr = (lr,) * 3
        disc_opt = torch.optim.Adam(self.dsc.parameters(), lr=lr[0])
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr[1])
        cls_opt = torch.optim.Adam(list(self.cls.parameters()) + list(self.gen.parameters()), lr=lr[2])
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

        if self.current_epoch % self.display_step == 0 and batch_idx == 4 and optimizer_idx == 2:
            fake = self.gen(conditions).detach()
            cls_pre = self.cls(fake).argmax(dim=-1).view(conditions.shape[0], self.hparams.n_objects)
            acc = torch.eq(cls_pre.flatten(), conditions.flatten())
            acc = acc.sum() / len(acc)
            self.log('Acc. Fake', acc, prog_bar=True, sync_dist=True)
            display_progress(self.current_epoch, conditions[:4], fake[:4], real[:4], save_fig=True)
        return loss


# %% Init
display_step = 10
lambda_cls = 1
lr = (0.0002, 0.0002, 0.001)
batch_size = 64
embedding_vector_len = 12
condition_channels = embedding_vector_len * 1
noise_channels = 12

image_shape = 28
init_feat_shape = 7
# %%Main
# train_transform = transforms.Compose([
#     transforms.Resize([64, 64]),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# train_dataset = CaptchaDataset('images', num=8229, transform=train_transform)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

train_loader = torch.utils.data.DataLoader(MNIST('files/', train=True, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(
                                                         (0.1307,), (0.3081,))
                                                 ])),
                                           batch_size=batch_size, shuffle=True)

pix2pix = Pix2Pix(condition_channels, noise_channels, learning_rate=lr, lambda_cls=lambda_cls, display_step=display_step)
trainer = pl.Trainer(max_epochs=1000, gpus=1)
trainer.fit(pix2pix, train_loader)
