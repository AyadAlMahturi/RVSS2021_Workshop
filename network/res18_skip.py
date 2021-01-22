import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torchvision import models
from torchvision.models.resnet import model_urls


class Resnet18Skip(nn.Module):
    def __init__(self, args):
        self.args = args
        super(Resnet18Skip, self).__init__()
        # Load pre-trained lyrn_backend
        pre_trained_backbone = models.resnet18(pretrained=True)
        # with torch.no_grad():
        self.res_conv0 = nn.Sequential(*list(
            pre_trained_backbone.children())[:-6])
        self.res_conv1 = nn.Sequential(*list(
            pre_trained_backbone.children())[-6:-5])
        self.res_conv2 = nn.Sequential(*list(
            pre_trained_backbone.children())[-5:-4])
        self.res_conv3 = nn.Sequential(*list(
            pre_trained_backbone.children())[-4:-3])
        self.res_conv4 = nn.Sequential(*list(
            pre_trained_backbone.children())[-3:-2])

        self.top_conv = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.ReLU())
        
        self.skip_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU())
        
        self.skip_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU())

        self.skip_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU())
        
        # backgound is automatically considered as one additional class,
        # with label '0'
        self.seg_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, self.args.n_classes + 1, kernel_size=1, bias=True)
        )
        
        self.criterion = CrossEntropyLoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=args.lr,
                                          weight_decay=args.weight_decay)
        self.lr_scheduler = lr_scheduler.StepLR(
            self.optimiser, gamma=self.args.scheduler_gamma,
            step_size=self.args.scheduler_step)

    def upsample_add(self, low_res_map, high_res_map):
        upsampled_map = nn.UpsamplingBilinear2d(scale_factor=2)(low_res_map)
        return upsampled_map + high_res_map
        
    def forward(self, img):
        # Encoder
        c1 = self.res_conv0(img)
        c2 = self.res_conv1(c1)  # 96 x 128
        c3 = self.res_conv2(c2)  # 48x64
        c4 = self.res_conv3(c3)  # 24x32
        c5 = self.res_conv4(c4)  # 12x16
        p5 = self.top_conv(c5)
        # Decoder
        p4 = self.upsample_add(p5, self.skip_conv1(c4))
        p3 = self.upsample_add(p4, self.skip_conv2(c3))
        p2 = self.upsample_add(p3, self.skip_conv3(c2))# 48x64
        out = nn.ReLU()(p2)
        out = self.seg_conv(out)
        return out

    def step(self, batch):
        image, label = batch
        pred = self.forward(image)
        loss = self.criterion(pred, label)
        return loss

    def get_optimiser(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr,
                                weight_decay=self.args.weight_decay)

    def get_lr_scheduler(self, optimiser):
        """
        Returns:
            This function by default returns None
        """
        return lr_scheduler.StepLR(
            optimiser, gamma=self.args.scheduler_gamma,
            step_size=self.args.scheduler_step)
