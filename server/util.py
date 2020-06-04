from __future__ import division
import os

import torch
import torch.nn as nn
# from torch.utils.serialization import load_lua  # 最新版torch没有
from torchfile import load
import torchvision.utils as vutils

from modelsNIPS import decoder1, decoder2, decoder3, decoder4, decoder5
from modelsNIPS import encoder1, encoder2, encoder3, encoder4, encoder5


class WCT(nn.Module):
    def __init__(self, args):
        super(WCT, self).__init__()
        # 加载预训练的vgg：
        vgg1 = load(args.vgg1, force_8bytes_long=True)
        # vgg1 = vgg1.eval()
        decoder1_torch = load(args.decoder1, force_8bytes_long=True)
        vgg2 = load(args.vgg2, force_8bytes_long=True)
        decoder2_torch = load(args.decoder2, force_8bytes_long=True)
        vgg3 = load(args.vgg3, force_8bytes_long=True)
        decoder3_torch = load(args.decoder3, force_8bytes_long=True)
        vgg4 = load(args.vgg4, force_8bytes_long=True)
        decoder4_torch = load(args.decoder4, force_8bytes_long=True)
        vgg5 = load(args.vgg5, force_8bytes_long=True)
        decoder5_torch = load(args.decoder5, force_8bytes_long=True)

        self.e1 = encoder1(vgg1)
        self.d1 = decoder1(decoder1_torch)
        self.e2 = encoder2(vgg2)
        self.d2 = decoder2(decoder2_torch)
        self.e3 = encoder3(vgg3)
        self.d3 = decoder3(decoder3_torch)
        self.e4 = encoder4(vgg4)
        self.d4 = decoder4(decoder4_torch)
        self.e5 = encoder5(vgg5)
        self.d5 = decoder5(decoder5_torch)

        # self.e1 = torch.load('visdom_/e1.pth')
        # self.d1 = torch.load('visdom_/d1.pth')
        # self.e2 = torch.load('visdom_/e2.pth')
        # self.d2 = torch.load('visdom_/d2.pth')
        # self.e3 = torch.load('visdom_/e3.pth')
        # self.d3 = torch.load('visdom_/d3.pth')
        # self.e4 = torch.load('visdom_/e4.pth')
        # self.d4 = torch.load('visdom_/d4.pth')
        # self.e5 = torch.load('visdom_/e5.pth')
        # self.d5 = torch.load('visdom_/d5.pth')

    def whiten_and_color(self, cF, sF):
        cFSize = cF.size()
        c_mean = torch.mean(cF, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean

        contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double()
        c_u, c_e, c_v = torch.svd(contentConv, some=False)

        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        sFSize = sF.size()
        s_mean = torch.mean(sF, 1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF, sF.t()).div(sFSize[1] - 1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)

        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cF)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature

    def transform(self, cF, sF, csF, alpha):
        cF = cF.double()
        sF = sF.double()
        C, W, H = cF.size(0), cF.size(1), cF.size(2)
        _, W1, H1 = sF.size(0), sF.size(1), sF.size(2)
        cFView = cF.view(C, -1)
        sFView = sF.view(C, -1)

        targetFeature = self.whiten_and_color(cFView, sFView)
        targetFeature = targetFeature.view_as(cF)

        # wct_layer = targetFeature.clone().detach()
        # wct_layer = wct_layer.to(torch.device('cpu'))
        # wct_layer = wct_layer.float().squeeze(0)
        # print(wct_layer.size(), wct_layer.dtype, wct_layer.ndimension())
        # vutils.save_image(wct_layer, os.path.join('in-place/', '_wct_layer5.jpg'))
        # self.save_image(vutils.make_grid(targetFeature, normalize=True))

        ccsF = alpha * targetFeature + (1.0 - alpha) * cF
        ccsF = ccsF.float().unsqueeze(0)
        # csF.data.resize_(ccsF.size()).copy_(ccsF)
        with torch.no_grad():
            csF.resize_(ccsF.size()).copy_(ccsF)
            # wct_layer.resize_(targetFeature).copy_()
        return csF

    @staticmethod
    def save_image(tensor):
        from torchvision import transforms
        import matplotlib.pyplot as plt
        import pylab
        unloader = transforms.ToPILImage()
        image = tensor.cpu().clone()  # clone the tensor
        image = image.float().squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        pylab.show()
        image.save('in-place/example.jpg')
        plt.pause(0.1)

    def visdom_(self):
        import shutil

        # 保存模型用于可视化
        model_dict = {'e1': self.e1, 'd1': self.d1,
                      'e2': self.e2, 'd2': self.d2,
                      'e3': self.e3, 'd3': self.d3,
                      'e4': self.e4, 'd4': self.d4,
                      'e5': self.e5, 'd5': self.d5}
        print('2')
        # try:
        #     if os.path.exists('visdom_/'):
        #         shutil.rmtree('visdom_/')
        #     os.mkdir('visdom_/')
        # except OSError:
        #     pass

        # for k, v in model_dict.items():
        #     self.save_model(v, k+'.pth')

    @staticmethod
    def save_model(model, name):
        torch.save(model, os.path.join('visdom_', name))
        print('1')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='WCT Pytorch')
    parser.add_argument('--contentPath', default='images/content', help='path to train')
    parser.add_argument('--stylePath', default='images/style', help='path to train')
    parser.add_argument('--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--vgg1', default='models/vgg_conv1_1.t7', help='Path to the VGG conv1_1')
    parser.add_argument('--vgg2', default='models/vgg_conv2_1.t7', help='Path to the VGG conv2_1')
    parser.add_argument('--vgg3', default='models/vgg_conv3_1.t7', help='Path to the VGG conv3_1')
    parser.add_argument('--vgg4', default='models/vgg_conv4_1.t7', help='Path to the VGG conv4_1')
    parser.add_argument('--vgg5', default='models/vgg_conv5_1.t7', help='Path to the VGG conv5_1')
    parser.add_argument('--decoder5', default='models/feature_conv5_1.t7', help='Path to the decoder5')
    parser.add_argument('--decoder4', default='models/feature_conv4_1.t7', help='Path to the decoder4')
    parser.add_argument('--decoder3', default='models/feature_conv3_1.t7', help='Path to the decoder3')
    parser.add_argument('--decoder2', default='models/feature_conv2_1.t7', help='Path to the decoder2')
    parser.add_argument('--decoder1', default='models/feature_conv1_1.t7', help='Path to the decoder1')

    args = parser.parse_args()

    wct = WCT(args)
    wct.visdom_()
