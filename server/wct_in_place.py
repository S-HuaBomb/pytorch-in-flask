import argparse
import os
# from torch.utils.serialization import load_lua
import time

import torchvision.utils as vutils
from torch.autograd import Variable

from Loader import Dataset
from util import *

parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--contentPath', default='images/content', help='path to train')
parser.add_argument('--stylePath', default='images/style', help='path to train')
parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 4)')
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
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--fineSize', type=int, default=512,
                    help='resize image to fineSize x fineSize,leave it to 0 if not resize')
parser.add_argument('--outf', default='in-place/', help='folder to output images')
parser.add_argument('--alpha', type=float, default=1, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")

args = parser.parse_args()

try:
    os.makedirs(args.outf)
except OSError:
    pass

# Data loading code
dataset = Dataset(args.contentPath, args.stylePath, args.fineSize)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

wct = WCT(args)


def styleTransfer(contentImg, styleImg, imname, csF):
    sF5 = wct.e5(styleImg)
    cF5 = wct.e5(contentImg)
    sF5 = sF5.data.cpu().squeeze(0)
    cF5 = cF5.data.cpu().squeeze(0)
    # csF5 = wct.transform(cF5, sF5, csF, args.alpha)
    csF5, wct_layer5 = wct.transform(cF5, sF5, csF, args.alpha)
    # vutils.save_image(wct_layer5, os.path.join(args.outf, imname+'_wct_layer5.jpg'))
    Im5 = wct.d5(csF5)
    # vutils.save_image(Im5.data.cpu().float(), os.path.join(args.outf, imname+'_de5.jpg'))

    # sF4 = wct.e4(styleImg)
    # cF4 = wct.e4(Im5)
    # sF4 = sF4.data.cpu().squeeze(0)
    # cF4 = cF4.data.cpu().squeeze(0)
    # csF4 = wct.transform(cF4, sF4, csF, args.alpha)
    # Im4 = wct.d4(csF4)
    # # vutils.save_image(Im4.data.cpu().float(), os.path.join(args.outf, imname+'_de4.jpg'))
    #
    # sF3 = wct.e3(styleImg)
    # cF3 = wct.e3(Im4)
    # sF3 = sF3.data.cpu().squeeze(0)
    # cF3 = cF3.data.cpu().squeeze(0)
    # csF3 = wct.transform(cF3, sF3, csF, args.alpha)
    # Im3 = wct.d3(csF3)
    # # vutils.save_image(Im3.data.cpu().float(), os.path.join(args.outf, imname+'_de3.jpg'))
    #
    # sF2 = wct.e2(styleImg)
    # cF2 = wct.e2(Im3)
    # sF2 = sF2.data.cpu().squeeze(0)
    # cF2 = cF2.data.cpu().squeeze(0)
    # csF2 = wct.transform(cF2, sF2, csF, args.alpha)
    # Im2 = wct.d2(csF2)
    # # vutils.save_image(Im2.data.cpu().float(), os.path.join(args.outf, imname+'_de2.jpg'))
    #
    # sF1 = wct.e1(styleImg)
    # cF1 = wct.e1(Im2)
    # sF1 = sF1.data.cpu().squeeze(0)
    # cF1 = cF1.data.cpu().squeeze(0)
    # csF1 = wct.transform(cF1, sF1, csF, args.alpha)
    # Im1 = wct.d1(csF1)
    # # save_image has this wired design to pad images with 4 pixels at default.
    # vutils.save_image(Im1.data.cpu().float(), os.path.join(args.outf, imname+'_de1.jpg'))
    return


avgTime = 0
cImg = torch.Tensor()
sImg = torch.Tensor()
csF = torch.Tensor()
csF = Variable(csF)
if args.cuda:
    cImg = cImg.cuda(args.gpu)
    sImg = sImg.cuda(args.gpu)
    csF = csF.cuda(args.gpu)
    wct.cuda(args.gpu)
for i, (contentImg, styleImg, imname) in enumerate(loader):
    imname = imname[0].split('.')[0]
    print('Transferring ' + imname)
    if args.cuda:
        contentImg = contentImg.cuda(args.gpu)
        styleImg = styleImg.cuda(args.gpu)
    cImg = Variable(contentImg, volatile=True)
    sImg = Variable(styleImg, volatile=True)
    start_time = time.time()
    # WCT Style Transfer
    styleTransfer(cImg, sImg, imname, csF)
    end_time = time.time()
    print('Elapsed time is: %f' % (end_time - start_time))
    avgTime += (end_time - start_time)

print('Processed %d images. Averaged time is %f' % ((i + 1), avgTime / (i + 1)))