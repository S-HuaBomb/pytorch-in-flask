import argparse
import os
import time

import torchvision.utils as vutils
import torch
from torch.autograd import Variable

from Loader import Dataset, one_loader
from util import *


# parser = argparse.ArgumentParser(description='WCT Pytorch')
# parser.add_argument('--contentPath', default='output/contents', help='path to train')
# parser.add_argument('--stylePath', default='stylized', help='path to train')
# parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 4)')
# parser.add_argument('--vgg1', default='models/vgg_conv1_1.t7', help='Path to the VGG conv1_1')
# parser.add_argument('--vgg2', default='models/vgg_conv2_1.t7', help='Path to the VGG conv2_1')
# parser.add_argument('--vgg3', default='models/vgg_conv3_1.t7', help='Path to the VGG conv3_1')
# parser.add_argument('--vgg4', default='models/vgg_conv4_1.t7', help='Path to the VGG conv4_1')
# parser.add_argument('--vgg5', default='models/vgg_conv5_1.t7', help='Path to the VGG conv5_1')
# parser.add_argument('--decoder5', default='models/feature_conv5_1.t7', help='Path to the decoder5')
# parser.add_argument('--decoder4', default='models/feature_conv4_1.t7', help='Path to the decoder4')
# parser.add_argument('--decoder3', default='models/feature_conv3_1.t7', help='Path to the decoder3')
# parser.add_argument('--decoder2', default='models/feature_conv2_1.t7', help='Path to the decoder2')
# parser.add_argument('--decoder1', default='models/feature_conv1_1.t7', help='Path to the decoder1')
# parser.add_argument('--cuda', action='store_true', help='enables cuda')
# parser.add_argument('--batch_size', type=int, default=1, help='batch size')
# parser.add_argument('--fineSize', type=int, default=512,
#                     help='resize image to fineSize x fineSize,leave it to 0 if not resize')
# parser.add_argument('--outf', default='output/stylized', help='folder to output images')
# parser.add_argument('--alpha', type=float, default=0.6, help='hyperparameter to blend wct feature and content feature')
# parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")
# # parser.add_argument('run', default='python.exe -m flask run')

# args = parser.parse_args()


class Args:
  """
  模型路径，和各种参数
  """

  def __init__(self):
    self.vgg1 = 'models/vgg_conv1_1.t7'
    self.vgg2 = 'models/vgg_conv2_1.t7'
    self.vgg3 = 'models/vgg_conv3_1.t7'
    self.vgg4 = 'models/vgg_conv4_1.t7'
    self.vgg5 = 'models/vgg_conv5_1.t7'

    self.decoder5 = 'models/feature_conv5_1.t7'
    self.decoder4 = 'models/feature_conv4_1.t7'
    self.decoder3 = 'models/feature_conv3_1.t7'
    self.decoder2 = 'models/feature_conv2_1.t7'
    self.decoder1 = 'models/feature_conv1_1.t7'

    self.outf = 'output/stylized'

    self.contentPath = 'output/contents'
    self.stylePath = 'stylized'
    self.fineSize = 512

    self.cuda = False
    self.gpu = 0


args = Args()

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


def styleTransfer(contentImg, styleImg, imname, csF, alpha):
  # sF5 = wct.e5(styleImg)  # 编码
  # cF5 = wct.e5(contentImg)
  # sF5 = sF5.data.cpu().squeeze(0)
  # cF5 = cF5.data.cpu().squeeze(0)
  # csF5 = wct.transform(cF5, sF5, csF, alpha)  # wct转换
  # Im5 = wct.d5(csF5)  # 解码
  #
  # sF4 = wct.e4(styleImg)
  # cF4 = wct.e4(Im5)
  # sF4 = sF4.data.cpu().squeeze(0)
  # cF4 = cF4.data.cpu().squeeze(0)
  # csF4 = wct.transform(cF4, sF4, csF, alpha)
  # Im4 = wct.d4(csF4)

  sF3 = wct.e3(styleImg)
  cF3 = wct.e3(contentImg)
  sF3 = sF3.data.cpu().squeeze(0)
  cF3 = cF3.data.cpu().squeeze(0)
  csF3 = wct.transform(cF3, sF3, csF, alpha)
  Im3 = wct.d3(csF3)

  sF2 = wct.e2(styleImg)
  cF2 = wct.e2(Im3)
  sF2 = sF2.data.cpu().squeeze(0)
  cF2 = cF2.data.cpu().squeeze(0)
  csF2 = wct.transform(cF2, sF2, csF, alpha)
  Im2 = wct.d2(csF2)

  sF1 = wct.e1(styleImg)
  cF1 = wct.e1(Im2)
  sF1 = sF1.data.cpu().squeeze(0)
  cF1 = cF1.data.cpu().squeeze(0)
  csF1 = wct.transform(cF1, sF1, csF, alpha)
  Im1 = wct.d1(csF1)
  vutils.save_image(Im1.data.cpu().float(), os.path.join(args.outf, imname))
  return


def get_stylize(contentPath, stylePath, alpha):
  imname = contentPath.split('/')[-1]

  cImg = torch.Tensor()
  sImg = torch.Tensor()
  csF = torch.Tensor()
  csF = Variable(csF)

  contentImg, styleImg = one_loader(contentPath, stylePath, args.fineSize)

  print('Transferring ' + imname)
  if args.cuda:
    contentImg = contentImg.cuda(args.gpu)
    styleImg = styleImg.cuda(args.gpu)
  with torch.no_grad():
    cImg = Variable(contentImg)
    sImg = Variable(styleImg)
  start_time = time.time()
  # WCT Style Transfer
  styleTransfer(cImg, sImg, imname, csF, alpha)
  end_time = time.time()
  print('用时: %f' % (end_time - start_time))


if __name__ == '__main__':
  get_stylize(r'output/contents/in2.jpg', r'styles/2.jpg')
