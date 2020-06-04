import os
from PIL import Image


EXT = ['jpg', 'png', 'jpeg']

def get_img(path):
  imgs = []
  if path.split('.')[-1] in EXT:
    imgs.append(Image.open(path))
  else:
    for i in os.listdir(path):
      imgs.append(Image.open(os.path.join(path, i)))
  return imgs


imgs = get_img(r'D:\graduate_project\sketch-to-art\src\assets\pictures')


def center_crop_save(imgs, out_path):
  for idx, i in enumerate(imgs):
    resize = i.resize((150, 150), Image.ANTIALIAS)
    resize.save(out_path + str(idx+1) + '.png')
    print(f'resize {i.size} to {resize.size}')


if __name__ == '__main__':
  center_crop_save(imgs, r'pictures/')
