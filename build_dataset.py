import os
from PIL import Image
from torchvision import transforms


def build_celeba(src='../data/img_align_celeba/',
                 dst='../data/img_align_celeba_resize/images/',
                 image_size=64):
    os.makedirs(dst, exist_ok=True)

    image_list = os.listdir(src)

    trans = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size)
    ])

    for fname in image_list:
        img = Image.open(src + fname)
        img = trans(img)
        img.save(dst + fname)
        print(dst + fname)


if __name__ == '__main__':
    build_celeba()
