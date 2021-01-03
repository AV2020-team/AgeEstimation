import random
import imgaug.augmenters as iaa
from imgaug import parameters as iap  # per parametro gaussiano
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image
import numpy as np
import cv2


random_mirror = True


def ShearX(img, v):  # [-15, 15]
    assert -15 <= v <= 15
    if random_mirror and random.random() > 0.5:
        v = -v
    aug = iaa.ShearX(v)
    img = aug(images=[img])[0]
    return img


def ShearY(img, v):  # [-15, 15]
    assert -15 <= v <= 15
    if random_mirror and random.random() > 0.5:
        v = -v
    aug = iaa.ShearY(v)
    img = aug(images=[img])[0]
    return img


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    aug = iaa.Rotate(v)
    img = aug(images=[img])[0]
    return img


def AutoContrast(img, _):
    aug = iaa.pillike.Autocontrast()
    img = aug(images=[img])[0]
    return img


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    aug = iaa.Fliplr(1)
    img = aug(images=[img])[0]
    return img


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    aug = iaa.pillike.EnhanceBrightness(factor=v)
    img = aug(images=[img])[0]
    return img


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0.1, 0.15]
    assert 0.05 <= v <= 0.15
    aug = iaa.Cutout(nb_iterations=(1, 3), size=v, squared=False)
    img = aug(images=[img])[0]
    return img


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)
    return f


def Crop(img, v):  # [0, 0.4]
    assert 0.1 <= v <= 0.4
    aug = iaa.Crop(percent=(iap.Uniform(0, v), iap.Uniform(0, v), iap.Uniform(0, v), iap.Uniform(0, v)))
    img = aug(images=[img])[0]
    return img


def JPEGCompression(img, v):  # [0, 70]
    assert 0 <= v <= 70
    aug = iaa.JpegCompression(compression=v)
    img = aug(images=[img])[0]
    return img


def SaltAndPepper(img, v):  # [0, 0.05]
    assert 0 <= v <= 0.05
    aug = iaa.SaltAndPepper(v)
    img = aug(images=[img])[0]
    return img


def GaussianNoise(img, v):  # [1, 2]
    assert 1 <= v <= 2
    aug = iaa.imgcorruptlike.GaussianNoise(severity=v)
    img = aug(images=[img])[0]
    return img


def MotionBlur(img, v):  # [1.0, 2]
    v = round(v)
    assert 1 <= v <= 5
    aug = iaa.imgcorruptlike.MotionBlur(severity=v)
    img = aug(images=[img])[0]
    return img


def ZoomBlur(img, v):  # [1, 5]
    v = round(v)
    assert 1 <= v <= 5
    aug = iaa.imgcorruptlike.ZoomBlur(severity=v)
    img = aug(images=[img])[0]
    return img


def GaussianBlur(img, v):  # [0, 5]
    assert 0 <= v <= 5
    aug = iaa.GaussianBlur(sigma=v)
    img = aug(images=[img])[0]
    return img


def augment_list(for_autoaug=True):  # 16 oeprations and their ranges
    l = [
        (ShearX, -15, 15),  # 0
        (ShearY, -15, 15),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0.05, 0.15),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
        (Flip, 0, 1),  # 16
        (Crop, 0.1, 0.4),  # 17
        (JPEGCompression, 0, 70),  # 18
        (SaltAndPepper, 0, 0.05),  # 19
        (GaussianNoise, 1, 2),  # 20
        (GaussianBlur, 0, 5),  # 21
        (MotionBlur, 1, 5),  # 22
        (ZoomBlur, 1, 5)  # 23
    ]
    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)


def img_to_min_size(img, shape=(32,32)):
    if img.shape[1] < shape[0]:  # reshape width
        img = cv2.resize(img, (shape[0], img.shape[0]))
    if img.shape[0] < shape[1]:  # reshape height
        img = cv2.resize(img, (img.shape[1], shape[1]))

    return img


def apply_policy(policy, img):
    for xform in policy:
        assert len(xform) == 3
        name, probability, level = xform
        if random.random() < probability:
            img = img_to_min_size(img, (32, 32))
            img = apply_augment(img, name, level)
    return img
