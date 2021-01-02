import random
import imgaug.augmenters as iaa
from imgaug import parameters as iap  # per parametro gaussiano
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image
import numpy as np
import cv2


random_mirror = True


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


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
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


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
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0.1, 0.15]
    assert 0.05 <= v <= 0.15
    img = pil_unwrap(img)
    aug = iaa.Cutout(nb_iterations=(1, 3), size=v, squared=False)
    img = aug(images=[img])[0]
    img = pil_wrap(img)
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
    img = pil_unwrap(img)
    aug = iaa.Crop(percent=(iap.Uniform(0, v), iap.Uniform(0, v), iap.Uniform(0, v), iap.Uniform(0, v)))
    img = aug(images=[img])[0]
    img = pil_wrap(img)
    return img


def JPEGCompression(img, v):  # [0, 70]
    assert 0 <= v <= 70
    img = pil_unwrap(img)
    aug = iaa.JpegCompression(compression=v)
    img = aug(images=[img])[0]
    img = pil_wrap(img)
    return img


def SaltAndPepper(img, v):  # [0, 0.05]
    assert 0 <= v <= 0.05
    img = pil_unwrap(img)
    aug = iaa.SaltAndPepper(v)
    img = aug(images=[img])[0]
    img = pil_wrap(img)
    return img


def GaussianNoise(img, v):  # [1, 2]
    assert 1 <= v <= 2
    img = pil_unwrap(img)
    aug = iaa.imgcorruptlike.GaussianNoise(severity=v)
    img = aug(images=[img])[0]
    img = pil_wrap(img)
    return img


def MotionBlur(img, v):  # [1.0, 2]
    v = round(v)
    assert 1 <= v <= 5
    img = pil_unwrap(img)
    aug = iaa.imgcorruptlike.MotionBlur(severity=v)
    img = aug(images=[img])[0]
    img = pil_wrap(img)
    return img


def ZoomBlur(img, v):  # [1, 5]
    v = round(v)
    assert 1 <= v <= 5
    img = pil_unwrap(img)
    aug = iaa.imgcorruptlike.ZoomBlur(severity=v)
    img = aug(images=[img])[0]
    img = pil_wrap(img)
    return img


def GaussianBlur(img, v):  # [0, 5]
    assert 0 <= v <= 5
    img = pil_unwrap(img)
    aug = iaa.GaussianBlur(sigma=v)
    img = aug(images=[img])[0]
    img = pil_wrap(img)
    return img


def augment_list(for_autoaug=True):  # 16 oeprations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
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


def pil_wrap(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.squeeze(img, 2)
    return Image.fromarray(img)


def pil_unwrap(pil_img):
    dim1 = pil_img.size[0]
    dim2 = pil_img.size[1]
    n = len(pil_img.split())
    pic_array = np.array(list(pil_img.getdata()))
    resh_array = np.reshape(pic_array, (dim2, dim1, n))
    pil_unwrapped = resh_array.clip(0, 255).astype(np.uint8)
    if n == 3:
        r, g, b = cv2.split(pil_unwrapped)
        pil_unwrapped = cv2.merge((b, g, r))
    return pil_unwrapped


def img_to_min_size(img, size=(32,32)):
    if img.size[0] < size[0]:
        img = img.resize((size[0], img.size[1]), Image.LANCZOS)
    if img.size[1] < size[1]:
        img = img.resize((img.size[0], size[1]), Image.LANCZOS)

    return img


def apply_policy(policy, img):
    pil_img = pil_wrap(img)

    for xform in policy:
        assert len(xform) == 3
        name, probability, level = xform
        if random.random() < probability:
            pil_img = img_to_min_size(pil_img, (32, 32))
            pil_img = apply_augment(pil_img, name, level)
    pil_img = pil_img.convert('RGB')
    return pil_unwrap(pil_img)