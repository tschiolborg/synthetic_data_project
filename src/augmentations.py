import cv2
import numpy as np


def brightness(img, alpha=None, beta=None):
    """
    Applies brightness contrast augmentation on image
    img: image to apply augmentation on
    alpha: value to multiply image with
    beta: value to add to image
    Return: (img, alpha, beta)
    """

    if alpha is None:
        alpha = np.random.uniform(0.75, 1.25)
    if beta is None:
        beta = np.random.uniform(-120, 120)

    adjusted_image = alpha * (img.astype(np.float32) + beta)
    return np.clip(adjusted_image, a_min=0, a_max=255).astype(np.uint8), alpha, beta


def perspective(tmp, mask, x=None):
    """
    performs a perspective augmentation on image
    tmp: image (template)
    mask: mask of image
    x: x position of template on background
    Return: (tmp, mask, p) , p is constant used in augmentation
    """

    h, w = tmp.shape[0], tmp.shape[1]

    left = x - round(1000 / 2) > 0 if x is not None else bool(np.random.randint(0, 2))

    p_min, p_max = round(w * 0.07), round(w * 0.14)
    p = np.random.randint(p_min, p_max + 1)

    points0 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    points1 = (
        (np.float32([[p, p], [w - 1, 0], [p, h - 1 - p], [w - 1, h - 1]]))
        if left
        else (np.float32([[0, 0], [w - 1 - p, p], [0, h - 1], [w - 1 - p, h - 1 - p]]))
    )

    m = cv2.getPerspectiveTransform(points0, points1)
    tmp = cv2.warpPerspective(tmp, m, (w, h))
    mask = cv2.warpPerspective(mask, m, (w, h))

    return tmp, mask, p


def rotate(tmp, mask, theta=None):
    """
    rotates image and mask
    tmp: image (template) to rotate
    mask: mask of image
    theta: theta angle to rotate
    Return: (tmp, mask, theta)
    """

    if theta is None:
        theta = np.random.uniform(-10, 10)

    # padding to avoid clipping
    pad = max(tmp.shape[:2]) // 2 + 1
    tmp = cv2.copyMakeBorder(tmp, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    # rotate
    h, w = tmp.shape[0], tmp.shape[1]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), theta, scale=1)
    tmp = cv2.warpAffine(tmp, m, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_LINEAR)

    # remove padding
    tmp, mask = remove_padding(tmp, mask)

    return tmp, mask, theta


def scale(tmp, mask, relative_x, factor=None):
    """
    scales image and mask
    tmp: image (template) to scale
    mask: mask of image
    relative_x: relative x position of template on background: [0, 2]
    factor: scale factor
    Return: (tmp, mask, factor)
    """

    if factor is None:
        factor = (0.175 + relative_x) * min(tmp.shape[:2]) / 100
        c = 0
        while min(factor * np.array(tmp.shape[:2])) < 30:
            if c > 100:
                return tmp, mask, 1
            factor = (0.175 + relative_x) * min(tmp.shape[:2]) / 100
            c += 1

    tmp = cv2.resize(tmp, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

    return tmp, mask, factor


def brightness_adjust(tmp, mask, back_region):
    """
    Adjusts brightness of image according to mean of background
    tmp: image (template)
    mask: mask of image
    back_regoin: crop of backgorund where template is placed
    Return: tmp
    """

    back_grey = cv2.cvtColor(back_region, cv2.COLOR_BGR2GRAY)
    mean = back_grey.mean().astype(np.int16)

    tmp = tmp.astype(np.int16)
    tmp[mask > 0] += mean - 170

    return np.clip(tmp, a_min=0, a_max=255).astype(np.uint8)


def uniform_noise(tmp, noise=None):
    """
    Applies uniform noise on image
    tmp: image (template)
    noise: constant used to apply noise
    Return: (tmp, noise)
    """

    if noise is None:
        noise = np.random.randint(-15, 15, size=tmp.shape)

    tmp = tmp + noise

    return np.clip(tmp, a_min=0, a_max=255).astype(np.uint8), noise


def remove_padding(tmp, mask):
    """
    removes padding of image and mask
    tmp: image (template)
    mask: mask if template
    Return: (tmp, mask)
    """

    foregorund = np.argwhere(mask[:, :, 0] > 0)
    y0, x0 = foregorund.min(axis=0)
    y1, x1 = foregorund.max(axis=0) + 1

    tmp = tmp[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]

    return tmp, mask


def blend(tmp, mask, img, y0, x0):
    """
    pastes template into image by alpha blending
    tmp: template (H, W, 3)
    mask: mask of template (H, W, 3)
    img: background image to paste on
    y0: y position (ymin) on img to paste tmp
    x0: x position (xmin) on img to paste tmp
    Return: (blended image)
    """

    y1 = y0 + tmp.shape[0]
    x1 = x0 + tmp.shape[1]

    img_crop = img[y0:y1, x0:x1]
    w = mask.astype(np.float32) / 255

    blended_crop = img_crop.astype(np.float32) * (1 - w) + tmp.astype(np.float32) * w

    blend = img.copy()
    blend[y0:y1, x0:x1] = np.clip(blended_crop, a_min=0, a_max=255).astype(np.uint8)

    return blend
