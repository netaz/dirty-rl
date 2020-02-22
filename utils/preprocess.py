"""Atari frame pre-processing utilities."""

import torch
from torchvision import transforms
from .utils import get_device, func_footer


#@func_footer(500, lambda processed_img: transforms.ToPILImage()(processed_img.cpu() / 255.).show())
def pre_process_game_frame(I, n_channels, output_shape):
    """Run a bunch of pre-processing transformations on a game frame.

    The transformations come from Andrej Karpathy.
    Args:
        I: (ndarray) input game frame
        n_channels: (int) number of channels in the output frame
        output_shape: (tuple) the shape of the output frame
    """
    assert n_channels in (1, 3)
    if n_channels == 1:
        t = transforms.Compose([
                            transforms.Lambda(lambda grayscale: grayscale[:, :, 0]),
                            #transforms.Lambda(lambda img: karpathy_transform(img)),
                            transforms.Lambda(lambda resize: resize[::2, ::2]),
                            transforms.Lambda(lambda img: center_crop(img,
                                                                      output_shape[0],
                                                                      output_shape[1])),

        ])
        processed = torch.tensor(t(I), device=get_device(), dtype=torch.int8).unsqueeze(0)
    else:
        t = transforms.Compose([
            transforms.Lambda(lambda resize: resize[::2, ::2]),
            transforms.Lambda(lambda img: center_crop(img,
                                                      output_shape[0],
                                                      output_shape[1])),
            # Convert to pytorch's (c, w, h) channel ordering
            transforms.Lambda(lambda img: img.transpose(2, 0, 1)),
        ])
        processed = torch.tensor(t(I), device=get_device(), dtype=torch.int8)
    return processed


def center_crop(img, new_width, new_height):
    """Code similar to `center_crop` in TorchVision, with changes.

    Changes from TorchVision:
        - Works on 2D and 3D frames, instead of 2D only
        - Works on numpy ndarray frames, instead of PIL Image frames.
    """
    width, height = img.shape[0], img.shape[1]  # Get dimensions

    left = int(round((width - new_width) / 2))
    right = int(round((width + new_width) / 2))
    top = int(round((height - new_height) / 2))
    bottom = int(round((height + new_height) / 2))
    if len(img.shape) == 2:
        return img[left:right, top:bottom]
    if len(img.shape) == 3:
        return img[left:right, top:bottom, :]
    raise ValueError(f"image shape {img.shape} can only have 2 or 3 dimensions")


def karpathy_transform(I):
    """Karpathy's Pong image transform converts 8bit images to binary images"""
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I
