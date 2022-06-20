import os
import numpy as np
import torch
from torchvision import utils, transforms
import torchvision.transforms.functional as F

from escape_rooms.wrapper import EscapeRoomWrapper
from escape_rooms.play_wrapper import PlayWrapper
from escape_rooms.level_generators.crafter_generator import CrafterLevelGenerator
from escape_rooms.level_generators.human_generator import HumanDataGenerator


class SquarePad:
    def __call__(self, image):
        image_size = image.size()[1:]
        max_wh = max(image_size)
        p_top, p_left = [(max_wh - s) // 2 for s in image_size]
        p_bottom, p_right = [max_wh - (s + pad) for s, pad in zip(image_size, [p_top, p_left])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


if __name__ == '__main__':
    # env = EscapeRoomWrapper(30, 30)
    env = EscapeRoomWrapper(player_observer_type='GlobalSprite2D', level_generator_cls=HumanDataGenerator)
    # env = PlayWrapper(env, seed=100)

    save_dir = 'figures/'
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    seeds = list(range(0, 101))
    images = []
    for s in seeds:
        if s == 1:
            continue
        images.append(np.copy(env.reset(seed=s)))

    sqpad = SquarePad()
    tensor_images = [sqpad(torch.tensor(img, dtype=torch.float32).permute(0, 2, 1)) / 255 for img in images]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(100, 100)),
        transforms.ToTensor()
    ])

    downscaled_imgs = [transform(t) for t in tensor_images]

    all_tracks = utils.make_grid(downscaled_imgs, nrow=10)
    utils.save_image(all_tracks, os.path.join(save_dir, 'escape_rooms.png'))
