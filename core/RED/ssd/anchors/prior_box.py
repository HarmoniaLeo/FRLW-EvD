from itertools import product

import torch
from math import sqrt


class PriorBox:
    def __init__(self, cfg):
        self.H = cfg.H
        self.W = cfg.W
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps_x = prior_config.FEATURE_MAPS_x
        self.feature_maps_y = prior_config.FEATURE_MAPS_y
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        self.strides_x = prior_config.STRIDES_x
        self.strides_y = prior_config.STRIDES_y
        self.aspect_ratios = prior_config.ASPECT_RATIOS
        self.clip = prior_config.CLIP

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, (f_x, f_y) in enumerate(zip(self.feature_maps_x,self.feature_maps_y)):
            scale_x = self.W / self.strides_x[k]
            scale_y = self.H / self.strides_y[k]
            for i, j in product(range(f_y), range(f_x)):
                # unit center x,y
                cx = (j + 0.5) / scale_x
                cy = (i + 0.5) / scale_y

                # small sized square box
                size = self.min_sizes[k]
                h = size / self.H
                w = size / self.W
                priors.append([cx, cy, w, h])

                # big sized square box
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = size / self.H
                w = size / self.W
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size = self.min_sizes[k]
                h = size / self.H
                w = size / self.W
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
