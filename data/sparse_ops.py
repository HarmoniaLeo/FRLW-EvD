import torch
import numpy as np

def generate_agile_event_volume_cuda(events, B, shape, iter, past_volume = None, events_window = 50000, volume_bins=5, infer_time = 10000):
    H, W = shape

    b, x, y, t, p = events.unbind(-1)

    b, x, y, p = b.long(), x.long(), y.long(), p.long()

    if past_volume is None:
        t_star = (volume_bins * t.float() / events_window)[:,None,None]
        channels = volume_bins
    else:
        t_star = ((t.float() - iter + infer_time) / events_window * volume_bins)[:,None,None]
        channels = 2
    adder = torch.stack([torch.arange(channels),torch.arange(channels)],dim = 1).to(x.device)[None,:,:]   #1, 2, 2
    adder = (1 - torch.abs(adder-t_star)) * torch.stack([p,1 - p],dim=1)[:,None,:]  #n, 2, 2
    adder = torch.where(adder>=0,adder,torch.zeros_like(adder)).view(adder.shape[0], channels * 2) #n, 4

    if past_volume is None:
        img = torch.zeros((B * H * W, volume_bins * 2)).float().to(x.device)
        img.index_add_(0, H * W * b + x + W * y, adder)
        img = img.view(B * H * W, volume_bins, 2, 1)
    else:
        img_new = torch.zeros((B * H * W, 4)).float().to(x.device)
        img_new.index_add_(0, H * W * b + x + W * y, adder)
        img_new = img_new.view(B * H * W, 2, 2, 1)
        img_old = past_volume
        img_old = img_old[:,1:]
        img_old[:,-1] = img_old[:,-1] + img_new[:,0]
        img = torch.cat([img_old,img_new[:,1:]],dim=1)

    img_viewed = img.view((B, H, W, img.shape[1] * 2, 1)).permute(0, 3, 1, 2, 4).contiguous()
    return img_viewed, img

def generate_event_volume_cuda(events, B, shape, iter, memory = None, events_window = 50000, volume_bins=5, infer_time = 10000):
    H, W = shape

    if not (memory is None):
        events = torch.cat([memory,events])
    memory = events[events[:,3] >= iter - events_window + infer_time]

    b, x, y, t, p = events.unbind(-1)

    b, x, y, p = b.long(), x.long(), y.long(), p.long()

    # for i in range(B):
    #     t_b = t[b == i]
    #     if len(t_b) > 0:
    #         t_min = t_b.min()
    #         t_max = t_b.max()
    #         t[b == i] = (t_b - t_min)/(t_max - t_min + 1e-8)

    # t_star = (volume_bins - 1) * t.float()[:,None,None]

    t_star = ((volume_bins - 1) * t.float() / events_window)[:,None,None]
    channels = volume_bins

    adder = torch.stack([torch.arange(channels),torch.arange(channels)],dim = 1).to(x.device)[None,:,:]   #1, 2, 2
    adder = (1 - torch.abs(adder - t_star)) * torch.stack([p,1 - p],dim=1)[:,None,:]  #n, 2, 2
    adder = torch.where(adder>=0,adder,torch.zeros_like(adder)).view(adder.shape[0], channels * 2) #n, 4

    img = torch.zeros((B * H * W, volume_bins * 2)).float().to(x.device)
    img.index_add_(0, H * W * b + x + W * y, adder)
    img = img.view(B * H * W, volume_bins, 2, 1)

    img_viewed = img.view((B, H, W, img.shape[1] * 2, 1)).permute(0, 3, 1, 2, 4).contiguous()
    return img_viewed, memory


def generate_taf_cuda(events, B, shape, iter, past_volume = None, events_window = 50000, volume_bins=5, infer_time = 10000):
    b, x, y, t, c, p, features = events.unbind(-1)

    b, x, y, p, c = b.long(), x.long(), y.long(), p.long(), c.long()
    
    H, W = shape
    C = volume_bins * 2

    feature_map = torch.zeros(B * C * H * W * 2).float().to(events.device)
    feature_map.index_add_(0, b * C * H * W * 2 + c * H * W * 2 + y * W * 2 + x * 2 + p, features.float())

    volume = feature_map.view(B, C, H, W, 2).contiguous()
    volume[:,:,:,:,1] = torch.where(volume[:,:,:,:,1] ==0, torch.zeros_like(volume[:,:,:,:,1]).float() - 1e8, volume[:,:,:,:,1] + 1)
    return volume, None


def generate_event_frame_cuda(events, B, shape, iter, past_volume = None, events_window = 50000, volume_bins=5, infer_time = 10000):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
    coordinates u and v.
    """
    H, W = shape
    b, x, y, t, p = events.unbind(-1)

    b, x, y, p = b.long(), x.long(), y.long(), p.long()

    img = torch.zeros((B * H * W,)).float().to(x.device)

    img.index_add_(0, H * W * b + x + W * y, torch.ones_like(x).float())

    img = torch.where(img > 0, torch.ones_like(img).float() * 255, img)
    img = torch.cat([img, img])

    histogram = img.view((2, B, H, W, 1)).permute(1, 0, 2, 3, 4).contiguous()

    return histogram, None

def sparseToDense(locations,features,shape):
    B, H, W = shape
    C = features.shape[-1]

    b, y, x = locations.unbind(-1)

    b, x, y = b.long(), x.long(), y.long()

    feature_map = torch.zeros(B * H * W, C).float().to(locations.device)
    feature_map.index_add_(0, H * W * b + W * y + x , features)

    return feature_map.view(B,H,W,C)

def denseToSparse(dense_tensor):
    """
    Converts a dense tensor to a sparse vector.

    :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
    :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
    :return features: NumberOfActive x FeatureDimension
    """
    non_zero_indices = torch.nonzero(torch.abs(dense_tensor).sum(axis=-1))
    locations = torch.cat((non_zero_indices[:, 1:], non_zero_indices[:, 0, None]), dim=-1)

    select_indices = non_zero_indices.split(1, dim=1)
    features = torch.squeeze(dense_tensor[select_indices], dim=-2)

    return locations, features