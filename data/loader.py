import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


class Loader:
    def __init__(self, dataset, batch_size, num_workers, pin_memory, device, shuffle=True,sampler=None):
        self.device = device
        split_indices = list(range(len(dataset)))
        if sampler is None:
            if shuffle:
                sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
                self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                                        num_workers=num_workers, pin_memory=pin_memory,drop_last = False,
                                                        collate_fn=collate_events)
            else:
                self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                        num_workers=num_workers, pin_memory=pin_memory,drop_last = False,
                                                        collate_fn=collate_events)
        else:
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                                        num_workers=num_workers, pin_memory=pin_memory,drop_last = False,
                                                        collate_fn=collate_events)
                                       
    def __iter__(self):
        for data in self.loader:
            data = [data[0].cuda(non_blocking=True),data[1].cuda(non_blocking=True),data[2],data[3]]
            yield data

    def __len__(self):
        return len(self.loader)

def collate_events(data):
    imgs = []
    timestamps = []
    labels = []
    file_names = []
    for i, d in enumerate(data):
        labels.append(d[1])
        timestamps.append(d[3])
        imgs.append(d[0])
        file_names.append(d[2])
    imgs = torch.from_numpy(np.stack(imgs))
    labels = torch.from_numpy(np.stack(labels))
    timestamps = np.array(timestamps)
    return imgs, labels, file_names, timestamps