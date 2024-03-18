import torch
import time
import numpy as np

class fetcher:
    def __init__(self,events,shape,labels,timestamps,filenames,events_window,event_volume_bins,infer_time,to_volume):
        self.events_window_abin = infer_time
        self.events_window = events_window
        self.event_volume_bins = event_volume_bins
        self.shape = shape
        self.memory = None
        self.total_time = int(timestamps[0,1] - timestamps[0,0])
        self.iter = 0
        #self.events = torch.from_numpy(events).cuda(non_blocking=True)
        self.events = events
        self.labels = labels
        self.timestamps = timestamps
        self.filenames = filenames
        self.finish = False
        self.to_volume = to_volume

    def getLabels(self, timestamps):
        max_labels = 80
        tol = self.events_window_abin/2 - 1
        padded_labels = torch.zeros((len(self.timestamps), max_labels, self.labels.shape[1] - 1)).float().to(self.labels.device)
        for batch in range(len(self.timestamps)):
            timestamp = timestamps[batch]
            labels_ = self.labels[(self.labels[:, 0] == batch)&(self.labels[:, 6] + tol >= timestamp)&(self.labels[:, 6] - tol <= timestamp)]
            if len(labels_) == 0:
                return None
            assert max_labels >= len(labels_)
            padded_labels[batch, range(len(labels_))] = labels_[:, 1:].float()
        return padded_labels

    def fetch(self):
        if self.iter == 0:
            events_buf = self.events[self.events[...,3] < self.events_window]
            self.iter += self.events_window
            if self.iter >= self.total_time:
                self.finish = True
        else:
            events_buf = self.events[(self.events[...,3] < self.iter + self.events_window_abin)&(self.events[...,3] >= self.iter)]
            self.iter += self.events_window_abin
            if self.iter >= self.total_time:
                self.finish = True
        # events = []
        # for i in range(len(self.timestamps)):
        #     events.append(events_buf[events_buf[..., 0] == i][-1200000:])
        # events = torch.from_numpy(np.concatenate(events)).cuda(non_blocking=True)
        events = torch.from_numpy(events_buf).cuda(non_blocking=True)
        
        start = time.time()
        volume, self.memory = self.to_volume(events, len(self.timestamps), self.shape, self.iter, self.memory, self.events_window, self.event_volume_bins, self.events_window_abin)
        torch.cuda.synchronize()
        represent_time = time.time() - start
        if self.events_window == 60000000:
            timestamps = self.timestamps[..., 1]
        else:
            timestamps = self.timestamps[..., 0] + self.iter
        labels = self.getLabels(timestamps)
        #print(labels,timestamps)
        return volume, labels, timestamps, self.filenames, represent_time

class fetcherTrain(fetcher):
    def getLabels(self, timestamps):
        labels = super().getLabels(timestamps)
        if not (labels is None):
            return torch.cat([labels[:,:,4:5],labels[:,:,:4]],dim=-1)
        else:
            return None

class fetcherVal(fetcher):
    pass
