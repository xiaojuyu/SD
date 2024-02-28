import imp
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate


class Loader:
    def __init__(self, dataset, device, batch_size=1,num_workers=0,pin_memory=True,isNeuromorphic=True,isCaltech101=False):
        self.device = device
        split_indices = list(range(len(dataset)))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        if isNeuromorphic:
            self.loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler,num_workers=num_workers, pin_memory=pin_memory,collate_fn=collate_events)
        else:
            if isCaltech101:
                self.loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler,num_workers=num_workers, pin_memory=pin_memory,collate_fn=collate_images)
            else:
                self.loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler,num_workers=num_workers, pin_memory=pin_memory)


    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)


def collate_events(data):
    labels = []
    events = []
    for i, d in enumerate(data):
        labels.append(d[1])
        ev = np.concatenate([d[0], i*np.ones((len(d[0]),1), dtype=np.float32)],1)
        events.append(ev)
    events = torch.from_numpy(np.concatenate(events,0))
    labels = default_collate(labels)
    return events, labels

def collate_images(data):
    labels = []
    events = []
    for i, d in enumerate(data):
        labels.append(d[1])
        if d[0].shape[0]==1:
            # TODO no debug
            img = d.numpy().repeat(3,1,1)
            events.append(img)
            
        else:
            events.append(d[0].numpy())
    # events = torch.from_numpy(np.concatenate(events,0))
    # events = torch.cat(torch.tensor(events), dim=0)
    events = np.array(events)
    events = torch.tensor(events)
    labels = default_collate(labels)
    return events, labels