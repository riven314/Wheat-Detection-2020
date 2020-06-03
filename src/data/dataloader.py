from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler


def get_dataloaders(bs, workers_n = 4):
    train_ds = None
    valid_ds = None
    
    train_dl = DataLoader(
        train_ds,
        batch_size = bs,
        num_workers = workers_n,
        shuffle = True,
        sampler = RandomSampler(train_ds),
        pin_memory = True,
        collate_fn = collate_fn,
        drop_last = True,
    )
    valid_dl = DataLoader(
        valid_ds, 
        batch_size = bs,
        num_workers = workers_n,
        shuffle = False,
        sampler = SequentialSampler(valid_ds),
        pin_memory = True,
        collate_fn = collate_fn,
    )
    return train_dl, valid_dl
    

def collate_fn(batch):
    return tuple(zip(*batch))