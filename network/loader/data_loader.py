import queue
from multiprocessing import Queue, Process
from itertools import cycle
from socket import timeout
from loader.loader import Loader, collate
import numpy as np
import time

def worker_func(data, idx_queue, out_queue):
    
    #reads idx from idx and adds data to out queue
    
    while True:

        try:
            idx = idx_queue.get(timeout=0)
        except queue.Empty: continue

        if idx is None: break
        out_queue.put((idx, data[idx]))


class DataLoader(Loader):
    """
    Tried to implement my own DataLoader similar to PyTorch 

    Just added random sampling which which would be more appropriate for batch training (See sampler.py)
    """
    def __init__(self, data, batch_size=20, n_workers=1, prefetch=2, fn=collate, sampler=None):
        """
        Default collate function used, and batch size = 20
        Increasing n_workers can provide functionality of loading another batch at same time a current is loading
            
            As our data is not so large it may be unnecessary
        """
        super().__init__(data, batch_size, fn)

        self.n_workers, self.prefetch = n_workers, prefetch

        self.idx_queues, self.out_queue = [], Queue()

        self.workers = []
        self.work_cycle = cycle(range(n_workers))
        self.cache = {}
        self.sampler = sampler
        self.prefetch_idx = next(sampler) if sampler else 0

        for _ in range(n_workers):
            idx_queue = Queue()
            work = Process(target=worker_func, args=(data, idx_queue, self.out_queue))
            work.daemon = True
            work.start()
            self.workers.append(work)
            self.idx_queues.append(idx_queue)
        
        self.prefetch_batch()

    def prefetch_batch(self):
        """
        Fetches next batch
        """
        while (not len(self.data) <= self.prefetch_idx and self.prefetch_idx < self.idx + 2*self.batch*self.n_workers):
            # not 2 batches ahead and not end of data
            idx = self.prefetch_idx
            self.prefetch_idx = next(self.sampler) if self.sampler else self.prefetch_idx + 1

            self.idx_queues[next(self.work_cycle)].put(idx)

    def get(self):
        self.prefetch_batch()
        if not self.idx in self.cache:
            
            while True:
                try: (idx, data) = self.out_queue.get(timeout=0)
                except queue.Empty: continue
            
                if not self.idx == idx:
                    self.cache[idx] = data
                else:
                    it = data
                    break
                
        else:
            it = self.cache[self.idx]
            del self.cache[self.idx]

        self.idx = next(self.sampler) if self.sampler else self.idx + 1
        return it

    def __iter__(self):
        if self.sampler:
            self.idx = next(self.sampler)
            self.prefetch_idx = next(self.sampler)
        else:
            self.idx, self.prefetch_idx = 0, 0

        self.cache = {}
        self.prefetch_batch()
        return self

    def __del__(self):
        end = None
        try:
            for i, work in enumerate(self.workers):
                self.idx_queues[i].put(end)
                work.join(timeout=5)
            for queue in self.idx_queues:
                queue.cancel_join_thread()
                queue.close()
            self.out_queue.cancel_join_thread()
            self.out_queue.close()

        finally:
            [work.terminate() for work in self.workers if work.is_alive()]
    

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.d = list(zip(x, y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.d[idx]


if __name__ == "__main__":
    ds = Dataset(1024)
    dl = DataLoader(ds, n_workers=4, batch_size=64)

    x, y = next(dl)

    print(x.shape)
    print(y.shape)