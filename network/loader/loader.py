import numpy as np


def collate(b):
    ret, el = None, b[0]
    if isinstance(el, np.ndarray):
        ret = np.stack(b)
    if isinstance(el, (int, float)):
        ret = np.array(b)
    if isinstance(el, (list, tuple)):
        ret = tuple(collate(e) for e in zip(*b))
    return ret

class Loader:
    def __init__(self, data, batch_size, fn=collate):
        self.idx = 0
        self.data, self.batch, self.fn = data, batch_size, fn

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if len(self.data) <= self.idx:
            raise StopIteration
        batch = range(min(len(self.data) - self.idx, self.batch))
        return self.fn([self.get() for _ in batch])

    def get(self):
        it = self.data[self.idx]
        self.idx += 1
        return it

if __name__ == "__main__":
    data = list(range(10))

    train_loader = Loader(data, batch_size=6)
    for batch in train_loader:
        print(batch)
