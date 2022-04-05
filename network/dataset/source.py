import numpy as np
import io, requests 

def get_data_from_url():
    # get urls
    test_dat_url = requests.get("https://github.com/more-th4n-life/COMP5329-ASM1/blob/main/test_data.npy?raw=true")
    test_lab_url = requests.get("https://github.com/more-th4n-life/COMP5329-ASM1/blob/main/test_label.npy?raw=true")
    train_dat_url = requests.get("https://github.com/more-th4n-life/COMP5329-ASM1/blob/main/train_data.npy?raw=true")
    train_lab_url = requests.get("https://github.com/more-th4n-life/COMP5329-ASM1/blob/main/train_label.npy?raw=true")

    # load data
    test_dat =  np.load(io.BytesIO(test_dat_url.content))
    test_lab = np.load(io.BytesIO(test_lab_url.content))
    train_dat = np.load(io.BytesIO(train_dat_url.content))
    train_lab = np.load(io.BytesIO(train_lab_url.content))

    return train_dat, train_lab, test_dat, test_lab

def get_data_from_file():
    ld = lambda fn: np.load(fn)
    root = 'network/data/' 
    return \
    ld(root + "train_data.npy"), \
    ld(root + "train_label.npy"), \
    ld(root + "test_data.npy"), \
    ld(root + "test_label.npy")
