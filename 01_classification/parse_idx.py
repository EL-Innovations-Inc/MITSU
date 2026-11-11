"""Turn the idx files fetched by get_data.sh into the .npz files big.py reads.

    ./get_data.sh
    python3 parse_idx.py          # writes data/mnist.npz and data/fashion.npz
    python3 big.py mnist

Each .npz holds Xtr (60000, 784) and Xte (10000, 784) as uint8 pixels, and the
labels ytr, yte as int64.
"""
import gzip, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def images(path):
    with gzip.open(path) as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)


def labels(path):
    with gzip.open(path) as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)


for name in ("mnist", "fashion"):
    p = lambda part: os.path.join(DATA, f"{name}-{part}-ubyte.gz")
    Xtr, ytr = images(p("train-images-idx3")), labels(p("train-labels-idx1"))
    Xte, yte = images(p("t10k-images-idx3")), labels(p("t10k-labels-idx1"))
    assert Xtr.shape == (60000, 784) and Xte.shape == (10000, 784), "unexpected image file"
    assert ytr.shape == (60000,) and yte.shape == (10000,) and ytr.max() == 9, "unexpected label file"
    np.savez_compressed(os.path.join(DATA, f"{name}.npz"), Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print(f"wrote data/{name}.npz  train {Xtr.shape}  test {Xte.shape}")
