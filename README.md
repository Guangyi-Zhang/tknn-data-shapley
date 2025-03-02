# topt-knn-data-shapley

Requirements:

```bash
pip install pytest numpy 
```
<!--
pip install pytest numpy pandas matplotlib scikit-learn torch torchvision gdown # gdown for CelebA
-->

## Examples

```python
import numpy as np
from functools import partial

from topshap.helper import kernel_value
from topshap.topt import shapley_top
from topshap.naive import shapley_bf

D = [
    (np.array([0.5]), 1),
    (np.array([2.0]), 1),
    (np.array([1.0]), 0)
]
Z_test = [(np.array([0.0]), 1)]

# Run the top-t algorithm
top_idx = shapley_top(D, Z_test, kernel_fn=partial(kernel_value, sigma=1), t=1, K=2, n_clst=1)
assert top_idx == [0]

# Run the brute-force algorithm
shapley_values = shapley_bf(D, Z_test, K=2, kernel_fn=partial(kernel_value, sigma=1))
answer = [0.8374, 0.0902, -0.0451]
assert np.allclose(shapley_values, answer, atol=1e-03)
```


## Tests

```bash
py.test -vv -s -k "not test_shapley_top_2dplanes"
```

## Datasets

2dplanes.arff: https://www.openml.org/search?type=data&sort=runs&id=727&status=active

CIFAR10, EMNIST: https://pytorch.org/vision/main/datasets.html

DOTA2: https://archive.ics.uci.edu/dataset/367/dota2+games+results

Covtype: https://archive.ics.uci.edu/dataset/31/covertype

Poker: https://archive.ics.uci.edu/dataset/158/poker+hand