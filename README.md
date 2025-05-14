# tknn-data-shapley

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
from topshap.topt import shapley_tknn_expand
from topshap.naive import shapley_bf

D = [
    (np.array([0.5]), 1),
    (np.array([2.0]), 1),
    (np.array([1.0]), 0)
]
Z_test = [(np.array([0.0]), 1)]

# Run the tknn algorithm
values = shapley_tknn_expand(D, Z_test, K=2, radius=1, kernel_fn=partial(kernel_value, sigma=1), n_clst=2)
print(values)

# Run the brute-force algorithm
values = shapley_bf(D, Z_test, K=2, kernel_fn=partial(kernel_value, sigma=1))
print(values)
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