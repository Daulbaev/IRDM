# About

Code for reproducing the experiments in the paper:
> Daulbaev, T., Katrutsa, A., Markeeva, L., Gusak, J., Cichocki, A., & Oseledets, I. (2020). Interpolation Technique to Speed Up Gradients Propagation in Neural ODEs. Advances in Neural Information Processing Systems, 33.
> [[arxiv]](https://arxiv.org/abs/2003.05271) [[bibtex]](https://scholar.googleusercontent.com/scholar.bib?q=info:pRV1HO0t_YYJ:scholar.google.com/&output=citation&scisdr=CgWvWYqaEPD_3T0aF_Y:AAGBfm0AAAAAX8wfD_Y4Ya2WUusJ2ZIm1BUcz2gdrE5S&scisig=AAGBfm0AAAAAX8wfD9hsDNUadyTkwE-UuiXNGTeD19jt&scisf=4&ct=citation&cd=-1&hl=ru)

This code is based on the following repositories:
* https://github.com/rtqichen/torchdiffeq
* https://github.com/rtqichen/ffjord
* https://github.com/amirgholami/anode 
* https://github.com/juliagusak/neural-ode-norm

### Installation

```bash 
python3 setup.py install
```

### Usage

To apply IRDM, one has to create odeint_chebyshev function, which has the same interface as odeint_adjoint as follows.

```python3
from interpolated_torchdiffeq import odeint_chebyshev_func
from functools import partial 

n_nodes = 10  # if you want 10 grid points
odeint_chebyshev = partial(odeint_chebyshev_func, n_nodes=n_nodes)
# ... Then use odeint_chebyshev as odeint in torchdiffeq
```

Code for experiments is located in subfolders of `./experiments`. Please, see README files in these subfolders for instructions.
For logging, we use [Weights & Biases](wandb.com). You can specify `--wandb_name` to use wandb logging in all scripts.

Feel free to ask questions via authors' emails.
