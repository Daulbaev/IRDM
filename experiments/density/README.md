This code is based on https://github.com/rtqichen/ffjord

### For toy datasets:
```
python ./train_toy_new.py --data {choose from {circles, pinwheel, moons, 2spirals}} --manual_seed {int} --n_nodes {int} --solver dopri5 --odeint_type adjoint_chebyshev 
```

### For tabular datasets:
The data was borrowed from https://github.com/gpapamak/maf.
It is also available here: https://yadi.sk/d/fbUIv8cRyV_vQA?w=1.
Download it and place into `./data` folder.
```
python3 ./train_tabular.py --n_nodes {int} --manual_seed {int} --data miniboone --nhidden --hdim_factor 20 --num_blocks 1 --nonlinearity softplus --batch_size 1000 --test_batch_size 50 --lr 1e-3 --solver dopri5 --odeint_type adjoint_chebyshev
```

### For vae flows:
The data was borrowed from https://github.com/riannevdberg/sylvester-flows.
It is also available here: https://yadi.sk/d/fbUIv8cRyV_vQA?w=1.
Download it and place into `./data` folder.
```
python3 ./train_vae_flow.py --manual_seed {int} --n_nodes {int} --odeint_type adjoint_chebyshev --data freyfaces --solver dopri5 --flow cnf_rank --rank 20 --dims 512-512 --nonlinearity softplus --learning_rate 0.001 --batch_size 50 --early_stopping 100 --num_blocks 2 --num_flows 7
# or
python3 ./train_vae_flow.py --manual_seed {int} --n_nodes {int} --odeint_type adjoint_chebyshev --data caltech --solver dopri5 --flow cnf_rank --rank 20 --dims 2048 --nonlinearity tanh --learning_rate 0.001 --batch_size 50 --early_stopping 100 --num_blocks 1 --num_flows 7
```