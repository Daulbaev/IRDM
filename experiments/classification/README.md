This subfolder is based on https://github.com/amirgholami/anode and https://github.com/juliagusak/neural-ode-norm

### How to run: 
Arguments in braces should be specified.
```bash
python3 ./train_cifar.py --data_root {folder with CIFAR-10} --manual_seed {int} --save {save folder} --network resnet4 --batch_size 128 --lr 5e-3 --method dopri5 --n_nodes 8 --atol 1e-3 --rtol 1e-3 --inplanes 64 --normalization_resblock BN --param_normalization_odeblock WN --normalization_odeblock NormFree --normalization_bn1 BN --activation_resblock ReLU --activation_odeblock ReLU --odeint_type adjoint_chebyshev
```