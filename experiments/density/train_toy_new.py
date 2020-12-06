import matplotlib

matplotlib.use('Agg')

import argparse
import os
import time
import sys
sys.path.append('../../')

import torch
import torch.optim as optim

import lib.toy_data as toy_data
import lib.utils as utils
import lib.layers.odefunc as odefunc

# +
from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization
from train_misc import build_model_tabular

import numpy as np
import random

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals',
                       'checkerboard', 'rings'],
    type=str, default='pinwheel'
)
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)

# Track quantities
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")
parser.add_argument('--return_inter_points', type=eval, default=False)
parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=100)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--odeint_type', type=str, default='adjoint_chebyshev')
parser.add_argument('--n_nodes', type=int, default=10)
parser.add_argument('--manual_seed', type=int, default=None)
parser.add_argument('--forward_solver', type=str, default='nonspecified',
                    choices=['nonspecified', 'rk4', 'dopri5', 'euler', 'rk2'])
parser.add_argument('--wandb_name', type=str, default='')
args = parser.parse_args()

if args.wandb_name:
    import wandb
    print(wandb.init(project=args.wandb_name))
    wandb.config.update(args)

if args.layer_type == "blend":
    # logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


def get_transforms(model):
    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn


def compute_loss(args, model, batch_size=None):
    if batch_size is None: batch_size = args.batch_size

    # load data
    x = toy_data.inf_train_gen(args.data, batch_size=batch_size)
    x = torch.from_numpy(x).type(torch.float32).to(device)
    zero = torch.zeros(x.shape[0], 1).to(x)

    # transform to z
    z, delta_logp = model(x, zero)

    # compute log q(z)
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)

    logpx = logpz - delta_logp
    loss = -torch.mean(logpx)
    return loss


if __name__ == '__main__':

    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 100000)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, 2, regularization_fns, )[0].to(device)
    if args.spectral_norm: add_spectral_norm(model)
    set_cnf_options(args, model)

    # logger.info(model)
    # logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    nfef_meter = utils.RunningAverageMeter(0.93)
    nfeb_meter = utils.RunningAverageMeter(0.93)
    tt_meter = utils.RunningAverageMeter(0.93)

    end = time.time()
    end_f = time.time()
    end_b = time.time()
    best_loss = float('inf')
    model.train()
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        if args.spectral_norm: spectral_norm_power_iteration(model, 1)
        end_f = time.time()
        loss = compute_loss(args, model)
        ft_meter = time.time() - end_f
        loss_meter.update(loss.item())

        if len(regularization_coeffs) > 0:
            reg_states = get_regularization(model, regularization_coeffs)
            reg_loss = sum(
                reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
            )
            loss = loss + reg_loss

        total_time = count_total_time(model)
        nfe_forward = count_nfe(model)
        end_b = time.time()
        loss.backward()
        bt_meter = time.time() - end_b
        optimizer.step()

        nfe_total = count_nfe(model)
        nfe_backward = nfe_total - nfe_forward
        nfef_meter.update(nfe_forward)
        nfeb_meter.update(nfe_backward)

        time_meter.update(time.time() - end)
        tt_meter.update(total_time)
        end = time.time()
        if args.wandb_name:
            wandb.log({'iteration': itr,
                       'time_meter': time_meter.val,
                       'avg_time_meter': time_meter.avg,
                       'train_loss': loss_meter.val,
                       'avg_train_loss': loss_meter.avg,
                       'nfe_forward': nfef_meter.val,
                       'avg_nfe_forward': nfef_meter.avg,
                       'nfe_backward': nfeb_meter.val,
                       'avg_nfe_backward': nfeb_meter.avg,
                       'cnf_time': tt_meter.val,
                       'avg_cnf_time': tt_meter.avg,
                       'ft_time_meter': ft_meter,
                       'bt_time_meter': bt_meter})

        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                model.eval()
                test_loss = []
                for _ in range(10):
                    test_loss += [compute_loss(args, model, batch_size=args.test_batch_size).item()]
                test_loss = np.mean(test_loss)
                test_nfe = count_nfe(model)
                if args.wandb_name:
                    wandb.log({'test_loss': test_loss, 'test_nfe': test_nfe, 'test_iter': itr})

                if test_loss < best_loss:
                    best_loss = test_loss
                    utils.makedirs(args.save)
                    torch.save({
                        'args': args,
                        'state_dict': model.state_dict(),
                    }, os.path.join(args.save, 'checkpt.pth'))
                    if args.wandb_name:
                        wandb.save(os.path.join(args.save, 'checkpt.pth'))
                model.train()
