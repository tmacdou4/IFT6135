"""
Template for Question 3.
@author: Samuel Lavoie
"""
import torch
from q3_sampler import svhn_sampler
from q3_model import Critic, Generator
from torch import optim

def lp_reg(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """

    eps = torch.rand(x.size(0)).unsqueeze(1)
    x_hat = eps*x+(1-eps)*y
    x_hat.requires_grad = True

    f_x_hat = critic(x_hat)

    grad = torch.autograd.grad(f_x_hat, x_hat, grad_outputs=torch.ones_like(f_x_hat), retain_graph=True, create_graph=True, only_inputs=True)[0]

    norm = torch.norm(grad, 2, dim=-1)

    relu = torch.nn.functional.relu(norm-1)

    out = relu ** 2

    return out.mean()

def vf_wasserstein_distance(p, q, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    critic_x = critic(p)
    critic_y = critic(q)

    E_P = (critic_x).mean()
    E_Q = (critic_y).mean()

    out = E_P - E_Q

    return out



if __name__ == '__main__':
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_iter = 50000 # N training iterations
    n_critic_updates = 5 # N critic updates per generator update
    lp_coeff = 10 # Lipschitz penalty coefficient
    train_batch_size = 64
    test_batch_size = 64
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100

    train_loader, valid_loader, test_loader = svhn_sampler(data_root, train_batch_size, test_batch_size)

    generator = Generator(z_dim=z_dim).to(device)
    critic = Critic().to(device)

    optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

    # COMPLETE TRAINING PROCEDURE

    # COMPLETE QUALITATIVE EVALUATION
