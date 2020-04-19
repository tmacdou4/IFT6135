from q2_solution import *
import numpy as np
import torch
import matplotlib.pyplot as plt

dist = vf_wasserstein_distance
#dist = vf_squared_hellinger

thetas = np.arange(0,2.1,0.1)
dists = ["Wass", "Hell"]
for d in dists:
    est = []
    for t in thetas:
        p_sampler = iter(q2_sampler.distribution1(0))
        q_sampler = iter(q2_sampler.distribution1(t))
        c = q2_model.Critic()
        optimizer = torch.optim.SGD(c.parameters(), lr=1e-4)
        for _ in range(1000):
            c.zero_grad()
            p_sample = torch.Tensor(next(p_sampler))
            q_sample = torch.Tensor(next(q_sampler))
            if d == "Wass":
                out = -(vf_wasserstein_distance(p_sample, q_sample, c) - lp_reg(p_sample, q_sample, c))
            elif d == "Hell":
                out = -vf_squared_hellinger(p_sample, q_sample, c)
            out.backward()
            optimizer.step()
        c.eval()
        p_sample = torch.Tensor(next(p_sampler))
        q_sample = torch.Tensor(next(q_sampler))
        if d == "Wass":
            est.append(-(vf_wasserstein_distance(p_sample, q_sample, c) - lp_reg(p_sample, q_sample, c)))
        elif d == "Hell":
            est.append(-vf_squared_hellinger(p_sample, q_sample, c))

    plt.plot(thetas, est)
    plt.xlabel("Value of Theta")
    if d == "Wass":
        plt.title("Estimate of Earth-Mover distance for values of theta")
        plt.ylabel("Estimate of Earth-Mover distance")
    elif d == "Hell":
        plt.title("Estimate of Squared Hellinger distance for values of theta")
        plt.ylabel("Estimate of Squared Hellinger Distance")

    plt.savefig(d + ".png")
    plt.clf()
