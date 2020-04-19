from q2_solution import *
import numpy as np
import torch
import matplotlib.pyplot as plt

thetas = np.arange(0,2.1,0.1)
dists = ["Wass", "Hell"]
ests_Wass = []
ests_Hell = []

for dist_metric in dists:
    for t in thetas:
        p_sampler = iter(q2_sampler.distribution1(0, 512))
        q_sampler = iter(q2_sampler.distribution1(t, 512))
        c = q2_model.Critic(2)
        optimizer = torch.optim.SGD(c.parameters(), lr=1e-3)
        for _ in range(10000):
            c.zero_grad()
            p_sample = torch.Tensor(next(p_sampler))
            q_sample = torch.Tensor(next(q_sampler))
            if dist_metric == "Wass":
                dist = vf_wasserstein_distance(p_sample, q_sample, c)
                reg = 50 * lp_reg(p_sample, q_sample, c)
                out = -(dist - reg)
            else:
                out = -vf_squared_hellinger(p_sample, q_sample, c)
            out.backward()
            optimizer.step()

        c.eval()
        p_sample = torch.Tensor(next(p_sampler))
        q_sample = torch.Tensor(next(q_sampler))
        if dist_metric == "Wass":
            out = vf_wasserstein_distance(p_sample, q_sample, c)
            ests_Wass.append(out)
        else:
            out = vf_squared_hellinger(p_sample, q_sample, c)
            ests_Hell.append(out)

print(ests_Wass)
print(ests_Hell)
