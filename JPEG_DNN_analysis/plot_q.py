from matplotlib import pyplot as plt
import torch 
import torch.nn.functional as  f
# alpha =  torch.tensor([0.00001,0.0001, 0.001,  0.01, 0.1, 1 , 2, 3, 4, 5, 6, 7, 8, 9 , 10 , 100, 10000000])
# alpha =  torch.arange(0.01, 0.1, 0.01)
alpha =  torch.arange(0.1, 1, 0.1)
# alpha =  torch.arange(20, 50, 1)
# Reconstructed space after qantization

q = 1e-5
x_hat = q*torch.arange(-128, 129, 1)
# Original Space
x = torch.arange(-128, 127.01, 0.01)


def Random_Q(x_instant, alpha):   
    p = f.softmax(-alpha * (x_instant - x_hat)**2, dim=0)
    y = torch.sum(p * x_hat)
    return y

y = torch.empty_like(x)
for alp in alpha:
    for idx, x_instant in enumerate(x):
        y[idx] = Random_Q(x_instant, alp)
    plt.figure()
    plt.plot(x, y)
    plt.ylim([-10*q, 10*q])
    plt.xlim([-10*q, 10*q])
    plt.grid(True)
    plt.savefig("q_{}_{:.6f}.png".format(q, alp), dpi=500)
    

