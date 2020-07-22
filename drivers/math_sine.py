import numpy as np, matplotlib.pyplot as plt

def simple():
    period = 1
    N_draws = 10000

    #t = np.random.uniform(0, period/2, N_draws)
    t = np.linspace(0, period/2, N_draws, endpoint=True)
    ω = 2*np.pi/period
    A = 10

    x = ω*t

    #y = A*np.abs(np.sin(x))
    y = A*np.sin(x)

    plt.close('all')
    n, bins, patches = plt.hist(y, bins=1000, density=True, cumulative=True)

    x_mod = np.linspace(np.min(y), np.max(y), 1000)
    F_x = 2/np.pi * np.arcsin(np.sqrt(x_mod/A))
    plt.plot(x_mod, F_x)

    plt.savefig('temp.png')

if __name__ == '__main__':
    simple()
