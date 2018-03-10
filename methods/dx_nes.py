import numpy as np
from scipy.linalg import expm


class DX_NES():
    def __init__(self, dim, func, lam, init_m, init_sigma):
        self.dim = dim
        self.func = func
        self.lam = lam
        self.g = 0
        self.P_sigma = np.array([0] * dim)
        self.eps = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        self.m = [init_m] * dim
        self.sigma = init_sigma
        self.B = np.eye(dim)
        self.alpha = self.calc_alpha(dim, lam)
        self.population = []
        self.eval_count = 0
        self.best_eval = float('inf')

    def sample_population(self):
        self.population.clear()
        append = self.population.append
        for i in range(self.lam // 2):
            z = np.random.normal(size=self.dim)
            z2 = -z
            x = self.m + self.sigma * np.dot(self.B, z)
            x2 = self.m + self.sigma * np.dot(self.B, z2)
            append((x, z, self.func(x)))
            append((x2, z2, self.func(x2)))
        self.eval_count = self.eval_count + self.lam

    def calc_alpha(self, dim, lam):
        alpha = 1
        alpha_old = 0.001
        while np.abs(alpha - alpha_old) / alpha > 1e-5:
            alpha_old = alpha
            alpha = alpha - ((1 + alpha ** 2) * np.exp(alpha ** 2 / 2) / 0.24 - 10 - dim) / (
                (alpha ** 3 + 3 * alpha) * np.exp(alpha ** 2 / 2) / 0.24)
        return alpha * min(1.0, np.sqrt(lam / dim))

    def do_oneiteration(self):
        self.sample_population()
        self.population.sort(key=(lambda p: p[2]))
        _w_rank = np.array([np.maximum(0, np.log(self.lam / 2 + 1) - np.log(i + 1)) for i in range(self.lam)])
        w_rank_sum = np.sum(_w_rank)
        w_rank = _w_rank / w_rank_sum - 1/ self.lam
        _w_dist = np.array([np.exp(self.alpha * np.linalg.norm(p[1])) for p in self.population])
        w_sum = np.dot(w_rank, _w_dist)
        w_dist = np.array([_w_dist[i] * w_rank[i] / w_sum - 1 / self.lam for i in range(self.lam)])
        mu_eff = 1 / np.linalg.norm(w_rank)
        z = np.array([p[1] for p in self.population])
        c_sigma = (mu_eff + 2) / (2 * np.log(self.dim + 1) * (self.dim + mu_eff + 5))
        wz = np.sum([w_rank[i] * z[i] for i in range(self.lam)], axis=0)
        self.P_sigma = (1 - c_sigma) * self.P_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * wz

        eta_m = 1.0
        norm_P = np.linalg.norm(self.P_sigma)
        if norm_P >= self.eps:
            w = w_dist
            eta_sigma = 1.0
            eta_B = np.tanh(
                (self.lam + 0.5 * self.dim + 10) / (6 * (0.1 * self.dim ** 2 + 3 * self.dim + 6))) * np.minimum(1,
                                                                                                                np.sqrt(
                                                                                                                    self.lam / self.dim))
        elif norm_P >= 0.1 * self.eps:
            w = w_rank
            eta_sigma = np.tanh((0.024 * self.lam + 0.7 * self.dim + 20) / (self.lam + 12))
            eta_B = np.tanh(self.lam / (2 * (self.dim ** 2 + 6 * self.dim)))
        else:
            w = w_rank
            eta_sigma = 2 * np.tanh((0.025 * self.lam + 0.75 * self.dim + 10) / (self.lam + 4))
            eta_B = np.tanh(self.lam / (2 * (self.dim ** 2 + 6 * self.dim)))

        G_delta = np.sum([w[i] * z[i] for i in range(self.lam)], axis=0)
        G_M = np.sum(np.array(
            [w[i] * (np.dot(np.array([z[i]]).T, np.array([z[i]])) - np.eye(len(z[i]))) for i in range(self.lam)]),
            axis=0)
        G_sigma = np.trace(G_M) / self.dim
        G_B = G_M - G_sigma * np.eye(self.dim)
        self.m = self.m + eta_m * self.sigma * np.dot(self.B, G_delta)
        self.sigma = self.sigma * np.exp(eta_sigma * G_sigma / 2)
        self.B = np.dot(self.B, expm(eta_B * G_B / 2))
        self.get_besteval()

    def get_besteval(self):
        self.population.sort(key=(lambda p: p[2]))
        self.best_eval = self.population[0][2]
        return self.best_eval
