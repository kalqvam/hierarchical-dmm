import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class MixedEmitter(nn.Module):
    # z_t -> emission parameters for three distribution families
    # x layout: first n_studentt cols  = Student-t features (log returns)
    #           next  n_lognormal cols  = log-normal features (rv, vix)
    #           remaining n_normal cols = Gaussian features (yield curve)
    def __init__(self, z_dim, hidden_dim, n_studentt, n_lognormal, n_normal):
        super().__init__()
        self.n_studentt  = n_studentt
        self.n_lognormal = n_lognormal
        self.n_normal    = n_normal
        self.shared_st = nn.Linear(z_dim, hidden_dim)
        self.shared_ln = nn.Linear(z_dim, hidden_dim)
        self.shared_n  = nn.Linear(z_dim, hidden_dim)
        self.st_mu    = nn.Linear(hidden_dim, n_studentt)
        self.st_sigma = nn.Linear(hidden_dim, n_studentt)
        self.log_df   = nn.Linear(hidden_dim, n_studentt)
        self.ln_mu    = nn.Linear(hidden_dim, n_lognormal)
        self.ln_sigma = nn.Linear(hidden_dim, n_lognormal)
        self.n_mu     = nn.Linear(hidden_dim, n_normal)
        self.n_sigma  = nn.Linear(hidden_dim, n_normal)
        self.softplus = nn.Softplus()

    def forward(self, z_t):
        h_st = torch.tanh(self.shared_st(z_t))
        h_ln = torch.tanh(self.shared_ln(z_t))
        h_n  = torch.tanh(self.shared_n(z_t))
        df       = 2.0 + self.softplus(self.log_df(h_st))
        st_mu    = self.st_mu(h_st)
        st_sigma = self.softplus(self.st_sigma(h_st))
        ln_mu    = self.ln_mu(h_ln)
        ln_sigma = self.softplus(self.ln_sigma(h_ln))
        n_mu     = self.n_mu(h_n)
        n_sigma  = self.softplus(self.n_sigma(h_n))
        return df, st_mu, st_sigma, ln_mu, ln_sigma, n_mu, n_sigma


class GatedTransition(nn.Module):
    # maps z_{t-1} -> parameters of p(z_t | z_{t-1})
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.lin_z_gate               = nn.Linear(z_dim, hidden_dim)
        self.lin_z_gate_reverse       = nn.Linear(hidden_dim, z_dim)
        self.lin_z_transition         = nn.Linear(z_dim, hidden_dim)
        self.lin_z_transition_reverse = nn.Linear(hidden_dim, z_dim)
        self.lin_z_mu                 = nn.Linear(z_dim, z_dim)
        self.lin_z_sigma              = nn.Linear(z_dim, z_dim)
        nn.init.eye_(self.lin_z_mu.weight)
        nn.init.zeros_(self.lin_z_mu.bias)
        self.act      = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmoid  = nn.Sigmoid()

    def forward(self, z_prev):
        g_t = self.act(self.lin_z_gate(z_prev))
        g_t = self.sigmoid(self.lin_z_gate_reverse(g_t))
        h_t = self.act(self.lin_z_transition(z_prev))
        h_t = self.lin_z_transition_reverse(h_t)
        mu_t    = (1 - g_t) * self.lin_z_mu(z_prev) + g_t * h_t
        sigma_t = self.softplus(self.lin_z_sigma(self.act(h_t)))
        return mu_t, sigma_t


class ConditionedGatedTransition(nn.Module):
    # maps (f_{t-1}, s_tau) -> parameters of p(f_t | f_{t-1}, s_tau)
    # s_tau is fixed for K steps; concatenated with f_prev as joint input
    def __init__(self, f_dim, s_dim, hidden_dim):
        super().__init__()
        in_dim = f_dim + s_dim
        self.lin_gate               = nn.Linear(in_dim, hidden_dim)
        self.lin_gate_reverse       = nn.Linear(hidden_dim, f_dim)
        self.lin_transition         = nn.Linear(in_dim, hidden_dim)
        self.lin_transition_reverse = nn.Linear(hidden_dim, f_dim)
        self.lin_z_mu               = nn.Linear(f_dim, f_dim)   # identity baseline on f only
        self.lin_z_sigma            = nn.Linear(f_dim, f_dim)
        nn.init.eye_(self.lin_z_mu.weight)
        nn.init.zeros_(self.lin_z_mu.bias)
        self.act      = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmoid  = nn.Sigmoid()

    def forward(self, f_prev, s_tau):
        inp = torch.cat([f_prev, s_tau], dim=-1)
        g_t = self.act(self.lin_gate(inp))
        g_t = self.sigmoid(self.lin_gate_reverse(g_t))
        h_t = self.act(self.lin_transition(inp))
        h_t = self.lin_transition_reverse(h_t)
        mu_t    = (1 - g_t) * self.lin_z_mu(f_prev) + g_t * h_t
        sigma_t = self.softplus(self.lin_z_sigma(self.act(h_t)))
        return mu_t, sigma_t


class Combiner(nn.Module):
    # maps (z_{t-1}, h_right_t) -> parameters of q(z_t | z_{t-1}, x_{1:T})
    def __init__(self, z_dim, rnn_hidden_dim):
        super().__init__()
        self.lin_mu    = nn.Linear(rnn_hidden_dim, z_dim)
        self.lin_sigma = nn.Linear(rnn_hidden_dim, z_dim)
        self.lin_z     = nn.Linear(z_dim, rnn_hidden_dim)
        self.act      = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_prev, h_right):
        h_combined = 0.5 * (self.act(self.lin_z(z_prev)) + h_right)
        mu_t    = self.lin_mu(h_combined)
        sigma_t = self.softplus(self.lin_sigma(h_combined))
        return mu_t, sigma_t


class FastCombiner(nn.Module):
    # maps (f_{t-1}, s_tau, h_right_t) -> parameters of q(f_t | f_{t-1}, s_tau, x_{1:T})
    # three-way blend: previous fast state, fixed slow state, backward RNN context
    def __init__(self, f_dim, s_dim, rnn_hidden_dim):
        super().__init__()
        self.lin_f     = nn.Linear(f_dim, rnn_hidden_dim)
        self.lin_s     = nn.Linear(s_dim, rnn_hidden_dim)
        self.lin_mu    = nn.Linear(rnn_hidden_dim, f_dim)
        self.lin_sigma = nn.Linear(rnn_hidden_dim, f_dim)
        self.act      = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, f_prev, s_tau, h_right):
        h_combined = (self.act(self.lin_f(f_prev)) + self.act(self.lin_s(s_tau)) + h_right) / 3.0
        mu_t    = self.lin_mu(h_combined)
        sigma_t = self.softplus(self.lin_sigma(h_combined))
        return mu_t, sigma_t


class NestedDMM(nn.Module):
    def __init__(self,
                 n_st_fast, n_ln_fast, n_n_fast,
                 n_st_slow, n_ln_slow, n_n_slow,
                 s_dim, f_dim,
                 hidden_dim,
                 rnn_fast_hidden, rnn_slow_hidden,
                 rnn_num_layers,
                 K):
        super().__init__()
        self.n_st_fast   = n_st_fast
        self.n_ln_fast   = n_ln_fast
        self.n_n_fast    = n_n_fast
        self.n_st_slow   = n_st_slow
        self.n_ln_slow   = n_ln_slow
        self.n_n_slow    = n_n_slow
        self.s_dim       = s_dim
        self.f_dim       = f_dim
        self.K           = K

        n_fast = n_st_fast + n_ln_fast + n_n_fast
        n_slow = n_st_slow + n_ln_slow + n_n_slow

        self.slow_emitter    = MixedEmitter(s_dim, hidden_dim, n_st_slow, n_ln_slow, n_n_slow)
        self.fast_emitter    = MixedEmitter(f_dim, hidden_dim, n_st_fast, n_ln_fast, n_n_fast)
        self.slow_transition = GatedTransition(s_dim, hidden_dim)
        self.fast_transition = ConditionedGatedTransition(f_dim, s_dim, hidden_dim)
        self.slow_combiner   = Combiner(s_dim, rnn_slow_hidden)
        self.fast_combiner   = FastCombiner(f_dim, s_dim, rnn_fast_hidden)
        self.slow_rnn        = nn.GRU(n_slow, rnn_slow_hidden, rnn_num_layers, batch_first=True)
        self.fast_rnn        = nn.GRU(n_fast, rnn_fast_hidden, rnn_num_layers, batch_first=True)

        self.s_0      = nn.Parameter(torch.zeros(s_dim))
        self.f_0      = nn.Parameter(torch.zeros(f_dim))
        self.h_slow_0 = nn.Parameter(torch.zeros(rnn_num_layers, 1, rnn_slow_hidden))
        self.h_fast_0 = nn.Parameter(torch.zeros(rnn_num_layers, 1, rnn_fast_hidden))

    def _split_fast(self, x):
        a = self.n_st_fast
        b = self.n_st_fast + self.n_ln_fast
        return x[..., :a], x[..., a:b], x[..., b:]

    def _split_slow(self, x):
        a = self.n_st_slow
        b = self.n_st_slow + self.n_ln_slow
        return x[..., :a], x[..., a:b], x[..., b:]

    def _emit_obs(self, emitter, z, x_st, x_ln, x_n, t, prefix):
        df, st_mu, st_sigma, ln_mu, ln_sigma, n_mu, n_sigma = emitter(z)
        if emitter.n_studentt > 0:
            pyro.sample(f"{prefix}_st_{t}",
                        dist.StudentT(df, st_mu, st_sigma).to_event(1),
                        obs=x_st)
        if emitter.n_lognormal > 0:
            pyro.sample(f"{prefix}_ln_{t}",
                        dist.LogNormal(ln_mu, ln_sigma).to_event(1),
                        obs=x_ln)
        if emitter.n_normal > 0:
            pyro.sample(f"{prefix}_n_{t}",
                        dist.Normal(n_mu, n_sigma).to_event(1),
                        obs=x_n)

    def model(self, x_fast, x_slow):
        # x_fast: (batch, T_fast, n_fast)
        # x_slow: (batch, T_slow, n_slow)  T_slow = T_fast // K
        pyro.module("nested_dmm", self)
        batch_size, T_fast, _ = x_fast.shape
        T_slow = x_slow.shape[1]

        x_fast_st, x_fast_ln, x_fast_n = self._split_fast(x_fast)
        x_slow_st, x_slow_ln, x_slow_n = self._split_slow(x_slow)

        s_prev = self.s_0.expand(batch_size, self.s_dim)
        f_prev = self.f_0.expand(batch_size, self.f_dim)

        with pyro.plate("data", batch_size):
            for tau in range(T_slow):
                s_mean, s_std = self.slow_transition(s_prev)
                s_tau = pyro.sample(f"s_{tau}",
                                    dist.Normal(s_mean, s_std).to_event(1))

                self._emit_obs(self.slow_emitter, s_tau,
                               x_slow_st[:, tau, :],
                               x_slow_ln[:, tau, :],
                               x_slow_n[:, tau, :],
                               tau, "x_slow")

                for k in range(self.K):
                    t = tau * self.K + k
                    if t >= T_fast:
                        break
                    f_mean, f_std = self.fast_transition(f_prev, s_tau)
                    f_t = pyro.sample(f"f_{t}",
                                      dist.Normal(f_mean, f_std).to_event(1))

                    self._emit_obs(self.fast_emitter, f_t,
                                   x_fast_st[:, t, :],
                                   x_fast_ln[:, t, :],
                                   x_fast_n[:, t, :],
                                   t, "x_fast")
                    f_prev = f_t

                s_prev = s_tau

    def guide(self, x_fast, x_slow):
        # Strict t-1 posterior: the RNN context for step t is built from x[0:t-1] only.
        # Achieved by feeding x[:-1] to the RNN and prepending h_0 for step 0.
        # h_slow_ctx[:, tau, :] = context from x_slow[0 : tau]   (excludes x_slow[tau]).
        # h_fast_ctx[:, t,   :] = context from x_fast[0 : t]     (excludes x_fast[t]).
        pyro.module("nested_dmm", self)
        batch_size, T_fast, _ = x_fast.shape
        T_slow = x_slow.shape[1]

        h_slow_0 = self.h_slow_0.expand(-1, batch_size, -1).contiguous()
        h_slow_enc, _ = self.slow_rnn(x_slow[:, :-1, :], h_slow_0)
        h_slow_ctx = torch.cat(
            [h_slow_0[-1].unsqueeze(1), h_slow_enc], dim=1
        )  # (batch, T_slow, rnn_slow_hidden)

        h_fast_0 = self.h_fast_0.expand(-1, batch_size, -1).contiguous()
        h_fast_enc, _ = self.fast_rnn(x_fast[:, :-1, :], h_fast_0)
        h_fast_ctx = torch.cat(
            [h_fast_0[-1].unsqueeze(1), h_fast_enc], dim=1
        )  # (batch, T_fast, rnn_fast_hidden)

        s_prev = self.s_0.expand(batch_size, self.s_dim)
        f_prev = self.f_0.expand(batch_size, self.f_dim)

        with pyro.plate("data", batch_size):
            for tau in range(T_slow):
                s_mean, s_std = self.slow_combiner(s_prev, h_slow_ctx[:, tau, :])
                s_tau = pyro.sample(f"s_{tau}",
                                    dist.Normal(s_mean, s_std).to_event(1))

                for k in range(self.K):
                    t = tau * self.K + k
                    if t >= T_fast:
                        break
                    f_mean, f_std = self.fast_combiner(f_prev, s_tau, h_fast_ctx[:, t, :])
                    f_t = pyro.sample(f"f_{t}",
                                      dist.Normal(f_mean, f_std).to_event(1))
                    f_prev = f_t

                s_prev = s_tau

    def infer(self, x_fast, x_slow):
        # Strict t-1 inference: mirrors guide() exactly but returns means/stds.
        # h_slow_ctx[:, tau, :] = context from x_slow[0 : tau]   (excludes x_slow[tau]).
        # h_fast_ctx[:, t,   :] = context from x_fast[0 : t]     (excludes x_fast[t]).
        #
        # Returns:
        #   s_means, s_stds: (batch, T_slow, s_dim)
        #   f_means, f_stds: (batch, T_fast, f_dim)
        batch_size, T_fast, _ = x_fast.shape
        T_slow = x_slow.shape[1]

        h_slow_0 = self.h_slow_0.expand(-1, batch_size, -1).contiguous()
        h_slow_enc, _ = self.slow_rnn(x_slow[:, :-1, :], h_slow_0)
        h_slow_ctx = torch.cat(
            [h_slow_0[-1].unsqueeze(1), h_slow_enc], dim=1
        )  # (batch, T_slow, rnn_slow_hidden)

        h_fast_0 = self.h_fast_0.expand(-1, batch_size, -1).contiguous()
        h_fast_enc, _ = self.fast_rnn(x_fast[:, :-1, :], h_fast_0)
        h_fast_ctx = torch.cat(
            [h_fast_0[-1].unsqueeze(1), h_fast_enc], dim=1
        )  # (batch, T_fast, rnn_fast_hidden)

        s_prev = self.s_0.expand(batch_size, self.s_dim)
        f_prev = self.f_0.expand(batch_size, self.f_dim)

        s_means, s_stds = [], []
        f_means, f_stds = [], []

        for tau in range(T_slow):
            s_mean, s_std = self.slow_combiner(s_prev, h_slow_ctx[:, tau, :])
            s_means.append(s_mean)
            s_stds.append(s_std)
            s_tau = s_mean

            for k in range(self.K):
                t = tau * self.K + k
                if t >= T_fast:
                    break
                f_mean, f_std = self.fast_combiner(f_prev, s_tau, h_fast_ctx[:, t, :])
                f_means.append(f_mean)
                f_stds.append(f_std)
                f_prev = f_mean

            s_prev = s_mean

        s_means = torch.stack(s_means, dim=1)   # (batch, T_slow, s_dim)
        s_stds  = torch.stack(s_stds,  dim=1)
        f_means = torch.stack(f_means, dim=1)   # (batch, T_fast, f_dim)
        f_stds  = torch.stack(f_stds,  dim=1)

        return s_means, s_stds, f_means, f_stds
