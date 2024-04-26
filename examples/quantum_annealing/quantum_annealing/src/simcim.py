from typing import Union
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey, split, normal, randint
from quantum_annealing.src.energy_function import EnergyFunction

"""A class for SimCim optimizer."""


class SimCim:
    """Initializes a SimCim instance.
    Args:
        sigma: std of the additive noise used for exploration,
        attempt_num: number of runs,
        alpha: momentum parameter,
        c_th: restriction on amplitudes growth,
        zeta: coupling constant,
        N: number ot time steps,
        dt: time step (learning rate),
        o: overall factor,
        d: displacement,
        s: slope,
        seed: random seed.
    """

    def __init__(
        self,
        sigma: Union[float, Array],
        attempt_num: int,
        alpha: Union[float, Array] = 0.9,
        c_th: Union[float, Array] = 1.0,
        zeta: Union[float, Array] = 0.1,
        N: int = 1000,
        dt: Union[float, Array] = 0.1,
        o: Union[float, Array] = 0.1,
        d: Union[float, Array] = -0.5,
        s: Union[float, Array] = 0.1,
        seed: int = 42,
    ):
        self.N = N
        self.alpha = alpha
        self.dt = dt
        self.sigma = sigma
        self.attempt_num = attempt_num
        self.s = s
        self.d = d
        self.o = o
        self.zeta = zeta
        self.c_th = c_th
        self.seed = seed
        self.key = PRNGKey(seed)

    """Amplitude increment.
    Args:
        c: batch of amplitudes,
        p: pumping parameter,
        J: adjacency matrix,
        b: biases (magnetic fields).
    """

    def ampl_inc(
        self,
        c: Array,
        p: Union[float, Array],
        J: Array,
        b: Array,
    ):
        self.key, subkey = split(self.key)
        return ((p * c + self.zeta * (J @ c + b)) * self.dt) + (
            self.sigma * normal(subkey, (c.shape[0], self.attempt_num))
        )

    """Tanh pump parametrization.
    Args:
        J: adjacency matrix.
    """

    def tanh_pump(self, J: Array):
        i = jnp.arange(self.N)
        arg = (i / self.N - 0.5) * self.s
        Jmax = jnp.max(jnp.sum(jnp.abs(J), 1))
        return self.o * (jnp.tanh(arg) + self.d) * Jmax

    """Linear pump parametrization.
    Args:
        J: adjacency matrix.
    """

    def pump_lin(self, J):
        t = self.dt * jnp.arange(self.N)
        eigs = jnp.linalg.eigh(J)[0]
        eig_min = jnp.min(eigs)
        eig_max = jnp.max(eigs)
        p = -self.zeta * eig_max + self.zeta * (eig_max - eig_min) / t[-1] * t
        return p

    """Runs optimization loop.
    Args:
        energy_function: Energy Function.
    """

    def evolve(self, energy_function: EnergyFunction):
        b = energy_function.fields.reshape((-1, 1))
        dim = b.shape[0]
        J = jnp.zeros((dim, dim))
        J = J.at[
            (
                energy_function.coupled_spin_pairs[:, 0],
                energy_function.coupled_spin_pairs[:, 1],
            )
        ].set(energy_function.coupling_amplitudes)
        J = J + J.T
        self.key, subkey = split(self.key)
        random_attempt = randint(subkey, (1,), 0, self.attempt_num)[0]
        c_current = jnp.zeros((dim, self.attempt_num))
        c_evol = jnp.zeros((dim, self.N))
        c_evol = c_evol.at[:, 0].set(c_current[:, random_attempt])
        p = self.pump_lin(J)
        dc_momentum = jnp.zeros((dim, self.attempt_num))
        for i in range(1, self.N):
            dc = self.ampl_inc(c_current, p[i], J, b)
            dc_momentum = self.alpha * dc_momentum + (1 - self.alpha) * dc
            c1 = c_current + dc_momentum
            th_test = jnp.abs(c1) < self.c_th
            c_current = (
                th_test * (c_current + dc_momentum)
                + (1.0 - th_test) * jnp.sign(c_current) * self.c_th
            )
            c_evol = c_evol.at[:, i].set(c_current[:, random_attempt])
        return c_current, c_evol
