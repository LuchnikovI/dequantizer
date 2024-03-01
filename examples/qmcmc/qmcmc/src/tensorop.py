import jax.numpy as jnp
from jax import Array


def _eval_energy(
    config: Array,
    coupling_amplitudes: Array,
    coupled_spin_pairs: Array,
    fields: Array,
) -> Array:
    external_field_energy = jnp.tensordot(fields, config, axes=1)
    coupling_energy = jnp.tensordot(
        config[coupled_spin_pairs[:, 0]] * config[coupled_spin_pairs[:, 1]],
        coupling_amplitudes,
        axes=1,
    )
    return -external_field_energy - coupling_energy
