from typing import Optional
from pathlib import Path
from tempfile import NamedTemporaryFile
from subprocess import run
from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array
from quantum_annealing.src.energy_function import EnergyFunction

# path to MQLib executable inside docker container
path2mqlib = Path("/dequantizer/MQLib/bin/MQLib")


@dataclass
class MQLibResult:
    configuration: Array
    energy: Array


class MQLib:
    """Builds an instance of MQLib solver.
    Args:
        time_limit: maximal runtime in seconds;
        seed: random seed ranging from 0 to 65535;
        heuristic: heuristic being used."""

    def __init__(
        self,
        time_limit: Optional[int],
        seed: int,
        heuristic: str,
    ):
        self.seed = seed
        self.time_limit = time_limit
        self.heuristic = heuristic

    """Runs MQLib solver on a given energy function."""

    def run(self, energy_function: EnergyFunction) -> MQLibResult:
        file_content = energy_function.to_mqlib_format()
        with NamedTemporaryFile() as file:
            file.write(file_content)
            file.flush()
            result = run(
                [
                    path2mqlib,
                    "-h",
                    f"{self.heuristic}",
                    "-fQ",
                    f"{file.name}",
                    "-r",
                    f"{self.time_limit}",
                    "-ps",
                ],
                capture_output=True,
            )
            if len(result.stderr) != 0:
                # TODO: proper exception here
                raise ValueError(f"MQLib return non-empty stderr: {result.stderr}")
            lines = result.stdout.split(sep=b"\n")
            energy = (
                jnp.array(float(lines[0].split(sep=b",")[3]))
                - energy_function.fields.sum()
                + energy_function.coupling_amplitudes.sum()
            )
            configuration = (
                2 * jnp.array(list(map(lambda x: int(x), lines[-2].split(b" ")))) - 1
            )
            return MQLibResult(configuration, energy)
