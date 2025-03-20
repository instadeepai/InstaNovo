from __future__ import annotations

import bisect
import os
import pickle
from dataclasses import dataclass

import numpy as np

from instanovo.__init__ import console
from instanovo.types import KnapsackChart, MassArray
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


@dataclass
class Knapsack:
    """A class that precomputes and stores a knapsack chart.

    Args:
        max_mass (float):
            The maximum mass up to which the chart is
            calculated.

        mass_scale (int):
            The scale in Daltons at which masses are
            calculated and rounded off. For example,
            a scale of 10000 would represent masses
            at a scale of 1e4 Da.

        residues (list[str]):
            The list of residues that are considered
            in knapsack decoding. The order of this
            list is the inverse of `residue_indices`.

        residue_indices (dict[str, int]):
            A mapping from residues as strings
            to indices in the knapsack chart.
            This is the inverse of `residues`.

        masses (numpy.ndarray[number of masses]):
            The set of realisable masses in ascending order.

        chart (numpy.ndarray[number of masses, number of residues]):
            The chart of realisable masses and residues that
            can lead to these masses.
            `chart[mass, residue]` is `True` if and only if
            a sequence of `mass` can be generated starting with
            the residue with index `residue`.
    """

    max_mass: float
    mass_scale: int
    residues: list[str]
    residue_indices: dict[str, int]
    masses: MassArray
    chart: KnapsackChart

    def save(self, path: str) -> None:
        """Save the knapsack file to a directory.

        Args:
            path (str):
                The path to the directory.

        Raises:
            FileExistsError: If the directory `path` already exists,
                this message raise an exception.
        """
        if os.path.exists(path):
            raise FileExistsError

        os.mkdir(path=path)
        parameters = (
            self.max_mass,
            self.mass_scale,
            self.residues,
            self.residue_indices,
        )
        pickle.dump(parameters, open(os.path.join(path, "parameters.pkl"), "wb"))
        np.save(os.path.join(path, "masses.npy"), self.masses)
        np.save(os.path.join(path, "chart.npy"), self.chart)

    @classmethod
    def construct_knapsack(
        cls,
        residue_masses: dict[str, float],
        residue_indices: dict[str, int],
        max_mass: float,
        mass_scale: int,
    ) -> "Knapsack":
        """Construct a knapsack chart using depth-first search.

        Previous construction algorithms have used dynamic
        programming, but its space and time complexity
        scale linearly with mass resolution since every
        `possible` mass is iterated over rather than only the
        `feasible` masses.

        Graph search algorithms only
        iterate over `feasible` masses which become a
        smaller and smaller share of possible masses as the
        mass resolution increases. This leads to dramatic
        performance improvements.

        This implementation uses depth-first search since
        its agenda is a stack which can be implemented
        using python lists whose operations have amortized
        constant time complexity.

        Args:
            residue_masses (dict[str, float]):
                A mapping from considered residues
                to their masses.

            max_mass (float):
                The maximum mass up to which the chart is
                calculated.

            mass_scale (int):
                The scale in Daltons at which masses are
                calculated and rounded off. For example,
                a scale of 10000 would represent masses
                at a scale of 1e4 Da.
        """
        # Convert the maximum mass to units of the mass scale
        scaled_max_mass = round(max_mass * mass_scale)

        logger.info("Scaling masses.")
        # Load residue information into appropriate data structures
        residues, scaled_residue_masses = [""], {}
        for residue, mass in residue_masses.items():
            residues.append(residue)
            if abs(mass) > 0:
                scaled_residue_masses[residue] = round(mass * mass_scale)

        # Initialize the search agenda
        mass_dim = round(max_mass * mass_scale) + 1
        residue_dim = max(residue_indices.values()) + 1
        chart = np.full((mass_dim, residue_dim), False)
        logger.info("Initializing chart.")
        agenda, visited = [], set()
        for residue, mass in scaled_residue_masses.items():
            agenda.append(mass)
            chart[mass, residue_indices[residue]] = True

        # Perform depth-first search
        logger.info("Performing search.")
        while agenda:
            current_mass = agenda.pop()

            if current_mass in visited:
                continue

            for residue, mass in scaled_residue_masses.items():
                next_mass = current_mass + mass
                if next_mass <= scaled_max_mass:
                    agenda.append(next_mass)
                    chart[next_mass, residue_indices[residue]] = True
            visited.add(current_mass)

        masses = np.array(sorted(visited))
        return cls(
            max_mass=max_mass,
            mass_scale=mass_scale,
            residues=residues,
            residue_indices=residue_indices,
            masses=masses,
            chart=chart,
        )

    @classmethod
    def from_file(cls, path: str) -> "Knapsack":
        """Load a knapsack saved to a directory.

        Args:
            path (str):
                The path to the directory.

        Returns:
            _type_: _description_
        """
        max_mass, mass_scale, residues, residue_indices = pickle.load(
            open(os.path.join(path, "parameters.pkl"), "rb")
        )
        masses = np.load(os.path.join(path, "masses.npy"))
        chart = np.load(os.path.join(path, "chart.npy"))
        return cls(
            max_mass=max_mass,
            mass_scale=mass_scale,
            residues=residues,
            residue_indices=residue_indices,
            masses=masses,
            chart=chart,
        )

    def get_feasible_masses(self, target_mass: float, tolerance: float) -> list[int]:
        """Find a set of feasible masses for a given target mass and tolerance using binary search.

        Args:
            target_mass (float):
                The masses to be decoded in Daltons.

            tolerance (float):
                The mass tolerance in Daltons.

        Returns:
            list[int]:
                A list of feasible masses.
        """
        scaled_min_mass = round(self.mass_scale * (target_mass - tolerance))
        scaled_max_mass = round(self.mass_scale * (target_mass + tolerance))

        left_endpoint = bisect.bisect_right(self.masses, scaled_min_mass)
        right_endpoint = bisect.bisect_left(self.masses, scaled_max_mass)

        feasible_masses: list[int] = self.masses[left_endpoint:right_endpoint].tolist()
        return feasible_masses
