"""
SwAds - Simulation of Pressure Swing Adsorption (PSA) Processes

A Python package for modeling and simulation of pressure swing adsorption processes
with support for multi-component gas mixtures, various PSA cycle configurations,
and detailed column dynamics.
"""

__version__ = "0.0.1"

from swads.column import Column
from swads.cycle import Cycle

__all__ = ["Column", "Cycle"]
