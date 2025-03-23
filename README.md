> [!IMPORTANT]
> This package is not actively maintained, please report any version incompatibility, bugs and errors using via Issues :)

<p align="center">
  <img src="assets/logo.png" alt="SwAds Logo" width="300"/>
</p>

SwAds is a Python package for simulating Pressure Swing Adsorption (PSA) processes with support for multi-component gas mixtures, various PSA cycle configurations, and detailed column dynamics.

## Features

- Detailed modeling of PSA columns with configurable steps
- Support for multi-component gas mixtures and multiple adsorbents
- Thermal effects and heat transfer to environment
- Flexible cycle configuration with customizable boundary conditions
- HDF5-based storage of simulation results
- Built upon open source tools ([CasADi](https://web.casadi.org/) & [IPOPT](https://coin-or.github.io/Ipopt/))

## Prerequisite

You need to have `casadi` and `ipopt` (preferably with HSL solver) installed to run simulations. There are many ways to install those tools you could find online. One of the easiest way I found is to install `casadi` using `pip`, this will give you `ipopt` wihout HSL solver. Then you can obtain prebuild libraries of HSL from https://licences.stfc.ac.uk/products/Software/HSL/LibHSL and provide the HSL library at runtime (export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:your/libHSL/lib/path).

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/swads.git
cd swads

# Install the package
pip install -e .
```

## Documentation

See the `docs` directory for detailed documentation on:

- [Theory and Equations](docs/theory.md)
- [API Reference](docs/api.md)
- [Example](docs/example.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
