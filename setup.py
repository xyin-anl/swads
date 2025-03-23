from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="swads",
    version="0.0.1",
    author="Xiangyu Yin",
    author_email="shawn.pypi@gmail.com",
    description="Simulation of Pressure Swing Adsorption (PSA) Processes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xyin-anl/swads",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "casadi",
        "h5py",
    ],
    include_package_data=True,
    package_data={"": ["assets/*.png"]},
)
