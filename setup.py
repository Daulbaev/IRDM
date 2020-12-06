import setuptools

setuptools.setup(
    name="interpolated_torchdiffeq",
    version="0.0.1",
    description="ODE solvers and adjoint sensitivity analysis in PyTorch.",
    url="",
    packages=['interpolated_torchdiffeq', 'interpolated_torchdiffeq._impl'],
    package_dir={'interpolated_torchdiffeq': './interpolated_torchdiffeq'},
    install_requires=['torch>=0.4.1'],
    classifiers=(
        "Programming Language :: Python :: 3"),)
