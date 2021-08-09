from pathlib import Path

import cmdstanpy
import pkg_resources

_available_models = [
    "fitter.stan",

]

base_path = Path("stan_models")

def get_stan_model(stan_model:str = "fitter.stan", mpi:bool=False, threads: bool = True) -> cmdstanpy.CmdStanModel:

    assert (
        stan_model in _available_models
    ), f"{stan_model} is not in {','.join(_available_models)}"

    stan_file = pkg_resources.resource_filename(
        "wriggle", str(base_path / stan_model)
    )

    cpp_options = {}

    if mpi:
        cpp_options["STAN_MPI"] = True

    if threads:

        cpp_options["STAN_THREADS"] = True

    model = cmdstanpy.CmdStanModel(
        stan_file=stan_file, cpp_options=cpp_options
    )

    return model


def list_stan_models() -> None:
    for m in _available_models:

        print(m)
