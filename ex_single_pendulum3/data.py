import argparse
import os
import shutil
import sys
import time
import traceback

import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(".")
sys.path.append(PARENT_DIR)

from dynamic_single_pendulum_DAE import DynamicSinglePendulumDAE
from integrator import ODEIntegrate
from utils import to_pickle, from_pickle


def truncated_lambdas(values, dof):
    """
    截断lambda值。

    Args:
        values: lambda值

    Returns:
        torch.Tensor: 截断后的lambda值
    """
    q, qt, lambdas = torch.tensor_split(values, (dof, dof * 2), dim=-1)
    return torch.cat([q, qt], dim=-1)


def generate_dataset(dynamics, config, logger):
    logger.info("Start generating dataset...")

    start_time = time.time()
    y0 = torch.tensor(config.y0, device=config.device, dtype=config.dtype)
    y0 = y0.repeat(config.data_num, 1)

    t, sol = ODEIntegrate(
        func=dynamics,
        t0=config.t0,
        t1=config.t1,
        dt=config.dt,
        y0=y0,
        method=config.ode_solver,
        device=config.device,
        dtype=config.dtype,
        dof=config.dof,
    )

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"The running time of ODE solver: {execution_time} s")

    return t, sol


def get_data(config, logger, path):
    path = os.path.join(path, "single-pendulum-dataset.pkl")

    if os.path.exists(path):
        data = from_pickle(path)
        logger.info("Dataset loaded at: {}".format(path))
        return data

    dynamics = DynamicSinglePendulumDAE(config, logger)
    # logger.info("dynamics: {}".format(dynamics))

    t, y = generate_dataset(dynamics, config, logger)
    yt = torch.stack([dynamics(t, yi).clone().detach().cpu() for yi in y])
    y = y.squeeze()
    y0 = y[0].reshape(1, -1)
    yt = yt.squeeze()

    # 截断lambda值
    y = truncated_lambdas(y, config.dof)
    y0 = truncated_lambdas(y0, config.dof)
    yt = truncated_lambdas(yt, config.dof)

    data = {
        "y0": y0,
        "t": t,
        "y": y,
        "yt": yt,
    }
    data["meta"] = None
    
    # save the dataset
    to_pickle(data, path)
    logger.info("Dataset generated at: {}".format(path))

    # Log the configuration and dataset generation completion
    logger.info("Congratulations, the dataset is generated!")

    return data
