from enum import Enum
from src.parameterization import ParamSpec, parameterized, BaseParams


# XXX not really used at the moment...
class ChangeType(Enum):
    MODEL_PARAMS = "model_params"
    SIM_PARAMS = "sim_params"
    EXP_PARAMS = "exp_params"


# Those parameters get passed to DftDriftModel's constructor.
# Can change shapes, and thus cause recompilation,
static_model_ps: ParamSpec = {
    "n_xs": {"type": int, "min": 51, "max": 1001, "step": 50, "value": 1001},  # 1001
    "w_exc": {"type": float, "min": 0.1, "max": 10, "step": 0.1, "value": 2},
    "w_inh": {"type": float, "min": 0.1, "max": 10, "step": 0.1, "value": 1},
    # Scaling factor for both excitatory and inhibitory kernels.
    "base_kernel_sigma": {"type": float, "min": 1, "max": 32, "step": 0.1, "value": 4},
    # Relative std of inhibitory kernel compared to excitatory kernel.
    "rel_inh_sigma": {"type": float, "min": 0.1, "max": 10, "step": 0.1, "value": 3},
    # In degrees of visual angle, what the population is capable of coding for.
    "amplitude_coded_for": {
        "type": float,
        "min": 50,
        "max": 400,
        "step": 1,
        "value": 200,
    },
}

sim_ps: ParamSpec = {
    "dt0": {"type": float, "min": 0.01, "max": 0.5, "step": 0.01, "value": 0.1},
    "sampling_dt": {"type": float, "min": 0.1, "max": 2, "step": 0.1, "value": 1.0},
    "t_start": {
        "type": float,
        "min": -300,
        "max": 0,
        "step": 50,
        "value": -250.0,
    },
    "t_end": {"type": float, "min": 0, "max": 400, "step": 50, "value": 400.0},
}

perisaccadic_flash_ps: ParamSpec = {
    "onset": {"type": float, "min": -200, "max": 150, "step": 0.1, "value": -200},
    "saccade_amplitude": {"type": float, "min": 2, "max": 50, "step": 0.1, "value": 8}, # This used to default to 10 in an earlier version.
    "flash_duration": {
        "type": float,
        "min": 0.1,
        "max": 200.0,
        "step": 0.1,
        "value": 2.0,
    },
    "input_strength": {
        "type": float,
        "min": 0,
        "max": 10000,
        "step": 0.01,
        "value": 500,
    },
    "flash_x": {
        "type": float,
        "min": -20,
        "max": 20,
        "step": 0.1,
        "value": 0.0,  # Default to current behavior
    },
}


# This will be used for class generation, sliders, and hyperparameter optimization.
# Some of these are model parameters, some are input / experimental parameters...
model_ps: ParamSpec = {
    "tau": {"type": float, "min": 0.1, "max": 100, "step": 0.1, "value": 20},
    "h": {"type": float, "min": -1000, "max": 10, "step": 0.1, "value": -40},
    "alpha0": {
        "type": float,
        "min": 1,
        "max": 100_000,
        "step": 10,
        "value": 12000,
    },
    "beta0": {"type": float, "min": 0, "max": 1000, "step": 0.1, "value": 20},
    "initial_u": {"type": float, "min": -200, "max": 20, "step": 0.1, "value": -100},
    "eta_tau_rise": {"type": float, "min": 0.1, "max": 90, "step": 0.1, "value": 15},
    "eta_tau_fall": {"type": float, "min": 0.1, "max": 90, "step": 0.1, "value": 45},
    "eta_duration": {"type": float, "min": 5, "max": 300, "step": 1, "value": 250},
    "eta_center": {"type": float, "min": -100, "max": 200, "step": 1, "value": 75},
    "act_offset": {"type": float, "min": 0, "max": 300, "step": 1, "value": 100},
    "act_slope": {"type": float, "min": 1, "max": 100, "step": 1, "value": 30},
    "smearing": {
        "type": bool,
        "value": False,
    },
    "saccade_duration": {
        "type": float,
        "min": 10,
        "max": 100,
        "step": 0.1,
        "value": 35, # to intercept 3c interpolation - XXX done manually, ideally would be automated.
    },
    "decoding_time": {
        "type": float,
        "min": -250,
        "max": 400,
        "step": 0.1,
        "value": 300,
    },
    "retina_to_lip_delay": {
        "type": float,
        "min": 15,
        "max": 60,
        "step": 0.1,
        "value": 40,
    },
    "input_onset_to_peak": {
        "type": float,
        "min": 0.1,
        "max": 50,
        "step": 0.1,
        "value": 6,
    },
    "input_spatial_std": {
        "type": float,
        "min": 0.1,
        "max": 5,
        "step": 0.1,
        "value": 2.5,
    },
    "input_temporal_sigma": {
        "type": float,
        "min": 0.1,
        "max": 50.0,
        "step": 0.1,
        "value": 15.0,
    },
    "input_offset_tau": {
        "type": float,
        "min": 0.1,
        "max": 50.0,
        "step": 0.1,
        "value": 15.0,
    },
    "input_stable_baseline": {
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "value": 1 / 6,
    },
}


@parameterized(model_ps)
class DftDriftParams(BaseParams):
    pass


@parameterized(sim_ps)
class SimParams(BaseParams):
    pass


@parameterized(perisaccadic_flash_ps)
class ExpParams(BaseParams):
    pass
