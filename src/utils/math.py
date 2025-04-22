# DO NOT make modeling decisions here.

from typing import TypedDict
import jax
import jax.numpy as jnp


@jax.jit
def compute_barycenter(ws, xs):
    """
    This is meant to be used with POSITIVE weights!
    TODO: assert with JIT?
    """
    return jnp.dot(ws, xs) / jnp.sum(ws)


class BinarySearchResult(TypedDict):
    value: float
    arg: float
    n_iters: int
    prev_value: float
    prev_arg: float


def binary_search(
    func, low, high, value_tol=1e-2, arg_tol=1e-1, max_iterations=100, verbose=False
) -> BinarySearchResult:
    """
    - `func` is a function that will be minimized, assumed to be monotonically increasing.
    - `low` and `high` set the bounds of the parameter search space.
    - `value_tol`: if the absolute output value is less than this, we can immediately exit.
    - `arg_tol`: otherwise, tolerance on the argument.
    - `max_iterations`: self-explanatory.
    - `verbose`: whether to print at each binary search step.
    """

    def cond(state):
        low, high, iteration, prev_value, prev_midpoint = state
        current_midpoint = 0.5 * (low + high)
        current_value = func(current_midpoint)
        tight_value = jnp.abs(current_value) < value_tol
        tight_arg_bound = jnp.abs(high - low) < arg_tol
        within_budget = iteration < max_iterations
        return ~tight_value & ~tight_arg_bound & within_budget

    def body(state):
        low, high, iteration, prev_value, _ = state
        midpoint = 0.5 * (low + high)
        current_value = func(midpoint)
        new_low = jnp.where(current_value < 0, midpoint, low)
        new_high = jnp.where(current_value >= 0, midpoint, high)

        if verbose:
            ndec = 4
            jax.debug.print(
                "Iteration {iteration}: low={low}, high={high}, midpoint={midpoint}, value={value}",
                iteration=iteration,
                low=round(new_low, ndec),
                high=round(new_high, ndec),
                midpoint=round(midpoint, ndec),
                value=round(current_value, ndec),
            )

        return (new_low, new_high, iteration + 1, current_value, midpoint)

    initial_midpoint = 0.5 * (low + high)
    initial_value = func(initial_midpoint)
    initial_state = (low, high, 0, initial_value, initial_midpoint)
    final_state = jax.lax.while_loop(cond, body, initial_state)
    low, high, iteration, prev_value, prev_arg = final_state
    opt_arg = 0.5 * (low + high)
    opt_value = func(opt_arg)
    return {
        "value": opt_value,
        "arg": opt_arg,
        "n_iters": iteration,
        "prev_value": prev_value,
        "prev_arg": prev_arg,
    }
