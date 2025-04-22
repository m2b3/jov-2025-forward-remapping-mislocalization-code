import equinox as eqx


def update_with_overrides(base_params, overrides=None):
    if not overrides:
        return base_params
    return eqx.tree_at(
        lambda p: tuple(getattr(p, k) for k in overrides),
        base_params,
        tuple(overrides.values()),
    )
