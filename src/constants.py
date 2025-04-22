import jax
import jax.numpy as jnp

# Manually chosen to be denser around "key" points
# to get quick feedback when experimenting.
# In final figures, dense sampling is used.
notable_onset_times = [
    -250,
    -240,
    -200,
    -175,
    -150,
    -100,
    -90,
    -80,
    -75,
    -70,
    -65,
    -60,
    -50,
    -30,
    -10,
    -1,
    0,
    1,
    2,
    3,
    5,
    10,
    15,
    20,
    25,
    30,
    38,
    40,
    50,
    80,
    100,
    140,
    150,
    175,
    200,
    250,
]


def flash_onset_times(notable_only=True) -> jax.Array:
    times = notable_onset_times if notable_only else jnp.linspace(-250, 250, 401)
    return jnp.array(times)
