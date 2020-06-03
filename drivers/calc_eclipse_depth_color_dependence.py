"""
We want to know for plausible HEB or BEB scenarios, what kind of eclipse depth
variations to expect over potentially observable bandpasses.
"""

import numpy as np
from timmy.multicolor import (
    get_delta_obs_given_mstars, run_bulk_depth_color_grids
)

DEBUG = 0

def main():

    if DEBUG:
        test()

    else:
        run_bulk_depth_color_grids()

def test():

    np.random.seed(42)
    delta_obs_dict = get_delta_obs_given_mstars(0.205, 0.205)

if __name__ == "__main__":
    main()
