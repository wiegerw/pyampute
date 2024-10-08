import os

import numpy as np
import pandas as pd


from pyampute.ampute import MultivariateAmputation


def generate_shift_lookup_table(
    n_samples: int = int(1e6),
    lower_range: float = MultivariateAmputation.DEFAULTS["lower_range"],
    upper_range: float = MultivariateAmputation.DEFAULTS["upper_range"],
    max_iter: int = MultivariateAmputation.DEFAULTS["max_iter"],
    max_diff_with_target: float = MultivariateAmputation.DEFAULTS[
        "max_diff_with_target"
    ],
):
    """
    Note: This should be run from the root folder so it is properly stored in "/data".
    """
    rng = np.random.default_rng()
    normal_sample = rng.standard_normal(size=n_samples)
    percent_missing = np.arange(0.01, 1.01, 0.01)
    score_to_prob_func_names = [
        "SIGMOID-RIGHT",
        "SIGMOID-LEFT",
        "SIGMOID-TAIL",
        "SIGMOID-MID",
    ]
    shifts = []
    for func in score_to_prob_func_names:
        shifts.append(
            [
                MultivariateAmputation._binary_search(
                    normal_sample,
                    func,
                    percent,
                    lower_range,
                    upper_range,
                    max_iter,
                    max_diff_with_target,
                )[0]
                for percent in percent_missing
            ]
        )
    percent_missing_2_decimal = ["{:.2f}".format(p) for p in percent_missing]
    lookup_table = pd.DataFrame(
        shifts, index=score_to_prob_func_names, columns=percent_missing_2_decimal,
    )
    lookup_table.to_csv(os.getcwd())


if __name__ == "__main__":
    generate_shift_lookup_table()
