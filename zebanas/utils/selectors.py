import numpy as np
from kneed import KneeLocator


class KneeSelector:
    def __call__(self, solutions):
        objs = solutions.get_obj()

        kneedle = KneeLocator(
            objs[:, 0],
            objs[:, 1],
            S=2.0,
            curve="convex",
            direction="decreasing",
            interp_method="polynomial"
        )

        solution_idx = np.where(objs == kneedle.knee)[0]
        solutions = solutions[solution_idx]
        for chromo in solutions:
            print(chromo.data)
            print(chromo.obj)
        return solutions
