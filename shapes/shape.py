import numpy as np


class Shape:
    @classmethod
    def get_random(cls, h: int, w: int) -> "Shape":
        raise NotImplementedError

    def mutate(self) -> None:
        raise NotImplementedError

    def mask(self) -> np.ndarray:
        raise NotImplementedError

    def copy(self) -> "Shape":
        raise NotImplementedError
