import numpy as np

from shapes.shape import Shape


class State:
    def __init__(self, image):
        self.origin_image = image
        self.height = image.shape[0]
        self.width = image.shape[1]

        self.noise_image = None
        self._init()

    def _init(self) -> None:
        color = tuple(np.mean(self.origin_image, axis=(0, 1)).astype(int))
        self.noise_image = np.full_like(self.origin_image, color)

    def add(self, shape: Shape, color=None, in_place: bool = False) -> np.ndarray:
        mask = shape.mask()
        if in_place:
            self.noise_image[mask] = color
            return self.noise_image

        new_image = np.copy(self.noise_image)
        new_image[mask] = color
        return new_image

    def extract_color(self, shape: Shape, alpha: float = 0.5) -> tuple[int, int, int]:
        mask = shape.mask()
        if self.origin_image[mask].size == 0:
            return tuple(np.zeros(3).astype(int))
        color = np.mean(self.origin_image[mask], axis=(0,)).astype(int)
        color = alpha * color + (1 - alpha) * self.noise_image[mask]
        color = tuple(np.clip(color, 0, 255).astype(int))
        return color
