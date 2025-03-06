import random

from shapes.shape import Shape


class Circle(Shape):
    def __init__(self, x: int, y: int, radius: int):
        self.x = x
        self.y = y
        self.radius = radius
