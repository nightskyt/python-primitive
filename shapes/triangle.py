import math
import random

import numpy as np

from shapes.shape import Shape


class Triangle(Shape):
    def __init__(
        self,
        height: int,
        width: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        x3: int,
        y3: int,
    ):
        self.height = height
        self.width = width

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3

    @classmethod
    def get_random(cls, h: int, w: int) -> Shape:
        """Get a random triangle given the width and height of the image

        Args:
            w (int): image width
            h (int): image height

        Returns:
            Triangle: An Triangle object
        """

        xx1 = random.randint(0, w - 1)
        yy1 = random.randint(0, h - 1)
        xx2 = xx1 + random.randint(0, 30) - 15
        yy2 = yy1 + random.randint(0, 30) - 15
        xx3 = xx1 + random.randint(0, 30) - 15
        yy3 = yy1 + random.randint(0, 30) - 15
        return cls(h, w, xx1, yy1, xx2, yy2, xx3, yy3)

    def mutate(self) -> None:
        """mutate the triangle

        Args:
            w (int): image width
            h (int): image height
        """
        m = 16
        rnd = random.Random()
        w, h = self.width, self.height

        while True:
            point = random.randint(0, 2)
            if point == 0:
                self.x1 = np.clip(self.x1 + int(rnd.gauss() * 16), -m, w - 1 + m)
                self.y1 = np.clip(self.y1 + int(rnd.gauss() * 16), -m, h - 1 + m)
            elif point == 1:
                self.x2 = np.clip(self.x2 + int(rnd.gauss() * 16), -m, w - 1 + m)
                self.y2 = np.clip(self.y2 + int(rnd.gauss() * 16), -m, h - 1 + m)
            else:
                self.x3 = np.clip(self.x3 + int(rnd.gauss() * 16), -m, w - 1 + m)
                self.y3 = np.clip(self.y3 + int(rnd.gauss() * 16), -m, h - 1 + m)

            if self.valid():
                break

    def valid(self) -> bool:
        """Check if the triangle is valid

        Returns:
            bool: True if the triangle is valid
        """
        min_degrees = 15

        def calculate_angle(x1, y1, x2, y2, x3, y3):
            dx1, dy1 = x2 - x1, y2 - y1
            dx2, dy2 = x3 - x1, y3 - y1
            d1 = math.sqrt(dx1 * dx1 + dy1 * dy1) + 1e-5
            d2 = math.sqrt(dx2 * dx2 + dy2 * dy2) + 1e-5
            dx1, dy1 = dx1 / d1, dy1 / d1
            dx2, dy2 = dx2 / d2, dy2 / d2
            return math.degrees(math.acos(dx1 * dx2 + dy1 * dy2))

        a1 = calculate_angle(self.x1, self.y1, self.x2, self.y2, self.x3, self.y3)
        a2 = calculate_angle(self.x2, self.y2, self.x1, self.y1, self.x3, self.y3)
        a3 = 180 - a1 - a2

        return a1 > min_degrees and a2 > min_degrees and a3 > min_degrees

    @staticmethod
    def point_in_triangle(pts, v1, v2, v3) -> bool:
        """
        判断点pts是否在三角形v1, v2, v3内
        pts: (N, 2) 的点集
        v1, v2, v3: 三角形的顶点
        """

        def sign(p1, p2, p3):
            return (p1[:, 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (
                p1[:, 1] - p3[1]
            )

        d1 = sign(pts, v1, v2)
        d2 = sign(pts, v2, v3)
        d3 = sign(pts, v3, v1)

        has_neg = np.logical_or(d1 < 0, np.logical_or(d2 < 0, d3 < 0))
        has_pos = np.logical_or(d1 > 0, np.logical_or(d2 > 0, d3 > 0))

        return np.logical_not(np.logical_and(has_neg, has_pos))

    def mask(self) -> np.ndarray:
        h, w = self.height, self.width
        x1, y1, x2, y2, x3, y3 = self.x1, self.y1, self.x2, self.y2, self.x3, self.y3
        min_x, max_x = max(0, min(x1, x2, x3)), min(w - 1, max(x1, x2, x3))
        min_y, max_y = max(0, min(y1, y2, y3)), min(h - 1, max(y1, y2, y3))

        # 创建一个布尔掩码，表示哪些像素在边界框内
        mask = np.zeros((h, w), dtype=bool)
        mask[min_y : max_y + 1, min_x : max_x + 1] = True

        # 获取边界框内的所有像素点
        y_coords, x_coords = np.nonzero(mask)
        pts = np.column_stack((x_coords, y_coords))

        # 判断这些点是否在三角形内
        in_triangle = self.point_in_triangle(pts, (x1, y1), (x2, y2), (x3, y3))

        # 创建一个与输入图像大小相同的掩码
        mask = np.zeros((h, w), dtype=bool)
        mask[y_coords[in_triangle], x_coords[in_triangle]] = True
        return mask

    def copy(self) -> Shape:
        return self.__class__(
            self.height,
            self.width,
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.x3,
            self.y3,
        )
