import numpy as np
from PIL import Image

import utils


class TriangleVec:
    def __init__(self, n: int, h: int, w: int):
        self.n = n
        self.height = h
        self.width = w

        self.points = self.get_random()

    def get_random(self) -> np.ndarray:
        """Get n random triangle points given the width and height of the image

        Args:
            n (int): number of triangle points
            w (int): image width
            h (int): image height

        Returns:
            np.ndarray: (n, 6) array of triangle points
        """

        n = self.n

        xx1 = np.random.randint(0, self.width - 1, n)
        yy1 = np.random.randint(0, self.height - 1, n)
        xx2 = xx1 + np.random.randint(0, 30, n) - 15
        yy2 = yy1 + np.random.randint(0, 30, n) - 15
        xx3 = xx1 + np.random.randint(0, 30, n) - 15
        yy3 = yy1 + np.random.randint(0, 30, n) - 15
        return np.column_stack((xx1, yy1, xx2, yy2, xx3, yy3))

    def point_in_triangle(self, pts, v1, v2, v3) -> bool:
        # 从 pts 中提取三角形索引和坐标
        triangle_indices = pts[:, 0]
        x_coords = pts[:, 1]
        y_coords = pts[:, 2]

        # 根据三角形索引获取对应的顶点坐标
        v1 = v1[triangle_indices]
        v2 = v2[triangle_indices]
        v3 = v3[triangle_indices]

        # 计算符号函数
        def sign(p1, p2, p3):
            return (p1[:, 0] - p3[:, 0]) * (p2[:, 1] - p3[:, 1]) - (
                p2[:, 0] - p3[:, 0]
            ) * (p1[:, 1] - p3[:, 1])

        # 计算每个点与三角形的三个顶点的符号
        d1 = sign(pts[:, 1:], v1, v2)
        d2 = sign(pts[:, 1:], v2, v3)
        d3 = sign(pts[:, 1:], v3, v1)

        # 检查点是否在三角形内
        has_neg = np.logical_or(d1 < 0, np.logical_or(d2 < 0, d3 < 0))
        has_pos = np.logical_or(d1 > 0, np.logical_or(d2 > 0, d3 > 0))

        return np.logical_not(np.logical_and(has_neg, has_pos))

    def mask(self) -> np.ndarray:
        """Get the mask of n random triangle points given the width and height of the image

        Args:
            n (int): number of triangle points
            w (int): image width
            h (int): image height

        Returns:
            np.ndarray: (n, h, w) array of mask
        """

        h, w = self.height, self.width

        # 计算所有三角形的边界框
        min_x = np.min(self.points[:, [0, 2, 4]], axis=1)
        max_x = np.max(self.points[:, [0, 2, 4]], axis=1)
        min_y = np.min(self.points[:, [1, 3, 5]], axis=1)
        max_y = np.max(self.points[:, [1, 3, 5]], axis=1)

        # 生成掩码
        masks = np.zeros((self.n, h, w), dtype=bool)
        for i in range(self.n):
            masks[i, min_y[i] : max_y[i] + 1, min_x[i] : max_x[i] + 1] = True

        # 获取所有边界框内的像素点
        n_indexs, y_coords, x_coords = np.nonzero(masks)

        # 将像素点转换为 (non_zero_nums, 3) 的数组
        pts = np.column_stack((n_indexs, x_coords, y_coords))

        # 判断这些点是否在三角形内
        in_triangle = self.point_in_triangle(
            pts, self.points[:, [0, 1]], self.points[:, [2, 3]], self.points[:, [4, 5]]
        )

        # 创建一个与输入图像大小相同的掩码
        masks = np.zeros((self.n, h, w), dtype=bool)
        masks[n_indexs[in_triangle], y_coords[in_triangle], x_coords[in_triangle]] = (
            True
        )
        return masks

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

    # def add(self, shape: Shape, color=None, in_place: bool = False) -> np.ndarray:
    #     mask = shape.mask()
    #     if in_place:
    #         self.noise_image[mask] = color
    #         return self.noise_image

    #     new_image = np.copy(self.noise_image)
    #     new_image[mask] = color
    #     return new_image

    # def extract_color(self, shape: Shape, alpha: float = 0.5) -> tuple[int, int, int]:
    #     mask = shape.mask()
    #     if self.origin_image[mask].size == 0:
    #         return tuple(np.zeros(3).astype(int))
    #     color = np.mean(self.origin_image[mask], axis=(0,)).astype(int)
    #     color = alpha * color + (1 - alpha) * self.noise_image[mask]
    #     color = tuple(np.clip(color, 0, 255).astype(int))
    #     return color

def loss_func(noise_img, target_img):
    # 计算图之间的距离
    dist = np.sum((noise_img - target_img) ** 2)
    return np.sqrt(dist / noise_img.size)

@utils.timeit
def get_random_shape(state: State):
    # generate 1000 random shape and return the best one
    h, w = state.origin_image.shape[:2]

    triangles = TriangleVec(100, h, w)
    masks = triangles.mask() # (1000, h, w)
    masks = masks[:, :, :, np.newaxis] # (1000, h, w, 1)
    ori_img = state.origin_image[np.newaxis, :, :, :] # (1, h, w, 3)
    # 提取像素值
    pixels = ori_img * masks # (1000, h, w, 3)
    colors = np.sum(pixels, axis=(1, 2)) / (np.sum(masks, axis=(1, 2)) + 1e-6)
    
    masks_expanded = masks.repeat(3, axis=3)
    average_colors_expanded = colors[:, np.newaxis, np.newaxis, :]
    # n, h, w, 3
    colored_image = state.noise_image * (1 - masks_expanded) + average_colors_expanded * masks_expanded


    # return best_shape

# def hillclimb(state: State, shape: Shape):
#     color = state.extract_color(shape)
#     noise_image = state.add(shape, color)

#     best_shape = shape
#     best_loss = loss_func(noise_image, state.origin_image)

#     # loop for 100 times
#     for _ in range(100):
#         original_shape = shape.copy()
#         shape.mutate()

#         color = state.extract_color(shape)
#         noise_image = state.add(shape, color=color)
#         loss = loss_func(noise_image, state.origin_image)

#         if loss < best_loss:
#             best_shape = shape
#             best_loss = loss
#         else:
#             shape = original_shape

#     return best_shape, best_loss

# def collect_optimized_shape(state, number_of_process=16):
#     best_shape = None
#     best_loss = None

#     with multiprocessing.Pool(processes=number_of_process) as pool:
#         results = pool.starmap(optimize_shape, [(state,)] * number_of_process)

#     for shape, shape_loss in results:
#         if best_shape is None or shape_loss < best_loss:
#             best_shape = shape
#             best_loss = shape_loss

#     return best_shape

# def optimize_shape(state: State):
#     shape = get_random_shape(state)
#     shape, shape_loss = hillclimb(state, shape)
#     return shape, shape_loss


# def step(state) -> Shape:
#     shape = collect_optimized_shape(state)
#     return shape


def perprocess(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    origin_height, origin_width = image.size
    height, width = size
    if origin_width > origin_height:
        height = int(origin_height * (width / origin_width))
    else:
        width = int(origin_width * (height / origin_height))

    image = image.resize((height, width))
    return image

def main():
    image_path = "examples/monalisa.png"
    image = Image.open(image_path)
    image = image.convert("RGB")

    origin_height, origin_width = image.size
    height, width = 256, 256
    image = perprocess(image, (height, width))
    
    image_array = np.array(image)
    state = State(image_array)
    
    get_random_shape(state)

    # n = 100
    # for i in tqdm(range(n)):
    #     shape = step(state)
    #     color = state.extract_color(shape)
    #     state.add(shape, color, in_place=True)

    # output = Image.fromarray(state.noise_image)
    # output = output.resize((origin_height, origin_width), Image.Resampling.BICUBIC)
    # output.save("result.png")

if __name__ == "__main__":
    main()