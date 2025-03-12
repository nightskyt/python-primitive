import multiprocessing

import numpy as np
from PIL import Image
from tqdm import tqdm

from shapes.shape import Shape
from shapes.triangle import Triangle
from state import State


def loss_func(noise_img, target_img):
    # 计算图之间的距离
    dist = np.sum((noise_img - target_img) ** 2)
    return np.sqrt(dist / noise_img.size)


def get_random_shape(state: State):
    # generate 1000 random shape and return the best one
    h, w = state.origin_image.shape[:2]
    best_shape = None
    best_loss = float("inf")

    for _ in range(1000):
        shape = Triangle.get_random(h, w)
        color = state.extract_color(shape)
        noise_image = state.add(shape, color)

        loss = loss_func(noise_image, state.origin_image)
        if loss < best_loss:
            best_loss = loss
            best_shape = shape

    return best_shape


def hillclimb(state: State, shape: Shape):
    color = state.extract_color(shape)
    noise_image = state.add(shape, color)

    best_shape = shape
    best_loss = loss_func(noise_image, state.origin_image)

    # loop for 100 times
    for _ in range(100):
        original_shape = shape.copy()
        shape.mutate()

        color = state.extract_color(shape)
        noise_image = state.add(shape, color=color)
        loss = loss_func(noise_image, state.origin_image)

        if loss < best_loss:
            best_shape = shape
            best_loss = loss
        else:
            shape = original_shape

    return best_shape, best_loss


def optimize_shape(state: State):
    shape = get_random_shape(state)
    shape, shape_loss = hillclimb(state, shape)
    return shape, shape_loss


def collect_optimized_shape(state, number_of_process=16):
    best_shape = None
    best_loss = None

    with multiprocessing.Pool(processes=number_of_process) as pool:
        results = pool.starmap(optimize_shape, [(state,)] * number_of_process)

    for shape, shape_loss in results:
        if best_shape is None or shape_loss < best_loss:
            best_shape = shape
            best_loss = shape_loss

    return best_shape


def step(state) -> Shape:
    shape = collect_optimized_shape(state)
    return shape


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
    # 读取图片
    image_path = "examples/person.jpg"
    # image_path = "examples/swan.jpg"
    # image_path = "examples/monalisa.png"
    image = Image.open(image_path)
    image = image.convert("RGB")

    origin_height, origin_width = image.size
    height, width = 256, 256
    image = perprocess(image, (height, width))

    # 将图像数据转换为NumPy数组
    image_array = np.array(image)
    state = State(image_array)

    # color = tuple(np.mean(image_array, axis=(0, 1)).astype(int))
    # canvas = Image.new("RGBA", (height, width), (color[0], color[1], color[2], 255))
    # draw = ImageDraw.Draw(canvas)

    # 图形数量
    n = 100
    for i in tqdm(range(n)):
        shape = step(state)
        color = state.extract_color(shape)
        state.add(shape, color, in_place=True)

        # draw.polygon(
        #     [(shape.x1, shape.y1), (shape.x2, shape.y2), (shape.x3, shape.y3)],
        #     fill=(color[0], color[1], color[2], 128),
        # )

    output = Image.fromarray(state.noise_image)
    # # output = output.resize((origin_height, origin_width), Image.Resampling.BICUBIC)
    output.save("result.png")

    # # 显示画布
    # canvas.show()
    # canvas.save("output.png")


if __name__ == "__main__":
    main()
