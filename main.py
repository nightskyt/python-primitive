import argparse
import multiprocessing
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from shapes.shape import Shape
from shapes.triangle import Triangle
from state import State


def loss_func(noise_img, target_img):
    dist = np.sum((noise_img - target_img) ** 2)
    return np.sqrt(dist / noise_img.size)


def get_random_shape(state: State):
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
    best_shape = shape
    best_loss = float("inf")

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


def collect_optimized_shape(args, state):
    best_shape = None
    best_loss = None

    with multiprocessing.Pool(processes=args.processes) as pool:
        results = pool.starmap(optimize_shape, [(state,)] * args.processes)

    for shape, shape_loss in results:
        if best_shape is None or shape_loss < best_loss:
            best_shape = shape
            best_loss = shape_loss

    return best_shape


def step(args, state) -> Shape:
    shape = collect_optimized_shape(args, state)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, default="")
    parser.add_argument("-o", "--output", type=str, default="output.jpg")
    parser.add_argument("-n", type=int, default=50)
    parser.add_argument("-p", "--processes", type=int, default=os.cpu_count())

    args = parser.parse_args()

    image = Image.open(args.image)
    image = image.convert("RGB")

    origin_height, origin_width = image.size
    height, width = 256, 256
    image = perprocess(image, (height, width))

    image_array = np.array(image)
    state = State(image_array)

    n = args.n
    for _ in tqdm(range(n)):
        shape = step(args, state)
        color = state.extract_color(shape)
        state.add(shape, color, in_place=True)

    output = Image.fromarray(state.noise_image)
    output = output.resize((origin_height, origin_width), Image.Resampling.BICUBIC)
    output.save(args.output)


if __name__ == "__main__":
    main()
