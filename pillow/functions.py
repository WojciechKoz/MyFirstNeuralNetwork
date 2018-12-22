from PIL import Image
from math import log, exp
from random import *


def import_picture(path: str) -> list:  # import some picture 28x28 pixels in gray-scale
    img = Image.open(path)
    pix = img.load()
    output = []

    for y in range(0, 28):
        for x in range(0, 28):
            output.append(pix[x, y][0])

    return neuronValuePack(output)  # return one-dimension list of float in range 0 1 represents brightness of pixel


def number_to_network_output(number: int) -> list:
    output = [0] * 10
    output[number] = 1

    return output


def neuronValuePack(image: list) -> list:
    for pix in image:
        pix /= 255.0

    return image


def dying_ReLU(x: float) -> float:
    if x < 0:
        return 0.1 * x
    return x


def derivative_of_ReLU(x: float) -> float:
    if x >= 0:
        return 1
    else:
        return 0.1


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def derivative_of_sigmoid(x: float) -> float:  # derivative of the sigmoid function
    return sigmoid(x) * (1 - sigmoid(x))


def create_random_list(value: int, a: float, b: float) -> list:
    output = []

    for i in range(0, value):
        output.append(uniform(a, b))

    return output
