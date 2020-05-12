#!/usr/bin/python

from boomer.algorithm._random import RNG


if __name__ == '__main__':
    random_state = 1
    rng = RNG(random_state)

    bagging = Bagging()

    for i in range(10000):
        print(str(rng.random(25, 30)))
