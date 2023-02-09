import math
import random

import torch
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque


def resetState(x=np.zeros(400, dtype=int)):
    # print("reset state", x)
    return 0


def addMove(_area, gamer, award: int, winner: int, aivsai: bool):
    area, i = randomMove11(_area, gamer)
    if aivsai:
        areaAi, pos = randomMove11(area, not gamer)
        return {"area": area.tolist(), "possition": i, "gamer": gamer, "aivsai": aivsai, "areaai": areaAi.tolist(),
                "pos": pos}
    # return learn(area, gamer, award, winner)
    return {"area": area.tolist(), "possition": i, "gamer": gamer, "aivsai": aivsai}


# def getMove():
#     return

def randomMove(area, gamer):
    if gamer:
        x = -1
    else:
        x = 1
    for i in range(len(area)):
        if not (area[i]):
            area[i] = x
            break
    return area, i


def randomMove11(area, gamer):
    # for i in range(len(area)):
    if gamer:
        x = -1
    else:
        x = 1
    index_agent0 = get_index_of(area, 0)
    i = random.choice(index_agent0)
    print(i, index_agent0)
    area[i] = x
    # if not (area[i]):
    #     area[i] = 1
    #     break
    return area, i


def get_index_of(lst, element):
    return list(map(lambda x: x[0], (list(filter(lambda x: x[1] == element, enumerate(lst))))))

# def addMove(area, gamer, award, winner):
#     area[steps_done] = 1
#     return [area.tolist(),1, gamer]
