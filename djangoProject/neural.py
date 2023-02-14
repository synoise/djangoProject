# import random
#
# import numpy as np
#
# # from . import RL
#
# AREA = np.zeros(400, dtype=int)
# #
#
# CURENTPLAYER = 0
#
# import pickle
#
# BOARD_ROWS = 20
# BOARD_COLS = 20
#
#
# def periodicMove(area, gamer):
#     if gamer:
#         x = -1
#     else:
#         x = 1
#     for i in range(len(area)):
#         if not (area[i]):
#             area[i] = x
#             break
#     return area, i
#
# def get_index_of(lst, element):
#     return list(map(lambda x: x[0], (list(filter(lambda x: x[1] == element, enumerate(lst))))))
# def randomMove(area, gamer):
#     # for i in range(len(area)):
#     if gamer:
#         x = -1
#     else:
#         x = 1
#     index_agent0 = get_index_of(area, 0)
#     try:
#         i = random.choice(index_agent0)
#     except:
#         i = -1
#     area[i] = x
#     return area, i
#
#
#
#
# def resetState(x=np.zeros(400, dtype=int)):
#     # print("reset state", x)
#     return x
#
#
#
#
# def addMove(_area, gamer, award: int, winner: int, aivsai: bool):
#     area, i = randomMove(_area, gamer)
#     CURENTPLAYER = gamer
#     AREA = area
#     if aivsai:
#         areaAi, pos = randomMove(area, not gamer)
#         AREA = area
#         return {"area": area.tolist(), "possition": i, "gamer": gamer, "aivsai": aivsai, "areaai": areaAi.tolist(),
#                 "pos": pos}
#     # return learn(area, gamer, award, winner)
#     return {"area": area.tolist(), "possition": i, "gamer": gamer, "aivsai": aivsai}
#
#
#
