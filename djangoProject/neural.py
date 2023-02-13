import random

import numpy as np

from . import RL

AREA = np.zeros(400, dtype=int)
#

CURENTPLAYER = 0

import pickle

BOARD_ROWS = 20
BOARD_COLS = 20


def periodicMove(area, gamer):
    if gamer:
        x = -1
    else:
        x = 1
    for i in range(len(area)):
        if not (area[i]):
            area[i] = x
            break
    return area, i

def get_index_of(lst, element):
    return list(map(lambda x: x[0], (list(filter(lambda x: x[1] == element, enumerate(lst))))))
def randomMove(area, gamer):
    # for i in range(len(area)):
    if gamer:
        x = -1
    else:
        x = 1
    index_agent0 = get_index_of(area, 0)
    try:
        i = random.choice(index_agent0)
    except:
        i = -1
    area[i] = x
    return area, i


class State:
    def __init__(self, p1, p2):
        # self.board = np.ndarray(BOARD_ROWS * BOARD_COLS)
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1

    # get unique hash of current board state
    def getHash(self):
        self.boardHash = AREA
        return self.boardHash

    def winner(self):
        #0,1,-1
        self.isEnd = False
        return 0

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS * BOARD_COLS):
            if AREA[i] == 0:
                positions.append(i)  # need to be tuple
        return positions

    def updateState(self, position):
        AREA[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    # board reset
    def reset(self):
        AREA = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=100):
        for i in range(rounds):
            if i % 10 == 0:
                print("Rounds {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, AREA, self.playerSymbol)
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end

                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, AREA, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

    # play with human
    def play2(self):
        while not self.isEnd:
            # Player 1
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, AREA, self.playerSymbol)
            # take action and upate board state
            self.updateState(p1_action)
            self.showBoard()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions)

                self.updateState(p2_action)
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if AREA[i* j] == 1:
                    token = 'x'
                if AREA[i* j] == -1:
                    token = 'o'
                if AREA[i* j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')


class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    def chooseAction(self, positions, current_board, symbol):
        action=0
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx, i = randomMove(AREA, CURENTPLAYER)
            action = i
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = p
        # print("{} takes action {}".format(self.name, action))
        return action

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


p1 = Player("p1")
p2 = Player("p2")

st = State(p1, p2)
print("training...")
st.play(5000)


def resetState(x=np.zeros(400, dtype=int)):
    # print("reset state", x)
    return x




def addMove(_area, gamer, award: int, winner: int, aivsai: bool):
    area, i = randomMove(_area, gamer)
    CURENTPLAYER = gamer
    AREA = area
    if aivsai:
        areaAi, pos = randomMove(area, not gamer)
        AREA = area
        return {"area": area.tolist(), "possition": i, "gamer": gamer, "aivsai": aivsai, "areaai": areaAi.tolist(),
                "pos": pos}
    # return learn(area, gamer, award, winner)
    return {"area": area.tolist(), "possition": i, "gamer": gamer, "aivsai": aivsai}



