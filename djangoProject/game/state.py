import json
import numpy as np
from channels.generic.websocket import WebsocketConsumer
from djangoProject.game.player import Player

BOARD_COLS = 20
BOARD_ROWS = 20


class State:
    def __init__(self, p1, p2, ROWS=20, COLS=20):
        BOARD_ROWS = ROWS
        BOARD_COLS = COLS
        # self.show_status = show_status
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1
        # chatConsumer = ChatConsumer

        # get unique hash of current board state

    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def retrievedReward(self, symbol, elementes):
        award = 0
        for i in elementes:
            if i == symbol:
                award += 0.05
            elif i == 0:
                award += 0.025
            else:
                break
        return award

    def rewardNew(self, symbol, x, y):
        award = 0
        award += self.retrievedReward(symbol, self.board[x, y - 5:y + 5])
        award += self.retrievedReward(symbol, self.board[x - 5:x + 5, y])
        award += self.retrievedReward(symbol,
                                      [self.board[x - 4:x - 3, y - 4:y - 3], self.board[x - 3:x - 2, y - 3:y - 2],
                                       self.board[x - 2:x - 1, y - 2:y - 1],
                                       self.board[x - 1:x, y - 1:y], self.board[x, y],
                                       self.board[x + 1:x + 2, y + 1:y + 2],
                                       self.board[x + 2:x + 3, y + 2:y + 2], self.board[x + 3:x + 4, y + 3:y + 3],
                                       self.board[x + 4:x + 5, y + 4:y + 6]])
        # award += self.retrievedReward(symbol,
        #                               [self.board[x + 4, y - 4], self.board[x + 3, y - 3], self.board[x + 2, y - 2],
        #                                self.board[x + 1, y - 1], self.board[x, y], self.board[x - 1, y + 1],
        #                                self.board[x - 2, y + 2], self.board[x - 3, y + 3],
        #                                self.board[x - 4, y + 4]])
        return award

    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if sum(self.board[i, j:j + 5]) == 5:
                    self.isEnd = True
                    return 1, None
                if sum(self.board[i, j:j + 5]) == -5:
                    self.isEnd = True
                    return -1, None
        # col
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):

                if sum(self.board[i:i + 5, j]) == 5:
                    self.isEnd = True
                    return 1, None
                if sum(self.board[i:i + 5, j]) == -5:
                    self.isEnd = True
                    return -1, None
        # diagonal1
        for i in range(BOARD_COLS - 5):
            for j in range(BOARD_ROWS - 5):
                if (self.board[j, i] + self.board[j + 1, i + 1] + self.board[j + 2, i + 2] + self.board[
                    j + 3, i + 3] +
                    self.board[j + 4, i + 4]) == 5:
                    self.isEnd = True
                    return 1, None
                if (self.board[j, i] + self.board[j + 1, i + 1] + self.board[j + 2, i + 2] + self.board[
                    j + 3, i + 3] +
                    self.board[j + 4, i + 4]) == -5:
                    self.isEnd = True
                    return -1, None
        # diagonal2
        for i in range(4, BOARD_COLS):
            for j in range(BOARD_ROWS - 5):
                if (self.board[j, i] + self.board[j + 1, i - 1] + self.board[j + 2, i - 2] + self.board[
                    j + 3, i - 3] +
                    self.board[j + 4, i - 4]) == 5:
                    self.isEnd14 = True
                    return 1, None
                if (self.board[j, i] + self.board[j + 1, i - 1] + self.board[j + 2, i - 2] + self.board[
                    j + 3, i - 3] +
                    self.board[j + 4, i - 4]) == -5:
                    self.isEnd = True
                    return -1, None
        # # diagonal1
        # for i in range(BOARD_COLS-5):
        #     for j in range(4,BOARD_ROWS):
        #         if sum(self.board[:j, i],self.board[:j-1, i+1],self.board[:j-2, i+2],self.board[:j-3, i+3],self.board[:j-4, i+4]) == 5:
        #             self.isEnd = True
        #             return 1
        #         if sum(self.board[:j, i],self.board[:j-1, i+1],self.board[:j-2, i+2],self.board[:j-3, i+3],self.board[:j-4, i+4]) == -5:
        #             self.isEnd = True
        #             return -1
        # diagonal
        # diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        # diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        # diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        #
        # if diag_sum == 3:
        #     self.isEnd = True
        #     if diag_sum1 == 3 or diag_sum2 == 3:
        #         return 1
        #     else:
        #         return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0, None
        # not end
        self.isEnd = False
        return None, None

    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if sum(self.board[i, j:j + 5]) == 5:
                    self.isEnd = True
                    return 1
                if sum(self.board[i, j:j + 5]) == -5:
                    self.isEnd = True
                    return -1
        # col
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):

                if sum(self.board[i:i + 5, j]) == 5:
                    self.isEnd = True
                    return 1
                if sum(self.board[i:i + 5, j]) == -5:
                    self.isEnd = True
                    return -1
        # diagonal1
        for i in range(BOARD_COLS - 5):
            for j in range(BOARD_ROWS - 5):
                if (self.board[j, i] + self.board[j + 1, i + 1] + self.board[j + 2, i + 2] + self.board[
                    j + 3, i + 3] +
                    self.board[j + 4, i + 4]) == 5:
                    self.isEnd = True
                    return 1
                if (self.board[j, i] + self.board[j + 1, i + 1] + self.board[j + 2, i + 2] + self.board[
                    j + 3, i + 3] +
                    self.board[j + 4, i + 4]) == -5:
                    self.isEnd = True
                    return -1
        # diagonal2
        for i in range(4, BOARD_COLS):
            for j in range(BOARD_ROWS - 5):
                if (self.board[j, i] + self.board[j + 1, i - 1] + self.board[j + 2, i - 2] + self.board[
                    j + 3, i - 3] +
                    self.board[j + 4, i - 4]) == 5:
                    self.isEnd14 = True
                    return 1
                if (self.board[j, i] + self.board[j + 1, i - 1] + self.board[j + 2, i - 2] + self.board[
                    j + 3, i - 3] +
                    self.board[j + 4, i - 4]) == -5:
                    self.isEnd = True
                    return -1
        # # diagonal1
        # for i in range(BOARD_COLS-5):
        #     for j in range(4,BOARD_ROWS):
        #         if sum(self.board[:j, i],self.board[:j-1, i+1],self.board[:j-2, i+2],self.board[:j-3, i+3],self.board[:j-4, i+4]) == 5:
        #             self.isEnd = True
        #             return 1
        #         if sum(self.board[:j, i],self.board[:j-1, i+1],self.board[:j-2, i+2],self.board[:j-3, i+3],self.board[:j-4, i+4]) == -5:
        #             self.isEnd = True
        #             return -1
        # diagonal
        # diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        # diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        # diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        #
        # if diag_sum == 3:
        #     self.isEnd = True
        #     if diag_sum1 == 3 or diag_sum2 == 3:
        #         return 1
        #     else:
        #         return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
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
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def playtoLearn(self, rounds=100, x=None):
        # for i in range(rounds):
        #     if i % 1 == 0:
        #     print("Rounds {}".format(i))
        print("Rounds {}", rounds)
        while not self.isEnd:
            # Player 1
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            # take action and upate board state
            self.updateState(p1_action)
            comunicationWS.send(x, text_data=json.dumps({"action": p1_action, "symbol": -1}))
            board_hash = self.getHash()
            self.p1.addState(board_hash)
            # check board status if it is end
            win = self.winner()
            if win is not None:
                print(win)
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
                p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p2_action)
                self.p1.feedReward(0)
                print(self.rewardNew(1, p2_action[0], p2_action[1]))
                # print(self.rewardNew(10, 10))
                comunicationWS.send(x, text_data=json.dumps({"action": p2_action, "symbol": 1}))
                board_hash = self.getHash()
                self.p2.addState(board_hash)

                win = self.winner()
                if win is not None:
                    print(win)
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
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
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

    # def showBoardOnline(self,x):
    #     self.show_status.send(x)
    def showBoard(self):
        # self.show_status.send()
        # print( self.show_status.send())
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            # print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')


class ComunicationWS(WebsocketConsumer):

    def connect(self):
        self.accept()
        print("connectAA")

    def disconnect(self, close_code):
        print("disconnectBB")
        pass

    def receive(self, text_data=None, bytes_data=None):
        text_data_json = json.loads(text_data)
        try:
            p1.loadPolicy("policy_p1")
            p2.loadPolicy("policy_p2")
            print("load policy")
        except:
            print("no policy")
        stan.playtoLearn(1, self)
        p1.savePolicy()
        p2.savePolicy()
        print("save policy")


comunicationWS = ComunicationWS

p1 = Player("p1")
p2 = Player("p2")

stan = State(p1, p2)
