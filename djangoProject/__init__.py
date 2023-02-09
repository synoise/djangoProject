import json

import numpy
from channels.generic.websocket import WebsocketConsumer
from . import neural


class ChatConsumer(WebsocketConsumer):

    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data=None, bytes_data=None):
        text_data_json = json.loads(text_data)
        award = text_data_json["award"]
        gamer = text_data_json["gamer"]
        state = numpy.array(text_data_json["message"], dtype=numpy.int8)
        winner = text_data_json["winner"]
        aivsai = text_data_json["aivsai"]
        # if aivsai:
        #     self.send(text_data=json.dumps({"message": neural.getMove(state, gamer, award, winner, aivsai)}))
        # print("winner", winner, "| gamer", gamer, "| award", award)
        # print("state", state)
        print(aivsai)
        print(aivsai)
        if not winner:
            self.send(text_data=json.dumps( neural.addMove(state, gamer, award, winner, aivsai)))
        else:
            neural.resetState()