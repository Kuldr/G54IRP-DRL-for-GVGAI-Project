from stable_baselines import A2C

from CustomPolicies import NatureCNN, ONet
from modelHelperFunctions import transformFrame

model = None

class Agent():
    def __init__(self):
        global model
        self.name = "Nature-Final"
        model = A2C.load("models/Final/AliensBoulderdashMissileCommand-NatureCNN-1-Final.pkl")#, policy=NatureCNN)

    def act(self, stateObs, actions):
        print(actions)
        global model
        actionID, _ = model.predict(transformFrame(stateObs, x=110, y=110))

        if actionID >= len(actions):
            actionID = 0

        return actionID
