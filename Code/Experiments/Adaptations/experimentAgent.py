from stable_baselines import A2C

from CustomPolicies import NatureCNN
from modelHelperFunctions import transformFrame

model = None

class Agent():
    def __init__(self):
        global model
        self.name = "NatureCNN-1"
        model = A2C.load("finalModels/AliensBoulderdashMissileCommand-NatureCNN-1-Final", policy=NatureCNN)

    def act(self, stateObs, _):
        global model
        actionID, _ = model.predict(transformFrame(stateObs, x=110, y=110))

        if actionID >= 4:
            actionID = 0

        return actionID
