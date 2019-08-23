from stable_baselines import A2C

from CustomPolicies import NatureCNN
from modelHelperFunctions import transformFrame

model = None

class Agent():
    def __init__(self):
        global model
        self.name = "modelAgentKuldr"
        model = A2C.load("models/Warp300x300/Aliens-Warp300x300-lvl1-1-Final", policy=NatureCNN)

    def act(self, stateObs, _):
        global model
        actionID, _ = model.predict(transformFrame(stateObs, x=300, y=300))
        return actionID
