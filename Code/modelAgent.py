from stable_baselines import A2C
from PIL import Image
import numpy as np

from Test.CustomPolicy import CustomPolicy
from Test.modelHelperFunctions import transformFrame

model = None

class Agent():
    def __init__(self):
        global model
        self.name = "modelAgentKuldr"
        model = A2C.load("BIGRUN/a2c-big-run-Final", policy=CustomPolicy)

    def act(self, stateObs, _):
        global model
        actionID, _ = model.predict(transformFrame(stateObs, x=260, y=130))
        return actionID
