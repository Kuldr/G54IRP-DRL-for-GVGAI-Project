from stable_baselines import A2C
from PIL import Image
import numpy as np

from Test.CustomPolicy import CustomPolicy

model = None

class Agent():
    def __init__(self):
        global model
        self.name = "modelAgentKuldr"
        model = A2C.load("BIGRUN/a2c-big-run-Final", policy=CustomPolicy)

    def act(self, stateObs, _):
        global model
        actionID, _ = model.predict(self.transfromFrame(stateObs))
        return actionID

    # TODO THIS NEEDS TO COME FROM THE ENVWRAPPER 
    def transfromFrame(self, frame):
        frame = frame[:,:,:3]
        # Convert to PIL Image and resize before converting back and adding to new array
        frameIm = Image.fromarray(frame)
        frameIm = frameIm.resize((260,130))
        frame = np.asarray(frameIm)
        return frame
