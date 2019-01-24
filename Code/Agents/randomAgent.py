from random import choice

# TODO: Comments

# Constants
POSSIBLE_ACTIONS = ["ACTION_NIL", "ACTION_USE", "ACTION_LEFT",
                    "ACTION_RIGHT", "ACTION_DOWN", "ACTION_UP"]

# Gobal Variables
actionsDict = None

class Agent():

    def __init__(self):
        self.name = "randomAgentKuldr"
        global actionsDict
        actionsDict = None

    def act(self, stateObs, actions):
        global actionsDict
        if actionsDict == None:
            actionsDict = dict(zip(actions, [i for i in range(len(actions))]))

        actionID = None
        while actionID == None:
            x = choice(POSSIBLE_ACTIONS)
            if x in actionsDict:
                actionID = actionsDict[x]

        return actionID
