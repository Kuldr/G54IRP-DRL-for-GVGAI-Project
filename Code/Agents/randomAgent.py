from random import choice

# Constants
POTENTIAL_ACTIONS = ["ACTION_NIL", "ACTION_USE", "ACTION_LEFT",
                    "ACTION_RIGHT", "ACTION_DOWN", "ACTION_UP"]
# Gobal Variables
actionsDict = None

class Agent():
    # Initialisation of the agent
    # should be called everytime the environment is reset
    def __init__(self):
        self.name = "randomAgentKuldr"
        global actionsDict
        actionsDict = None

    # Ask the agent for an action
    # passing in a state observation and a list of valid actions
    def act(self, stateObs, actions):
        # check if the valid action dictionary has been created if not make it
        global actionsDict
        if actionsDict == None:
            actionsDict = dict(zip(actions, [i for i in range(len(actions))]))

        # Select a random action from the list of all potential actions
        # Keep trying until a valid action is selected
        actionID = None
        while actionID == None:
            x = choice(POTENTIAL_ACTIONS)
            if x in actionsDict:
                actionID = actionsDict[x]

        # Return the chosen action 
        return actionID
