# Initial code for keyboard controller taken from the Open AI Gym framework
#   https://github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py

import sys, gym, time
import gym_gvgai

# Set time between frames (and actions) to the same as the agents will get
MILLISECONDS_PER_FRAME = 40

# Load the game here, can be changed to different games
env = gym.make('gvgai-aliens-lvl0-v0')

# Creates 2 dictionaries keys and actions which have a mapping for keys to actions and actions to action_ids
POSSIBLE_ACTIONS = ["ACTION_NIL", "ACTION_USE", "ACTION_LEFT", "ACTION_RIGHT", "ACTION_DOWN", "ACTION_UP"]
# Get the dictionary of actions in the env
actions = dict(zip(env.unwrapped.actions, [i for i in range(env.unwrapped.action_space.n)]))
# Add the other actions to the dictionary if required
#       This is to make sure that all inputs have a valid action
for x in POSSIBLE_ACTIONS:
    if x not in actions:
        actions[x] = actions["ACTION_NIL"]
#       119: W            97: A              115: S              100: D               106: J
KEYS = {119: "ACTION_UP", 97: "ACTION_LEFT", 115: "ACTION_DOWN", 100: "ACTION_RIGHT", 106: "ACTION_USE"}
# Set the NIL ACTION
NIL_ACTION = actions["ACTION_NIL"]

human_agent_action = NIL_ACTION
human_wants_restart = False
human_sets_pause = True # Starts the game paused needs to press space to start

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key == 0xff0d: # If return is pressed restart
        human_wants_restart = True
    elif key == 32:  # If space is press pause
        if human_sets_pause:
            print("UNPAUSED")
        else:
            print("PAUSED")
        human_sets_pause = not human_sets_pause
    elif key in KEYS: # If a key is pressed then perform the corisponding action
        human_agent_action = actions[KEYS[key]]

def key_release(key, mod):
    global human_agent_action
    if key in KEYS: # If a key is released return to NIL action
        human_agent_action = NIL_ACTION

# Open the window and set the relevant key actions to the functions
env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    env.reset()
    env.render()
    score = 0
    timestep = 0
    while True:
        actionToTake = human_agent_action
        _, reward, isOver, info = env.step(actionToTake)
        score += reward
        if reward != 0: # If gained a reward print out score and reward
            print("Score = %d, Reward = %d" % (score, reward))
        window_still_open = env.render()
        if not window_still_open:
            return window_still_open
        if isOver:
            print("\n\n-----FINISHD-----")
            print("Game Finished")
            print("Timesteps = %d" % timestep)
            print("Score = %d" % score)
            print("Winner = " + info['winner'])
            print("-----------------\n")
            break
        if human_wants_restart:
            print("\n\n-----RESTART-----")
            print("Player Reset")
            print("Timesteps = %d" % timestep)
            print("Score = %d" % score)
            print("-----------------\n")
            break
        while human_sets_pause:
            env.render()
        timestep += 1
        time.sleep(MILLISECONDS_PER_FRAME*0.001)
    # Pause game for the start of the next game
    human_sets_pause = not human_sets_pause
    print("The game starts paused so press SPACE to unpause\n\n")
    rollout(env)

print("\n\n-----CONTROLS-----")
print("WASD for Directional control")
print("J for use/action")
print("SPACE for pause")
print("RETURN for restart")
print("------------------\n")
# Tell the player that the game is initially paused
print("The game starts paused so press SPACE to unpause\n\n")

while True:
    window_still_open = rollout(env)
    if not window_still_open:
        break
