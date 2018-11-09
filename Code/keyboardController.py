# Initial code for keyboard controller taken from the Open AI Gym framework
#   https://github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py

import sys, gym, time
import gym_gvgai

# Set time between frames (and actions) to the same as the agents will get
MILLISECONDS_PER_FRAME = 40

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

ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = True # Starts the game paused needs to press space to start

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: # If return is pressed restart
        human_wants_restart = True
    elif key==32:  # If space is press pause
        human_sets_pause = not human_sets_pause

    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

# Open the window and set the relevant key actions to the functions
env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open == False:
            return False
        if done:
            break
        if human_wants_restart:
            break
        while human_sets_pause:
            env.render()
            time.sleep(MILLISECONDS_PER_FRAME*0.001)
        time.sleep(MILLISECONDS_PER_FRAME*0.001)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

print("\n\n-----CONTROLS-----")
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("SPACE for pause")
print("RETURN for restart")
print("------------------\n")
# Tell the player that the game is initially paused
print("The game starts paused so press SPACE to unpause")

while 1:
    window_still_open = rollout(env)
    if window_still_open == False:
        break
