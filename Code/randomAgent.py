import gym
import gym_gvgai
import random

RENDER_TO_SCREEN = False
GAME_TICKS = 2000
POSSIBLE_ACTIONS = ["ACTION_NIL", "ACTION_USE", "ACTION_LEFT", "ACTION_RIGHT", "ACTION_DOWN", "ACTION_UP"]

# Get all of the GVGAI Environments w/o ghostbuster as it crashes
envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai') and not env.id.startswith('gvgai-ghostbuster') ]

# Create and reset the enivronment
# env = gym.make('gvgai-aliens-lvl0-v0')
env = gym.make(random.choice(envs))
env.reset()

# Get the dictionary of actions in the env with their ids
actions = dict(zip(env.unwrapped.actions, [i for i in range(env.unwrapped.action_space.n)]))

# Set the score counter to 0
score = 0
# Run game for a number game ticks
for i in range(GAME_TICKS):
    # Render the game to the screen if option is selected
    if RENDER_TO_SCREEN:
        env.render()

    # Select a random action from all possible actions and check its valid
    action = None
    while action == None:
        x = random.choice(POSSIBLE_ACTIONS)
        if x in actions:
            action = actions[x]

    # Perform the action choosen and get the info from the environment
    state, reward, isOver, info = env.step(action)
    # Update the cumilative score based upon the reward given
    score += reward

    # Print the results of the action performed
    print("Action " + str(action) + " played at game tick " + str(i+1) + ", reward=" + str(reward) + ", new score=" + str(score))
    if isOver:
        print("Game over at game tick " + str(i+1) + " with player " + info['winner'] + ". Score = " + str(score) + " at Env " + str(env))
        break

# Close the environment render after game is finished
if RENDER_TO_SCREEN:
    env.close()
