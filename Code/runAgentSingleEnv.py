import gym
import gym_gvgai
import random

import Agents.randomAgent as Agent

RENDER_TO_SCREEN = True
GAME_TICKS = 2000

# Get all of the GVGAI Environments w/o ghostbuster, and killBill as they crash (in different ways)
envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai') and not env.id.startswith('gvgai-ghostbuster') and not env.id.startswith('gvgai-killBillVol1') ]

# Create and reset the enivronment
# env = gym.make('gvgai-aliens-lvl0-v0')
env = gym.make(random.choice(envs))
stateObs = env.reset()
# Create and Intitialise the agent
agent = Agent.Agent()

# Get the list of actions in the env to pass to the agent
actions = env.unwrapped.actions

# Set the score counter to 0
score = 0
# Run game for a number game ticks
for i in range(GAME_TICKS):
    # Render the game to the screen if option is selected
    if RENDER_TO_SCREEN:
        env.render()

    # Ask Agent to give an action action based on trained policy
    actionID = agent.act(stateObs, actions)

    # Perform the action choosen and get the info from the environment
    stateObs, reward, isOver, info = env.step(actionID)
    # Update the cumilative score based upon the reward given
    score += reward

    # Print the results of the action performed
    print("Action " + str(actionID) + " played at game tick " + str(i+1) + ", reward=" + str(reward) + ", new score=" + str(score))
    if isOver:
        print("Game over at game tick " + str(i+1) + " with player " + info['winner'] + ". Score = " + str(score) + " at Env " + str(env))
        break

# Close the environment render after game is finished
if RENDER_TO_SCREEN:
    env.close()
