import gym
import gym_gvgai
import random

from datetime import datetime

# import Agents.randomAgent as Agent
import experimentAgent as Agent

RENDER_TO_SCREEN = True
REPEAT_ON_OVER = True
GAME_TICKS = 2000
FRAME_TIME_MS = 40

# Get all of the GVGAI Environments w/o ghostbuster, and killBill as they crash (in different ways)
# envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai') and not env.id.startswith('gvgai-ghostbuster') and not env.id.startswith('gvgai-killBillVol1') ]
envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai-missilecommand') or env.id.startswith('gvgai-boulderdash') or env.id.startswith('gvgai-aliens')]
# envs = ['gvgai-boulderdash-lvl1-v0']

# Create and reset the enivronment
# env = gym.make('gvgai-missilecommand-lvl1-v0')
env = gym.make(random.choice(envs))
stateObs = env.reset()
# Create and Intitialise the agent
initStart = datetime.now()
agent = Agent.Agent()
initEnd = datetime.now()
print("Initialise Time= " + str((initEnd-initStart).total_seconds()*1000))

# Get the list of actions in the env to pass to the agent
actions = env.unwrapped.actions

# Set the score counter to 0
score = 0

while REPEAT_ON_OVER:
    # Run game for a number game ticks
    for i in range(GAME_TICKS):
        frameStart = datetime.now()
        # Render the game to the screen if option is selected
        if RENDER_TO_SCREEN:
            env.render()

        # Ask Agent to give an action action based on trained policy
        infStart = datetime.now()
        actionID = agent.act(stateObs, actions)
        infEnd = datetime.now()

        # Perform the action choosen and get the info from the environment
        stateObs, reward, isOver, info = env.step(actionID)
        # Update the cumilative score based upon the reward given
        score += reward

        # Print the results of the action performed
        print("Action " + str(actionID) + " played at game tick " + str(i+1) + ", reward=" + str(reward) + ", new score=" + str(score) + ", inference time=" + str((infEnd-infStart).total_seconds()*1000))
        if isOver:
            print("Game over at game tick " + str(i+1) + " with player " + info['winner'] + ". Score = " + str(score) + " at Env " + str(env))
            if REPEAT_ON_OVER:
                score = 0
                stateObs = env.reset()
            break

        frameEnd = datetime.now()
        delta = frameEnd - frameStart
        while delta.total_seconds()*1000 < FRAME_TIME_MS:
            frameEnd = datetime.now()
            delta = frameEnd - frameStart

# Close the environment render after game is finished
if RENDER_TO_SCREEN:
    env.close()
