import gym
import gym_gvgai
import json
import datetime
import os
import sys
import multiprocessing
import random

import experimentAgent as Agent

GAME_TICKS = 2000
RUNS_PER_ENV = 25

def runAgentInEnvironment(env):
    # Make the environment while supressing the output to terminal of the server
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    env = gym.make(env)
    sys.stdout.close()
    sys.stdout = original_stdout
    # Create and Intitialise the agent
    agent = Agent.Agent()
    # Create a list to return all the results in
    results = []
    # Get the list of actions in the env to pass to the agent
    actions = env.unwrapped.actions

    # Run each environment a number of times to get multiple samples per environment
    #   This should help avoid random flukes
    for y in range(RUNS_PER_ENV):
        # Reset the enivronment and score
        stateObs = env.reset()
        score = 0

        # Run game for a number game ticks
        for i in range(GAME_TICKS):
            # Ask Agent to give an action action based on trained policy
            actionID = agent.act(stateObs, actions)

            # Perform the action choosen and get the info from the environment
            stateObs, reward, isOver, info = env.step(actionID)
            # Update the cumilative score based upon the reward given
            score += reward

            # End the game and collect results if game is over
            if isOver:
                # Add the result to the result list
                result = {"Name" : agent.name,
                          "Game" : env.unwrapped.game,
                          "Level" : env.unwrapped.lvl,
                          "Version" : env.unwrapped.version,
                          "Score" : score,
                          "Winner" : info['winner'],
                          "GameTick" : (i+1)}
                results.append(result)
                break

    return results

# Get and print the start time
startTime = datetime.datetime.now()
print("Start Time: %s" % startTime.time())

# Get all of the GVGAI Environments w/o ghostbuster, and killBill as they crash (in different ways)
# envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai') and not env.id.startswith('gvgai-ghostbuster') and not env.id.startswith('gvgai-killBillVol1') ]
envs = ["gvgai-aliens-lvl1-v0"] * multiprocessing.cpu_count()

# Create an results list
gameResults = []
# Set up the multiprocessing pool
cpus = multiprocessing.cpu_count()
with multiprocessing.Pool(cpus) as pool:
    # Get the data for every environment
    for i, result in enumerate(pool.imap_unordered(runAgentInEnvironment, envs)):
        # Add the result to the list of results
        gameResults.append(result)
        # Print out progression info
        i += 1
        elapsedTime = datetime.datetime.now() - startTime
        remainingTime = elapsedTime/i * (len(envs)-i) # Time taken per item * items remaining
        print("%3d/%3d Environments Evalutated" % (i, len(envs)))
        print(" "*8 + "Elapsed Time: %s" % elapsedTime)
        print(" "*8 + "Time Left   : %s" % remainingTime)

    # Reallign the processes before saving results
    pool.close()
    pool.join()

# Save as a json file
with open('experimentResults.json') as data_file:
    results = json.load(data_file)
    for r in results:
        gameResults.append(r)
with open("experimentResults.json", "w") as outfile:
    json.dump(gameResults, outfile, indent = "\t")

# Print the running time
endTime = datetime.datetime.now()
print("Duration: %s" % (endTime - startTime))
