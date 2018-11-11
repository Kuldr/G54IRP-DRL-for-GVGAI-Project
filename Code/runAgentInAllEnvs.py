import gym
import gym_gvgai
import json
import datetime
import os
import sys
import multiprocessing
import random

GAME_TICKS = 2000
RUNS_PER_ENV = 5
POSSIBLE_ACTIONS = ["ACTION_NIL", "ACTION_USE", "ACTION_LEFT", "ACTION_RIGHT", "ACTION_DOWN", "ACTION_UP"]

def runAgentInEnvironment(env):
    # Make the environment while supressing the output to terminal of the server
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    env = gym.make(env)
    sys.stdout.close()
    sys.stdout = original_stdout

    # Create a list to return all the results in
    results = []

    # Get the dictionary of actions in the env with their ids
    actions = dict(zip(env.unwrapped.actions, [i for i in range(env.unwrapped.action_space.n)]))

    # Run each environment a number of times to get multiple samples per environment
    #   This should help avoid random flukes
    for y in range(RUNS_PER_ENV):
        # Reset the enivronment and score
        env.reset()
        score = 0

        # Run game for a number game ticks
        for i in range(GAME_TICKS):
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

            if isOver:
                # Add the result to the result list
                result = {"Game" : env.unwrapped.game,
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
envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai') and not env.id.startswith('gvgai-ghostbuster') and not env.id.startswith('gvgai-killBillVol1') ]

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
with open("randomAgentResults.json", "w") as outfile:
    json.dump(gameResults, outfile, indent = "\t")

# Print the running time
endTime = datetime.datetime.now()
print("Duration: %s" % (endTime - startTime))
