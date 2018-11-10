import gym
import gym_gvgai
import json
import datetime
import os
import sys
import multiprocessing

def multi(env):
    return actionsResult

# Get and print the start time
startTime = datetime.datetime.now()
print("Start Time: %s" % startTime.time())

# Get all of the GVGAI Environments w/o ghostbuster as it crashes
envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai') and not env.id.startswith('gvgai-ghostbuster') ]

# Create an results list
gameResults = []
# Set up the multiprocessing pool
cpus = multiprocessing.cpu_count()
with multiprocessing.Pool(cpus) as pool:
    # Get the data for every environment
    for i, result in enumerate(pool.imap_unordered(multi, envs)):
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
