import gym
import gym_gvgai
import json
import datetime
import os
import sys
import multiprocessing

def getActionSpaceInfo(env):
    # Make the environment while supressing the output to terminal of the server
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    env = gym.make(env)
    sys.stdout.close()
    sys.stdout = original_stdout

    # Wrap the relevant info in a dictionary and append it to the results list
    actionsResult = {"Game" : env.unwrapped.game,
                     "Level" : env.unwrapped.lvl,
                     "Version" : env.unwrapped.version,
                     "Actions" : env.unwrapped.actions}
    return actionsResult

# Get and print the start time
startTime = datetime.datetime.now()
print("Start Time: %s" % startTime.time())

# Get all of the GVGAI Environments w/o ghostbuster as it crashes
envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai') and not env.id.startswith('gvgai-ghostbuster') ]

# Create an results list
actionsResults = []
# Set up the multiprocessing pool
cpus = multiprocessing.cpu_count()
with multiprocessing.Pool(cpus) as pool:
    # Get the data for every environment
    for i, result in enumerate(pool.imap_unordered(getActionSpaceInfo, envs)):
        # Add the result to the list of results
        actionsResults.append(result)
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
with open("actionSpaceResults.json", "w") as outfile:
    json.dump(actionsResults, outfile, indent = "\t")

# Print the running time
endTime = datetime.datetime.now()
print("Duration: %s" % (endTime - startTime))
