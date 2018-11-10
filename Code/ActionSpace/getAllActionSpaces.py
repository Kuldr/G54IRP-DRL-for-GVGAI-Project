import gym
import gym_gvgai
import json
import datetime
import os
import sys

# Get and print the start time
startTime = datetime.datetime.now()
print("Start Time: %s" % startTime.time())

# Get all of the GVGAI Environments w/o ghostbuster as it crashes
envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai') and not env.id.startswith('gvgai-ghostbuster') ]

# Create an list results
actionsResults = []

# Test every environment
for i, env in enumerate(envs):

    # Make the environment
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
    actionsResults.append(actionsResult)

    print("%d/%d Environments Evalutated, Elapsed Time: %s"
        % ((i+1), len(envs), datetime.datetime.now() - startTime))

# Save as a json file
with open("actionSpaceResults.json", "w") as outfile:
    json.dump(actionsResults, outfile, indent = "\t")

# Print the running time
endTime = datetime.datetime.now()
print("Duration: %s" % (endTime - startTime))
