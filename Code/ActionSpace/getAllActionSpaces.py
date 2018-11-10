import gym
import gym_gvgai
import json

# Get all of the GVGAI Environments w/o ghostbuster as it crashes
envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai') and not env.id.startswith('gvgai-ghostbuster') ]

# Create an list results
actionsResults = []

# Test every environment
for x in envs:

    # Make the environment
    env = gym.make(x)

    # Wrap the relevant info in a dictionary and append it to the results list
    actionsResult = {"Game" : env.unwrapped.game,
                     "Level" : env.unwrapped.lvl,
                     "Version" : env.unwrapped.version,
                     "Actions" : env.unwrapped.actions}
    actionsResults.append(actionsResult)

# Save as a json file
with open("actionSpaceResults.json", "w") as outfile:
    json.dump(actionsResults, outfile, indent = "\t")
