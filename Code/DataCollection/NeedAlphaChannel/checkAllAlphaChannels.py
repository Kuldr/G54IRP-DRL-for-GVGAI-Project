import gym
import gym_gvgai
import json
import datetime
import os
import sys
import multiprocessing

from scipy.misc import imsave

def getAlphaUsage(env):
    # Make the environment while supressing the output to terminal of the server
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    env = gym.make(env)
    sys.stdout.close()
    sys.stdout = original_stdout

    # Wrap the relevant info in a dictionary and append it to the results list
    stateObs = env.reset()
    (y,x,c) = stateObs.shape
    alphaChannel = stateObs[:,:,3]
    alphaUsed = not bool(x*y*255 == alphaChannel.sum())

    # If alpha channel is used save the image and the alpha channel
    if alphaUsed:
        saveString = "Images/" + env.unwrapped.game + "-" + str(env.unwrapped.lvl)
        imsave(saveString+"-Alpha.png", alphaChannel)
        imsave(saveString+"-RGB.png", stateObs[:,:,:3])
        imsave(saveString+"-Full.png", stateObs)

    result = {"Game"      : env.unwrapped.game,
              "Level"     : env.unwrapped.lvl,
              "Version"   : env.unwrapped.version,
              "AlphaUsed" : alphaUsed,
              "Area*255"  : int(x*y*255),
              "AlphaSum"  : int(alphaChannel.sum())}
    return result

# Get and print the start time
startTime = datetime.datetime.now()
print("Start Time: %s" % startTime.time())

# Get all of the GVGAI Environments w/o ghostbuster as it crashes
envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai') and not env.id.startswith('gvgai-ghostbuster') ]

# Create an results list
alphaResults = []
# Set up the multiprocessing pool
cpus = multiprocessing.cpu_count()
with multiprocessing.Pool(cpus) as pool:
    # Get the data for every environment
    for i, result in enumerate(pool.imap_unordered(getAlphaUsage, envs)):
        # Add the result to the list of results
        alphaResults.append(result)
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
with open("alphaChannelResults.json", "w") as outfile:
    json.dump(alphaResults, outfile, indent = "\t")

# Print the running time
endTime = datetime.datetime.now()
print("Duration: %s" % (endTime - startTime))
