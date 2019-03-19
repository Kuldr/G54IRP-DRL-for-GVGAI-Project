from PIL import Image
import numpy as np

def transformFrame(frame, x, y):
    frame = frame[:,:,:3]
    # Convert to PIL Image and resize before converting back and adding to new array
    frameIm = Image.fromarray(frame)
    frameIm = frameIm.resize((x,y))
    frame = np.asarray(frameIm)
    return frame

# NOTE: THIS DOESN'T SCALE -IVE REWARDS BASED ON MIN SCORE BUT ON MAX SCORE
def normalizeReward(reward, envString):
    newReward = reward / MAX_SCORES[envString]
    print("Reward: " + str(reward) + " | newReward: " + str(newReward) + " | scale: " + str(MAX_SCORES[envString]) + " | env: " + envString)
    return newReward

MAX_SCORES = {"<TimeLimit<GVGAI_Env<gvgai-boulderdash-lvl0-v0>>>" : 22,
		      "<TimeLimit<GVGAI_Env<gvgai-boulderdash-lvl1-v0>>>" : 22,
              "<TimeLimit<GVGAI_Env<gvgai-boulderdash-lvl2-v0>>>" : 22,
              "<TimeLimit<GVGAI_Env<gvgai-boulderdash-lvl3-v0>>>" : 22,
              "<TimeLimit<GVGAI_Env<gvgai-boulderdash-lvl4-v0>>>" : 22,
              "<TimeLimit<GVGAI_Env<gvgai-missilecommand-lvl0-v0>>>" : 8,
              "<TimeLimit<GVGAI_Env<gvgai-missilecommand-lvl1-v0>>>" : 16,
              "<TimeLimit<GVGAI_Env<gvgai-missilecommand-lvl2-v0>>>" : 8,
              "<TimeLimit<GVGAI_Env<gvgai-missilecommand-lvl3-v0>>>" : 8,
              "<TimeLimit<GVGAI_Env<gvgai-missilecommand-lvl4-v0>>>" : 14,
              "<TimeLimit<GVGAI_Env<gvgai-aliens-lvl1-v0>>>" : 1}
