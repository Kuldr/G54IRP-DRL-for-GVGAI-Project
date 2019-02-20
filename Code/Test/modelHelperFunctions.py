from PIL import Image
import numpy as np

def transformFrame(frame, x, y):
    frame = frame[:,:,:3]
    # Convert to PIL Image and resize before converting back and adding to new array
    frameIm = Image.fromarray(frame)
    frameIm = frameIm.resize((x,y))
    frame = np.asarray(frameIm)
    return frame
