from PIL import Image
import numpy as np

def transformFrame(frame, x, y):
    frame = frame[:,:,:3]
    # Convert to PIL Image and resize before converting back and adding to new array
    frameIm = Image.fromarray(frame)
    frameIm = frameIm.resize((x,y))
    frame = np.asarray(frameIm)
    return frame

def transformBatch(batchObs, b, x, y, c):
    # Resize transformation
    resizedBatchObs = np.empty((b, y, x, c), dtype=np.uint8) # Create output array
    for i, frame in enumerate(batchObs[:]):
        frame = self.transfromFrame(frame, x, y)
        resizedBatchObs[i] = frame

    # Name output and return
    observation = resizedBatchObs
    return observation
