import cv2

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont


class AngleVisualizer:
    def __init__(self, exp, tracks):
        self._exp = exp
        self._tracks = tracks

    def visualize(self):
        if 'chemPos' not in self._exp._pointsOfInterest or \
                'chemRad' not in self._exp._pointsOfInterest:
            print('Error: cant find chem point.')
            return

        # First calculate the interesting frames
        allFrames = np.ravel(np.array([t._trackFrames for t in self._tracks]))
        relevantFrames = range(np.min(allFrames), np.max(allFrames))

        cap = self._exp._cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, np.min(allFrames))

        for i,frame in enumerate(relevantFrames):
            _, currentFrame = self._cap.read()

            for t in self._tracks:
                if frame in t._trackFrames[0:-1]:
                    currentPos = t.getPos()
                    currentStep = t.getStep(frame)

                    curIm = Image.fromarray(currentFrame).convert('RGB')
                    curImDraw = ImageDraw.Draw(curIm)

                    # Calculate line end point.
                    endPoint = currentPos + (currentStep / np.linalg.norm(currentStep )) * 25
                    curImDraw.line((currentPos[0], currentPos[1], endPoint[0], endPoint[1]))
                    #endPoint[0] = np.maximum(endPoint[0], 0)
                    #endPoint[1] = np.maximum(endPoint[1], 0)

                    #endPoint[0] = np.minimum(endPoint[0], currentFrame.shape[0])
                    #endPoint[1] = np.minimum(endPoint[1], currentFrame.shape[1])
                    plt.imshow(curIm)


















