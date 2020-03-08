#DEBUG
import sys
sys.path.append('/home/itskov/workspace/lab/DeepSemantic/WormSegmentation/')
#DEBUG

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties

import os.path as path
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont, ImageColor
import re
import cv2


from Behavior.General.ExpDir import ExpDir
from Behavior.Visualizers.RoiAnalysis import RoiAnalysis

'''
This class helps compare and visualize difference between
two experiments. Naturally, control and condition.
'''
class ExpPair:
    def __init__(self, firstExpPath, secExpPath, targetPath = None):
        self.FIRST_COLOR = "#3C65B7"
        self.SECOND_COLOR = "#00A99C"

        self._firstExp = np.load(firstExpPath)[0]
        self._secondExp = np.load(secExpPath)[0]

        self._firstExpDir = ExpDir(path.dirname(firstExpPath))
        self._secondExpDir = ExpDir(path.dirname(secExpPath))

        self._targetDir = path.dirname(firstExpPath)

        # Try to get the name of the condition from one of the experiment files.
        fn1 = path.basename(self._firstExpDir.getExpSegVid())
        fn2 = path.basename(self._secondExpDir.getExpSegVid())

        self._cond1 = re.search('.+-(.+)\.avi.*', fn1)[1]
        self._cond2 = re.search('.+-(.+)\.avi.*', fn2)[1]

        #DEBUG!!!
        #self._cond1 = 'Trained'
        #self._cond2 = 'Mock Trained'
        #DEBUG!!
        
        if targetPath is None:
            self._targetPath = path.dirname(firstExpPath)
        else:
            self._targetPath = targetPath




    def save(self):
        fileName = path.join(self._targetDir,'expsPair.npy')
        np.save(fileName, self)

    def alignImage(self, img):
        # First we roll the image to center the plate.
        rightBorder = np.min(np.where(img > 0)[1])
        leftBorder = img.shape[1] - np.max(np.where(img > 0)[1])
        allBorders = rightBorder + leftBorder
        correctBorder = np.floor(allBorders / 2)
        return int(correctBorder - rightBorder)


    def createPairVisualization(self, numberOfFrames=0, dpi=200):
        firstCap = cv2.VideoCapture(self._firstExpDir.getExpSegVid())
        secCap = cv2.VideoCapture(self._secondExpDir.getExpSegVid())

        firstMovieLength = int(firstCap.get(cv2.CAP_PROP_FRAME_COUNT))
        secondMovieLength = int(secCap.get(cv2.CAP_PROP_FRAME_COUNT))

        firstCap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        secCap.set(cv2.CAP_PROP_POS_FRAMES, 1)

        # Getting the minimum framecount of the two videos.
        frameLength = np.min((firstMovieLength, secondMovieLength))

        # Getting the requested frame number.
        frameLength = np.min((numberOfFrames, frameLength))

        # Trimming the experiments
        #self._firstExp.trimExperiment(numberOfFrames)
        #self._secondExp.trimExperiment(numberOfFrames)
        #

        # First, running the ROI analyses
        firstRoi = RoiAnalysis(self._firstExp)
        secondRoi = RoiAnalysis(self._secondExp)
        firstRoi.execute()
        secondRoi.execute()

        fig = plt.figure(facecolor='black')
        plt.style.use('dark_background')

        ax_vid = fig.add_subplot(2, 1, 1)
        ax_fig = fig.add_subplot(2, 1, 2)

        ax_vid.axis('off')
        im = None
        vertLine = None

        # The font for the ImageDraw.
        fnt = ImageFont.truetype("DejaVuSans-Bold.ttf", 96)

        firstAlign = 0
        secondAlign = 0

        def updateMovie(frameNum):
            fig.sca(ax_vid)
            nonlocal im
            nonlocal vertLine
            nonlocal firstAlign
            nonlocal secondAlign

            _, firstFrame = firstCap.read()
            _, secondFrame = secCap.read()

            if (frameNum == -1):
                firstAlign = self.alignImage(firstFrame)
                secondAlign = self.alignImage(secondFrame)

            firstFrame = np.roll(firstFrame, +firstAlign, axis=1)
            secondFrame = np.roll(secondFrame, +secondAlign, axis=1)

            firstImSeg = Image.fromarray(firstFrame)
            firstImRawDraw = ImageDraw.Draw(firstImSeg)
            firstImRawDraw.text((0, 0), self._cond1, font=fnt, fill=ImageColor.getrgb(self.FIRST_COLOR))

            countStr = "%d \ %d (%d%%)" % \
                       (int(firstRoi._results['arrived'][frameNum]),
                        int(firstRoi._results['wormCount']),
                        int((firstRoi._results['arrived'][frameNum] * 100) / firstRoi._results['wormCount']))

            firstImRawDraw.text((0, 120), countStr, font=fnt, fill=ImageColor.getrgb(self.FIRST_COLOR))


            firstChemoPos = np.fliplr(np.atleast_2d(self._firstExp._regionsOfInterest['endReg']['pos']))
            firstChemoPos  = np.ravel(firstChemoPos )
            firstChemoPos[0] += firstAlign
            firstRad = self._firstExp._regionsOfInterest['endReg']['rad']

            secChemoPos = np.fliplr(np.atleast_2d(self._secondExp._regionsOfInterest['endReg']['pos']))
            secChemoPos = np.ravel(secChemoPos)
            secChemoPos[0] += secondAlign
            secRad = self._firstExp._regionsOfInterest['endReg']['rad']



            secondImSeg = Image.fromarray(secondFrame)
            secondImRawDraw = ImageDraw.Draw(secondImSeg)
            secondImRawDraw.text((0, 0), self._cond2, font=fnt, fill=ImageColor.getrgb(self.SECOND_COLOR))


            countStr = "%d \ %d (%d%%)" % \
                       (int(secondRoi._results['arrived'][frameNum]),
                        int(secondRoi._results['wormCount']),
                        int((secondRoi._results['arrived'][frameNum] * 100) / secondRoi._results['wormCount']))

            secondImRawDraw.text((0, 120), countStr, font=fnt, fill=ImageColor.getrgb(self.SECOND_COLOR))


            width = 10
            for d in range(width):
                firstImRawDraw.arc((firstChemoPos[0] - (firstRad + d),
                                    firstChemoPos[1] - (firstRad + d),
                                    firstChemoPos[0] + (firstRad + d),
                                    firstChemoPos[1] + (firstRad + d)),
                              0,
                              360,
                              fill=ImageColor.getrgb(self.FIRST_COLOR))

                secondImRawDraw.arc((secChemoPos[0] - (secRad + d),
                                    secChemoPos[1] - (secRad + d),
                                    secChemoPos[0] + (secRad + d),
                                    secChemoPos[1] + (secRad + d)),
                              0,
                              360,
                              fill=ImageColor.getrgb(self.SECOND_COLOR))



            firstFrame = np.asarray(firstImSeg)
            secondFrame = np.asarray(secondImSeg)


            fullFrame = np.concatenate((firstFrame, secondFrame), axis=1)
            if (frameNum == -1):
                im = plt.imshow(fullFrame, aspect='auto')
            else:
                im.set_data(fullFrame)


            if (frameNum == -1):
                fig.sca(ax_fig)
                plt.plot(firstRoi._results['arrivedFrac'],
                         label=" %s, %d worms" % (self._cond1, firstRoi._results['wormCount']), color=self.FIRST_COLOR)
                plt.plot(secondRoi._results['arrivedFrac'],
                         label=" %s, %d worms" % (self._cond2, secondRoi._results['wormCount']), color=self.SECOND_COLOR)
                plt.xlabel('Frames (2Hz)')
                plt.ylabel('Worms Arrived')

                fontP = FontProperties()
                fontP.set_size('x-small')
                ax_fig.grid(alpha=0.2)
                #plt.legend(prop=fontP)

                #plt.legend()
                #plt.show()
            else:
                if vertLine is not None:
                    vertLine.remove()

                vertLine = ax_fig.axvline(x=frameNum, ymin=0, ymax=1)

            print('\rProcessed Frame %d / %d' % (frameNum, frameLength), end="")

            return (ax_vid, ax_fig)


        anim = FuncAnimation(fig, updateMovie, frames=range(frameLength - 1), init_func=lambda: updateMovie(-1))
        anim.save(path.join(self._targetPath, 'exp_pair_vis.mp4'), fps=70, extra_args=['-vcodec', 'libx264'], dpi=dpi)
        #plt.show()



def main():
    import cProfile
    import sys
    sys.path.append('/home/itskov/workspace/lab/DeepSemantic/WormSegmentation/')


    def func():
        import sys
        sys.path.append('/home/itskov/workspace/lab/DeepSemantic/WormSegmentation/')

        from Behavior.General.ExpPair import ExpPair

        expPair = ExpPair('/home/itskov/Temp/behav/10-Dec-2019/TPH_1_ATR_TRAIN_30M_NO_IAA3x5.avi_13.09.23/exp.npy',
                          '/home/itskov/Temp/behav/10-Dec-2019/NAIVE.avi_13.08.48/exp.npy')

        #expPair._firstExp.trimExperiment(4500)
        #expPair._secondExp.trimExperiment(4500)


        expPair.createPairVisualization(8000, dpi=250)

        from Behavior.Visualizers.PairwiseAnalyses import PairWiseRoi
        from Behavior.Visualizers.PairwiseAnalyses import  PairWiseProjectionDensity
        from Behavior.Visualizers.PairwiseAnalyses import PairWiseSpeedDensity


        #sns.set_context("talk")

        #PairWiseProjectionDensity('IAA-', expPair._firstExp,'IAA+', expPair._secondExp)
        #PairWiseSpeedDensity('IAA-', expPair._firstExp,'IAA+', expPair._secondExp)



    func()


if __name__ == "__main__":
    main()









