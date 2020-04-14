from Behavior.General.ExpDir import ExpDir
from Behavior.Tools.Artifacts import Artifacts
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import cv2


def alignImage(img):
    # First we roll the image to center the plate.
    rightBorder = np.min(np.where(img > 50)[1])
    leftBorder = img.shape[1] - np.max(np.where(img > 0)[1])
    allBorders = rightBorder + leftBorder
    correctBorder = np.floor(allBorders / 2)
    return int(correctBorder - rightBorder)


def createDemMovie(exp_dir):
    expDir = ExpDir(exp_dir)
    artifacts = Artifacts(expLocation=exp_dir)

    # Getting the frame intensities values
    frame_intensities = artifacts.getArtifact('frame_intensities')
    frame_intensities /= np.max(frame_intensities)

    frame_range = range(1200, 3000)

    #cap = cv2.VideoCapture(expDir.getVidFile())
    cap = cv2.VideoCapture(expDir.getExpSegVid())
    cap.set(cv2.CAP_PROP_POS_FRAMES, np.min(frame_range))
    frame_intensities = frame_intensities[frame_range]

    plt.style.use('dark_background')
    plt.tight_layout()
    sns.set_context('paper')

    # Setting a 3 to 1 ratio between video and figure.
    fig, axs = \
        plt.subplots(2, 3, gridspec_kw={'height_ratios':[4, 1], 'width_ratios':[1 ,3, 1]})

    ax_vid = axs[0][1]
    ax_fig = axs[1][1]

    ax_vid.axis('off')
    axs[0][0].axis('off')
    axs[0][2].axis('off')
    axs[1][0].axis('off')
    axs[1][2].axis('off')

    # Global variables
    im = None
    vidAlign = 0
    vertLine = None

    def updateMovie(frameNumber):
        nonlocal ax_vid
        nonlocal ax_fig
        nonlocal cap
        nonlocal im
        nonlocal vidAlign
        nonlocal frame_range
        nonlocal vertLine

        print('Frame number: %d' % frameNumber)

        #DEBUG
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
        #DEBUG

        # Updating the video
        fig.sca(ax_vid)
        _, frame = cap.read()
        frame = frame[480:1440, 1100:2300]
        if frameNumber == -1:
            vidAlign = alignImage(frame)
            frame = np.roll(frame, vidAlign, axis=1)
            im = plt.imshow(frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, np.min(frame_range))
        else:
            frame = np.roll(frame, vidAlign, axis=1)
            im.set_data(frame)

        # Updating the figure
        fig.sca(ax_fig)
        if frameNumber == -1:
            plt.gca().grid(alpha=0.2)
            plt.plot(np.array(frame_range) * 0.5, frame_intensities, linewidth=1.5, color=sns.xkcd_rgb['windows blue'], alpha=0.75)
            plt.xlabel('Time [s]')
            h = plt.ylabel('Power [au]')
            #h.set_rotation(0)
            plt.xlim(np.min(frame_range) * 0.5, np.max(frame_range) * 0.5)
        else:
            if vertLine is not None:
                vertLine.remove()

            vertLine = ax_fig.axvline(x=frameNumber * 0.5, ymin=0, ymax=1)

        return (ax_vid, ax_fig)

    anim = FuncAnimation(fig, updateMovie, frames=[1905], init_func=lambda: updateMovie(-1))
    anim.save('/home/itskov/Dropbox/DemMovie2.mp4', fps=50, extra_args=['-vcodec', 'libx264'], dpi=250)

    cap.release()



if __name__ == "__main__":
    exp_dir = '/home/itskov/Temp/behav/temp/TPH_1_ATR_ONLINE[NO_IAA]_5S.avi_21.16.44'
    createDemMovie(exp_dir)