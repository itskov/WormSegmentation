from Behavior.Tools.Artifacts import Artifacts
from Behavior.General.ExpDir import ExpDir
from skvideo.io import FFmpegWriter
from PIL import Image, ImageDraw, ImageFont, ImageColor

import numpy as np

import matplotlib.pyplot as plt


import cv2


def alignImage(img):
    # First we roll the image to center the plate.
    rightBorder = np.min(np.where(img > 0)[1])
    leftBorder = img.shape[1] - np.max(np.where(img > 0)[1])
    allBorders = rightBorder + leftBorder
    correctBorder = np.floor(allBorders / 2)
    return int(correctBorder - rightBorder)


def demonstrateLightingSegmentation(exp_dir):
    expDir = ExpDir(exp_dir)
    artifacts = Artifacts(expLocation=exp_dir)

    # Getting the light intensity file
    light_intensities = artifacts.getArtifact('frame_intensities')

    base_line = np.mean(light_intensities[1:10])

    #vid_cap = cv2.VideoCapture(expDir.getVidFile())
    vid_cap = cv2.VideoCapture('/mnt/storageNASRe/tph1/12.03.20/12-Mar-2020-21.16.44-MIC2-TPH_1_ATR_ONLINE[NO_IAA]_5S.avi.mj2')
    seg_cap = cv2.VideoCapture(expDir.getExpSegVid())

    spike_count = 0
    in_spike = False
    in_spike_count = 0

    full_vid_handle = FFmpegWriter('/home/itskov/Dropbox/dem.mp4', outputdict={'-crf': '0'})

    # Reading first frame for shape.
    _, first_frame = vid_cap.read()
    _, second_frame = seg_cap.read()

    # Alignining only the segmented version.
    seg_frame_align = alignImage(second_frame)

    start_frame = 700
    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    seg_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)


    for i in range(start_frame, 2000):
        _, vid_frame = vid_cap.read()
        _, seg_frame = seg_cap.read()

        vid_frame = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
        vid_frame = cv2.cvtColor(vid_frame, cv2.COLOR_GRAY2BGR)

        #vid_frame = np.reshape(vid_frame, (vid_frame.shape))

        vid_frame = np.roll(vid_frame, seg_frame_align, axis=1)
        seg_frame = np.roll(seg_frame, seg_frame_align, axis=1)


        if light_intensities[i] > (base_line + 7):
            in_spike = True
            in_spike_count += 1

            if in_spike_count == 6 and spike_count % 2 == 0 and spike_count > 0:
                splits_num = 80
                width = vid_frame.shape[1]
                segments = [tuple(s) for s in np.split(np.array(range(width)), splits_num)]

                for i in range(2 * splits_num - 1):
                    bar_step = i % (splits_num * 2)
                    if bar_step > splits_num:
                        bar_step = splits_num - (bar_step - splits_num)

                    if bar_step > 0 and bar_step < splits_num:
                        print(bar_step)
                        #print(segments)
                        #print(segments[bar_step:])


                        first_part = vid_frame[:, np.ravel(segments[bar_step:]), :]
                        second_part = seg_frame[:, np.ravel(segments[0:bar_step]), :]

                        new_frame = np.concatenate((second_part, first_part), axis=1)
                        cur_img = Image.fromarray(new_frame)
                        cur_imgdraw = ImageDraw.Draw(cur_img)

                        cur_imgdraw.ellipse((40, 40, 180, 180), fill='red', outline='red')

                        full_vid_handle.writeFrame(np.array(cur_img))

            else:
                # Do Spike stuff here.
                cur_img = Image.fromarray(vid_frame)
                cur_imgdraw = ImageDraw.Draw(cur_img)

                cur_imgdraw.ellipse((40, 40, 180, 180), fill='red', outline='red')
                full_vid_handle.writeFrame(np.array(cur_img))
        else:
            if in_spike == True:
                in_spike = False
                spike_count += 1

            in_spike_count = 0
            full_vid_handle.writeFrame(vid_frame)



        print('Frame %d. Spike Count; %d' % (i, spike_count))

    vid_cap.release()
    seg_cap.release()
    full_vid_handle.close()


if __name__ == "__main__":
    demonstrateLightingSegmentation('/home/itskov/Temp/behav/12-Mar-2020/TPH_1_ATR_ONLINE[NO_IAA]_5S.avi_21.16.44/')
