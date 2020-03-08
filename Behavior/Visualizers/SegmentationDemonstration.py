import numpy as np

from os import path
import cv2

from skvideo.io import FFmpegWriter


def alignImage(img):
    # First we roll the image to center the plate.
    rightBorder = np.min(np.where(img > 0)[1])
    leftBorder = img.shape[1] - np.max(np.where(img > 0)[1])
    allBorders = rightBorder + leftBorder
    correctBorder = np.floor(allBorders / 2)
    return int(correctBorder - rightBorder)


def create_segmentation_demonstration_movie(first_movie_path, second_movie_path):
    # Some constants
    bar_width_frac = 1 / 80
    frames_per_screen = 25

    output_filename = path.join(path.dirname(first_movie_path), "seg_demo.mp4")
    output_file = FFmpegWriter(output_filename, outputdict={'-crf': '20'})

    first_vid = cv2.VideoCapture(first_movie_path)
    second_vid = cv2.VideoCapture(second_movie_path)

    frame_num_1 = int(first_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num_2 = int(second_vid.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = np.min((frame_num_1, frame_num_2))

    # Reading first frame for shape.
    _, first_frame = first_vid.read()
    _, second_frame = second_vid.read()

    # Alignining only the segmented version.
    second_frame_align = alignImage(second_frame)

    width = first_frame.shape[1]
    bar_width = bar_width_frac * width

    splits_num = int(1 / bar_width_frac)
    segments = [tuple(s) for s in np.split(np.array(range(width)), splits_num)]


    for i in range(frame_num - 2):
    #for i in range(750):
        print('Working on frame %d' % (i,))

        _, first_frame = first_vid.read()
        _, second_frame = second_vid.read()

        first_frame = np.roll(first_frame, second_frame_align, axis=1)
        second_frame = np.roll(second_frame, second_frame_align, axis=1)

        bar_step = i % (splits_num * 2)
        if bar_step > splits_num:
            bar_step = splits_num - (bar_step - splits_num)

        #print(bar_step)
        if bar_step > 0 and bar_step < splits_num:
            first_part = first_frame[:, np.ravel(segments[bar_step:]), :]
            second_part = second_frame[:, np.ravel(segments[0:bar_step]), :]

            new_frame = np.concatenate((second_part, first_part), axis=1)


            output_file.writeFrame(new_frame)


    output_file.close()
    pass



    first_vid.release()
    second_vid.release()



def main():
    #first_movie_path = sys.argv[1]
    #second_movie_path = sys.argv[1]

    create_segmentation_demonstration_movie('/home/itskov/Temp/behav/27-Feb-2020/TPH_1_ATR_TRAIN_75M_0D.avi_14.20.27/27-Feb-2020-14.20.27-MIC2-TPH_1_ATR_TRAIN_75M_0D.avi_seg_raw_tracked.mp4',
                                            '/home/itskov/Temp/behav/27-Feb-2020/TPH_1_ATR_TRAIN_75M_0D.avi_14.20.27/27-Feb-2020-14.20.27-MIC2-TPH_1_ATR_TRAIN_75M_0D.avi_seg.mp4')



if __name__ == "__main__":
    main()

