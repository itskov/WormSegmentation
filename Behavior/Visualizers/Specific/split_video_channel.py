import cv2
from skvideo.io import FFmpegWriter

mj2_cap = cv2.VideoCapture('/mnt/storageNASRe/tph1/12.03.20/01/12-Mar-2020-13.28.49-MIC2-TPH_1_ATR_ONLINE[NO_IAA]_1S.avi.mj2')
compressed_vid_handle = FFmpegWriter('/home/itskov/Temp/temp.mp4', outputdict={'-crf': '25'})

print(mj2_cap.set(cv2.CAP_PROP_POS_FRAMES, 300))

for i in range(600):
    _, f = mj2_cap.read()
    print(f.shape)
    compressed_vid_handle.writeFrame(f[:, :, 0])
    print(i)


mj2_cap.release()
compressed_vid_handle.close()