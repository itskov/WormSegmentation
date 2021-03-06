from Behavior.Pipeline.AnalysisStep import AnalysisStep
from os import path

import cv2


class SplitChannelsStep(AnalysisStep):
    # Each child should implement these
    def process(self, artifacts):
        if 'mj2_cap' not in artifacts:
            mj2_path = artifacts['mj2_path']
            mj2_cap = cv2.VideoCapture(mj2_path)

            if not mj2_cap.isOpened:
                self.log("Couldn't open the mj2 file.")
                return None

            mj2_cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            artifacts['mj2_cap'] = mj2_cap
            artifacts['frame_num'] = 0

        if 'full_vid_filename' not in artifacts:
            inputPath = path.dirname(mj2_path)
            baseName = ".".join(path.basename(mj2_path).split(".")[0:-1])
            extension = 'mp4'

            channelFileUncompressed = path.join(inputPath, baseName + "_Full." + extension)

            artifacts['full_vid_filename'] = channelFileUncompressed

        if 'compressed_vid_filename' not in artifacts:
            inputPath = path.dirname(mj2_path)
            baseName = ".".join(path.basename(mj2_path).split(".")[0:-1])
            extension = 'mp4'

            channelFileCompressed = path.join(inputPath, baseName + "_Compressed." + extension)

            artifacts['compressed_vid_filename'] = channelFileCompressed

        mj2_cap = artifacts['mj2_cap']
        artifacts['frame_num'] += 1

        success, read_frame = mj2_cap.read()

        if not success:
            self.log('Error reading from mj2 file.')
            return None

        if not artifacts['take_first_channel']:
            current_frame = cv2.cvtColor(read_frame, cv2.COLOR_BGR2GRAY)
        else:
            current_frame = read_frame[:, :, 0]


        # Saving the current frame for future processing
        artifacts['current_frame'] = current_frame

        return artifacts


    def close(self, artifacts):
        mj2_cap = artifacts['mj2_cap']

        #vid_compressed_ffmpeg.close()
        #vid_full_ffmpeg.close()
        mj2_cap.release()


    def checkDependancies(self, artifacts):
        if 'mj2_path' not in artifacts:
            raise Exception('Cant find mj2_path in artifacts.')



    def stepName(self, artifacts):
        return 'Split Channel'
