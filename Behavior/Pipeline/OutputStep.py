from Behavior.Pipeline.AnalysisStep import AnalysisStep

from skvideo.io import FFmpegWriter

class OutputStep(AnalysisStep):

    # Return None if failed.
    def process(self, artifacts):
        if 'full_vid_handle' not in artifacts:
            full_vid_handle = FFmpegWriter(artifacts['full_vid_filename'], outputdict={'-crf': '0' })
            artifacts['full_vid_handle'] = full_vid_handle
        if 'compressed_vid_handle' not in artifacts:
            compressed_vid_handle = FFmpegWriter(artifacts['compressed_vid_filename'], outputdict={'-crf': '25'})
            artifacts['compressed_vid_handle'] = compressed_vid_handle
        if 'seg_vid_handle' not in artifacts:
            seg_vid_handle = FFmpegWriter(artifacts['seg_vid_filename'], outputdict={'-crf': '0'})
            artifacts['seg_vid_handle'] = seg_vid_handle

        artifacts['full_vid_handle'].writeFrame(artifacts['current_frame'])
        artifacts['compressed_vid_handle'].writeFrame(artifacts['current_frame'])
        artifacts['seg_vid_handle'].writeFrame(artifacts['segmented_frame'])


    def close(self, artifacts):
        artifacts['full_vid_handle'].close()
        artifacts['compressed_vid_handle'].close()
        artifacts['seg_vid_handle'].close()


    def stepName(self, artifacts):
        return 'Output'

    def checkDependancies(self, artifacts):
        return

