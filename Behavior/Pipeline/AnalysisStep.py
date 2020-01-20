from time import time

class AnalysisStep:
    # Each child should implement these

    # Return None if failed.
    def process(self, artifacts):
        raise Exception("NotImplementedException")

    def close(self, artifacts):
        raise Exception("NotImplementedException")

    def stepName(self, artifacts):
        raise Exception("NotImplementedException")

    def checkDependancies(self, artifacts):
        raise Exception("NotImplementedException")



    def run(self, artifacts):
        return self.process(artifacts)

    def log(self, msg):
        print(msg)


def process(pipline, artifacts):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(artifacts['mj2_path'])
    total_frames = np.min((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 6500))
    cap.release()

    start_time = time()
    # Change to 8000
    while artifacts['frame_num'] < total_frames:
        so_far_time = time() - start_time
        print('Frame: %d Time: %f m' % (artifacts['frame_num'], so_far_time / 60))
        all_time_before = time()
        for i, proc in enumerate(pipline):

            before_time = time()
            artifacts = proc.process(artifacts)
            duration_time = time() - before_time
            print('\t%d. %s time: %f s' % (i, proc.stepName(artifacts), duration_time))

        all_time_duration = time() - all_time_before
        print('\tOverall: %f s' %(all_time_duration,))

    print('Closing..')
    [p.close(artifacts) for p in pipline]

if __name__ == '__main__':
    import sys

    sys.path.append('/cs/phd/itskov/WormSegmentation')

    from Behavior.Pipeline.SplitChannelsStep import SplitChannelsStep
    from Behavior.Pipeline.SegmentStep import SegmentStep
    from Behavior.Pipeline.TrackStep import TrackStep
    from Behavior.Pipeline.OutputStep import OutputStep

    artifacts = {}
    artifacts['mj2_path'] = sys.argv[1]
    artifacts['restore_points'] = '/cs/phd/itskov/WormSegmentation/WormSegmentatioNetworks/WormSegmentation'
    artifacts['frame_num'] = 0

    pipline = [SplitChannelsStep(), SegmentStep(), OutputStep(), TrackStep()]
    process(pipline, artifacts)

    '''from Behavior.Pipeline.SplitChannelsStep import SplitChannelsStep
    from Behavior.Pipeline.SegmentStep import SegmentStep
    from Behavior.Pipeline.OutputStep import OutputStep
    from Behavior.Pipeline.TrackStep import TrackStep

    artifacts = {'mj2_path' : '/mnt/storageNASRe/tph1/31.12.19/31-Dec-2019-13.42.39-Mic1-TPH_1_NO_ATR_TRAIN_35M_NO_IAA3x5.avi.mj2',
                 'restore_points' : '/home/itskov/workspace/lab/DeepSemantic/WormSegmentation/WormSegmentatioNetworks/WormSegmentation'}

    sc = SplitChannelsStep()
    seg = SegmentStep()
    os = OutputStep()
    ts = TrackStep()

    artifacts = sc.process(artifacts)
    artifacts = seg.process(artifacts)
    artifacts = ts.process(artifacts)
    artifacts = sc.process(artifacts)
    artifacts = seg.process(artifacts)
    artifacts = ts.process(artifacts)
    artifacts = sc.process(artifacts)
    artifacts = seg.process(artifacts)
    artifacts = ts.process(artifacts)

    os.process(artifacts)
    os.process(artifacts)
    os.process(artifacts)
    os.close(artifacts)'''


