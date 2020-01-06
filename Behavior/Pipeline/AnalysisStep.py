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
    while artifacts['frame_num'] < 8000:
        for i, proc in enumerate(pipline):
            before_time = time()
            artifacts = proc.process(artifacts)
            duration_time = time() - before_time
            print('$d. %s time: %d', (i, proc.stepName(artifacts), duration_time))

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

    pipline = [SplitChannelsStep(), SegmentStep(), TrackStep(), OutputStep()]
    process(pipline, artifacts)

    '''from Behavior.Pipeline.SplitChannelsStep import SplitChannelsStep
    from Behavior.Pipeline.SegmentStep import SegmentStep
    from Behavior.Pipeline.OutputStep import OutputStep

    artifacts = {'mj2_path' : '/mnt/storageNASRe/tph1/31.12.19/31-Dec-2019-13.42.39-Mic1-TPH_1_NO_ATR_TRAIN_35M_NO_IAA3x5.avi.mj2',
                 'restore_points' : '/home/itskov/workspace/lab/DeepSemantic/WormSegmentation/WormSegmentatioNetworks/WormSegmentation'}

    sc = SplitChannelsStep()
    seg = SegmentStep()
    os = OutputStep()

    artifacts = sc.process(artifacts)
    artifacts = seg.process(artifacts)
    artifacts = sc.process(artifacts)
    artifacts = seg.process(artifacts)
    os.process(artifacts)
    os.process(artifacts)
    os.process(artifacts)
    os.close()'''


