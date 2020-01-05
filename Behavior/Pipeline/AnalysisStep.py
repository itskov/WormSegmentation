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


if __name__ == '__main__':
    from Behavior.Pipeline.SplitChannelsStep import SplitChannelsStep
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
    os.close()


