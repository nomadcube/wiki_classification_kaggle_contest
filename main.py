from submission import submission
from pipeline import PipeLine
from models.mnb import LaplaceSmoothedMNB
import cProfile, pstats, StringIO
import sys

if __name__ == '__main__':
    in_file = sys.argv[1] if len(
        sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train_subset.csv'
    # out_file = '/Users/wumengling/PycharmProjects/kaggle/output_data/submission.csv'
    pipeline = PipeLine(LaplaceSmoothedMNB, [95], [2])

    pr = cProfile.Profile()
    pr.enable()

    pipeline.run(in_file)
    # submission(in_file, out_file, pipeline)

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
