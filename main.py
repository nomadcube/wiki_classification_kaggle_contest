# coding=utf-8
from pipeline import PipeLine
from models.mnb import LaplaceSmoothedMNB
import cProfile, pstats, StringIO
import sys
from time import time

if __name__ == '__main__':
    train_file = sys.argv[1] if len(
        sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/sub_train.csv'
    exam_file = sys.argv[2] if len(
        sys.argv) > 2 else '/Users/wumengling/PycharmProjects/kaggle/input_data/test_subset.csv'
    exam_out_file = sys.argv[3] if len(
        sys.argv) > 3 else '/Users/wumengling/PycharmProjects/kaggle/output_data/submission.csv'
    model_file = sys.argv[4] if len(sys.argv) > 4 else '/Users/wumengling/PycharmProjects/kaggle/output_data'
    test_data_save_dir = sys.argv[5] if len(sys.argv) > 5 else '/Users/wumengling/PycharmProjects/kaggle/input_data'
    chuck_size = int(sys.argv[6]) if len(sys.argv) > 6 else 400
    tf_idf_thresholds = [int(t) for t in sys.argv[7].split(',')] if len(sys.argv) > 7 else [97]

    pr = cProfile.Profile()
    pr.enable()
    t = time()

    pipeline = PipeLine(LaplaceSmoothedMNB, tf_idf_thresholds, [5], model_file, test_data_save_dir)
    pipeline.model_selection(train_file, chuck_size)
    print repr(pipeline)
    # pipeline.submission(exam_file, exam_out_file, transformed_x_exited=True)

    print time() - t
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumtime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
