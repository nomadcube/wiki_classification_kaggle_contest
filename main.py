# coding=utf-8
from submission import submission
from pipeline import PipeLine
from models.mnb import LaplaceSmoothedMNB
import cProfile, pstats, StringIO
import sys
from time import time

if __name__ == '__main__':
    train_file = sys.argv[1] if len(
        sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/origin_train_subset.csv'
    exam_file = sys.argv[2] if len(
        sys.argv) > 2 else '/Users/wumengling/PycharmProjects/kaggle/input_data/test_subset.csv'
    exam_out_file = sys.argv[3] if len(
        sys.argv) > 3 else '/Users/wumengling/PycharmProjects/kaggle/output_data/submission.csv'
    pipeline = PipeLine(LaplaceSmoothedMNB, [99], [5])

    # pr = cProfile.Profile()
    # pr.enable()
    t = time()

    pipeline.run(train_file, 200)  # 分块大小和预测目标个数的乘积必须小于总分类数
    print repr(pipeline)
    # submission(exam_file, exam_out_file, pipeline)

    print time() - t
    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'tottime'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()
