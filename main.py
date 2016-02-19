from submission import submission
from pipeline import PipeLine


if __name__ == '__main__':
    import sys
    import cProfile, pstats, StringIO

    pr = cProfile.Profile()
    pr.enable()

    in_file = sys.argv[1] if len(
        sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/origin_train_subset.csv'
    # out_file = '/Users/wumengling/PycharmProjects/kaggle/output_data/submission.csv'
    pl = PipeLine([99.5], [1.0], [2])
    pl.run(in_file)
    # submission(in_file, out_file, pl)


    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
