from submission import submission
from pipeline import PipeLine


def main():
    in_file = '/Users/wumengling/PycharmProjects/kaggle/input_data/small_origin_train_subset.csv'
    out_file = '/Users/wumengling/PycharmProjects/kaggle/output_data/submission.csv'
    pl = PipeLine([97, 95], [1.0], [1, 2])
    pl.run(in_file)
    submission(in_file, out_file, pl)


if __name__ == '__main__':
    main()
