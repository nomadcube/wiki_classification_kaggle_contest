from submission import submission
from pipeline import PipeLine


def main():
    pl = PipeLine([97, 95, 93], [1.0, 0.0], [1, 2, 3])
    in_file = '/Users/wumengling/PycharmProjects/kaggle/input_data/small_origin_train_subset.csv'
    pl.run(in_file)
    out_file = '/Users/wumengling/PycharmProjects/kaggle/output_data/submission.csv'
    submission(in_file, out_file, pl)


if __name__ == '__main__':
    main()
