# coding=utf-8
from pipeline import PipeLine
from models.mnb import LaplaceSmoothedMNB
from models.lr import LR
import cProfile, pstats, StringIO
import sys
from time import time

if __name__ == '__main__':
    debug = sys.argv[1] if len(sys.argv) > 1 else 'debug'

    local = {'train_file': '/Users/wumengling/PycharmProjects/kaggle/input_data/sub_train.csv',
             'submission_infile': '/Users/wumengling/PycharmProjects/kaggle/input_data/sub_test.csv',
             'submission_out_file': '/Users/wumengling/PycharmProjects/kaggle/output_data/submission.csv',
             'submission_save_dir': '/Users/wumengling/PycharmProjects/kaggle/input_data',
             'model_save_dir': '/Users/wumengling/PycharmProjects/kaggle/output_data',
             'max_num_label': 400000,
             'chuck_num_label': 400,
             'tf_idf_threshold': [99.2],
             'num_predict': [1]
             }

    server = {'train_file': '/home/wml/wiki_classification_kaggle_contest/input_data/train.csv',
              'submission_infile': '/home/wml/wiki_classification_kaggle_contest/input_data/sub_test.csv',
              'submission_out_file': '/home/wml/wiki_classification_kaggle_contest/output_data/submission.csv',
              'submission_save_dir': '/home/wml/wiki_classification_kaggle_contest/input_data/',
              'model_save_dir': '/model',
              'max_num_label': 2000,
              'chuck_num_label': 1000,
              'tf_idf_threshold': [99.9],
              'num_predict': [1]
              }

    # pr = cProfile.Profile()
    # pr.enable()
    t = time()

    config = local if debug == 'debug' else server
    pipeline = PipeLine(LaplaceSmoothedMNB,
                        config['tf_idf_threshold'],
                        config['num_predict'],
                        config['model_save_dir'],
                        config['submission_save_dir'],
                        config['max_num_label'])
    pipeline.model_selection(config['train_file'], config['chuck_num_label'], config['submission_infile'])
    print repr(pipeline)

    print time() - t
    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'cumtime'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()
