# coding=utf-8
from read import Sample


def submission(test_file_path, output_file_path, pipeline):
    print pipeline.best_f_score

    exam_smp = Sample()
    exam_smp.read(test_file_path)

    transformed_x = pipeline.best_x_converter.convert(exam_smp.x)
    predicted_y = pipeline.best_model.partial_predict(transformed_x, pipeline.best_predicted_cnt)
    origin_predicted_y = pipeline.best_y_converter.withdraw_convert(predicted_y)

    with open(output_file_path, 'w') as out:
        out.write('Id,Predicted' + '\n')
        for i, each_predicted_y in enumerate(origin_predicted_y):
            out.write(repr(i) + ',' + ' '.join([str(i) for i in each_predicted_y]) + '\n')
        out.flush()


if __name__ == '__main__':
    from pipeline import PipeLine

    pl = PipeLine([97], [1.0], [3])
    in_file = '/Users/wumengling/PycharmProjects/kaggle/input_data/small_origin_train_subset.csv'
    pl.run(in_file)
    out_file = '/Users/wumengling/PycharmProjects/kaggle/output_data/submission.csv'
    submission(in_file, out_file, pl)
