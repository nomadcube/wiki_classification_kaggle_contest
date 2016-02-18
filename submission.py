# coding=utf-8
from read import Sample


def submission(test_file_path, output_file_path, pipeline):
    exam_smp = Sample()
    exam_smp.read(test_file_path)

    transformed_x = pipeline.x_converter.convert(exam_smp.x)
    predicted_y = pipeline.best_model.predict(transformed_x)
    origin_predicted_y = pipeline.y_converter.withdraw_convert(predicted_y)

    with open(output_file_path, 'w') as out:
        for i, each_predicted_y in enumerate(origin_predicted_y):
            out.write(repr(i) + ',' + ','.join([str(i) for i in each_predicted_y]) + '\n')
        out.flush()


if __name__ == '__main__':
    from pipeline import PipeLine

    pl = PipeLine([97, 95, 93], [1.0, 0.0], [1, 2, 3])
    in_file = '/Users/wumengling/PycharmProjects/kaggle/input_data/small_origin_train_subset.csv'
    pl.run(in_file)
    out_file = '/Users/wumengling/PycharmProjects/kaggle/output_data/submission.csv'
    submission(in_file, out_file, pl)
