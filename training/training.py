import liblinearutil


def select_labels_for_prediction(sample, max_label_count):
    distinct_labels = list()
    num_included_label = 0
    for each_y in sample.y.values():
        for single_label in each_y.split(','):
            if (num_included_label < max_label_count) and (single_label not in distinct_labels):
                distinct_labels.append(single_label)
                num_included_label += 1
    return distinct_labels


def train_and_collect_predict_result(current_predict_result, whole_sample, target_label, train_proportion):
    whole_sample.convert_to_binary_class(target_label)
    tr_y, tr_x, te_y, te_x, tr_keys, te_keys = whole_sample.split_train_test(train_proportion)
    model = liblinearutil.train(tr_y, tr_x, '-s 0 -c 0.03')
    predicted_y, accuracy, decision_value = liblinearutil.predict(te_y, te_x, model)
    current_predict_result.update(target_label, te_keys, predicted_y)


def preparation_for_train(config_dat):
    all_data_for_train = namedtuple('all_data',
                                    'dat train_y train_x test_y test_x instance_for_train instance_for_test')
    hierarchy_table = hierarchy.HierarchyTable()
    hierarchy_table.read_data(config_dat['hierarchy_f_path'])
    hierarchy_table.update(config_dat['hierarchy_upward_step'])

    dat = sample_reader(config_dat['sample_f_path'], config_dat['sampling_prop'])
    print(dat.description())

    dat.dimension_reduction(config_dat['tf_idf_threshold'])
    dat.collect_all_feature_and_count()
    dat.remap_feature_index_after_tfidf()
    print(dat.description())

    if config_dat['disassemble_multi_label']:
        dat.label_string_disassemble()
        print(dat.description())

    if config_dat['label_upward']:
        dat.label_upward(hierarchy_table)
        print(dat.description())

    return all_data_for_train(dat, dat.split_train_test(config_dat['train_prop']))
