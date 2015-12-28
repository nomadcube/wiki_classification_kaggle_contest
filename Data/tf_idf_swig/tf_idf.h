/*
 * File:   tf_idf.h
 * Author: wumengling
 *
 * Created on 2015年12月20日, 上午11:29
 */

#ifndef TF_IDF_H
#define TF_IDF_H

#include <map>
#include <vector>
#include <utility>
#include <cmath>

typedef std::map<int, float> term_val_t;
typedef std::map<int, term_val_t> doc_term_val_t;

float val_sum(term_val_t&);
term_val_t log_inverse_doc_frequency(doc_term_val_t&);
doc_term_val_t tf_idf(doc_term_val_t&, float threshold);

#endif /* TF_IDF_H */
