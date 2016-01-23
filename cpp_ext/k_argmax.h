/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   k_argmax.h
 * Author: wumengling
 *
 * Created on 2016年1月22日, 下午4:39
 */

#ifndef K_ARGMAX_H
#define K_ARGMAX_H

#include <vector>
#include <utility>


typedef std::vector<float> element_t;
typedef std::vector<int> result_t;

typedef std::pair<int, float> row_element_t;
typedef std::vector<row_element_t> row_t;
typedef std::vector<row_t> all_row_t;

all_row_t construct_all_row(element_t, std::vector<int>, std::vector<int>);
int k_argmax_per_row(row_t);
result_t k_argmax(element_t, std::vector<int>, std::vector<int>);

#endif /* K_ARGMAX_H */

