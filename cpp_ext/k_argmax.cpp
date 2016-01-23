/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "k_argmax.h"


all_row_t construct_all_row(element_t element, std::vector<int> column_indices, std::vector<int> row_indptr)
{
    all_row_t all_row;
    for(int row_no = 0; row_no < row_indptr.size() - 1; row_no ++)
    {
        row_t row;
        int begin_i = row_indptr[row_no];
        int end_i = row_indptr[row_no + 1] - 1;
        for(int i = begin_i; i <= end_i; ++i)
        {
            row_element_t e;
            e.first = element[i];
            e.second = column_indices[i];
            row.push_back(e);
        }
        all_row.push_back(row);
    }
    return all_row;
}

int k_argmax_per_row(row_t one_row)
{
    if(one_row.size() == 0)
        return -1;
    else{
        auto largest = one_row.begin();
        for(auto b = one_row.begin(); b != one_row.end(); ++b)
        {
            if((*b).first > (*largest).first)
                largest = b;
        }
        return (*largest).second;
    }
}

result_t k_argmax(element_t element, std::vector<int> column_indices, std::vector<int> row_indptr)
{
    all_row_t all_row = construct_all_row(element, column_indices, row_indptr);
    result_t res;
    for(int i = 0; i < all_row.size(); i++)
        res.push_back(k_argmax_per_row(all_row[i]));
    return res;
}
