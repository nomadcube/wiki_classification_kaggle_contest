/*
 * sparse_vector.h
 *
 *  Created on: 2015年12月28日
 *      Author: wumengling
 */

#ifndef SPARSE_VECTOR_H_
#define SPARSE_VECTOR_H_


#include <map>

typedef std::map<int, float> sparse_vector_dat_t;

class SparseVector
{
public:
	sparse_vector_dat_t dat;
	SparseVector(sparse_vector_dat_t init_dat):dat(init_dat){};
	~SparseVector(){};
	SparseVector dot_multiplication(float);
	SparseVector dot_multiplication(SparseVector);
};


#endif /* SPARSE_VECTOR_H_ */
