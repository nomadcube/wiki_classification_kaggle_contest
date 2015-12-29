/*
 * sparse_vector.cpp
 *
 *  Created on: 2015年12月28日
 *      Author: wumengling
 */


#include "sparse_vector.h"

SparseVector SparseVector::dot_multiplication(float scalar)
{
	sparse_vector_dat_t res_dat;
	SparseVector res(res_dat);
	for(auto component: this->dat)
	{
		int key = component.first;
		float value = component.second;
		res.dat[key] = value * scalar;
	}
	return res;
}

SparseVector SparseVector::dot_multiplication(SparseVector another_sparse_vector)
{
	sparse_vector_dat_t res_dat;
	SparseVector res(res_dat);
	for(auto component: this->dat)
	{
		int key = component.first;
		float value = component.second;
		res.dat[key] = value * another_sparse_vector.dat[key];
	}
	return res;
}

float SparseVector::inner_product(SparseVector another_sparse_vector)
{
	float res;
	for(auto component: this->dat)
	{
		int key = component.first;
		float value = component.second;
		res += value * another_sparse_vector.dat[key];
	}
	return res;
}

SparseVector SparseVector::operator +(SparseVector rhs_sparse_vector)
{
	sparse_vector_dat_t res_dat;
	SparseVector res(res_dat);
	for(auto component: this->dat)
	{
		int key = component.first;
		float value = component.second;
		if(rhs_sparse_vector.dat.count(key) > 0)
			res.dat[key] = value + rhs_sparse_vector.dat[key];
		else
			res.dat[key] = value;
	}
	return res;
}
