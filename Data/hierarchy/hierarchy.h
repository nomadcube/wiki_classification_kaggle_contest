/*
 * hierarchy.h
 *
 *  Created on: 2015年12月31日
 *      Author: wumengling
 */

#ifndef HIERARCHY_H_
#define HIERARCHY_H_

#include <vector>
#include <map>
#include <utility>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

struct LineObject
{
	int id;
	int parent_id;
};

typedef std::map<int, LineObject> hierarchy_data_t;

class HierarchyTable
{
public:
	hierarchy_data_t dat;
	void read_data(std::string);
	void update(int);
	void watch();
};


#endif /* HIERARCHY_H_ */
