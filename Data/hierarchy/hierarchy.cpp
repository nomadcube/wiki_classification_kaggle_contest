/*
 * hierarchy.cpp
 *
 *  Created on: 2015年12月31日
 *      Author: wumengling
 */




#include "hierarchy.h"


void HierarchyTable::read_data(std::string f_path)
{
	std::ifstream f_stream(f_path);
	std::string line;
	while(getline(f_stream, line))
	{
		LineObject line_object;
		std::istringstream pid_and_id (line);
		pid_and_id >> line_object.parent_id;
		pid_and_id >> line_object.id;
		this->dat[line_object.id] = line_object;
	}
}

void HierarchyTable::update(int max_upward_num)
{
	for(auto element: this->dat)
	{
		int id = element.first;
		LineObject line_object = element.second;
		int original_p_id = line_object.parent_id;
		int upward_num = 0;
		while(((this->dat).count(original_p_id) > 0) && (upward_num < max_upward_num))
		{
			LineObject new_p_line_object;
			new_p_line_object.id = (this->dat)[original_p_id].id;
			new_p_line_object.parent_id = (this->dat)[original_p_id].parent_id;
			if(new_p_line_object.parent_id != id)
				(this->dat)[id] = new_p_line_object;
			original_p_id = new_p_line_object.parent_id;
			upward_num ++;
		}
	}
}

void HierarchyTable::watch()
{
	for(auto element: this->dat)
	{
		std::cout << element.first << ": " << element.second.parent_id << std::endl;
	}
}
