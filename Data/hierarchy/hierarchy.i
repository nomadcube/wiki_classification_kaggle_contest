%module hierarchy

%{
#include "hierarchy.h"
%}

%include "std_vector.i"
%include "std_map.i"
%include "std_string.i"

namespace std {
    %template(hierarchy_data_t) map<int, LineObject>;
}

%include "hierarchy.h"
