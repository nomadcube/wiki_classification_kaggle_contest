%module sparse_vector
%{
#include "sparse_vector.h"
%}

%include "std_map.i"
// Instantiate templates used by example
namespace std {
    %template(sparse_vector_dat_t) map<int, float>;
}

// Include the header file with above prototypes
%include "sparse_vector.h"
