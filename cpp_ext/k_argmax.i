%module k_argmax
%{
#include "k_argmax.h"
%}

%include "std_vector.i"
%include "std_pair.i"

// Instantiate templates used by example
namespace std {
   %template(element_t) vector<float>;
   %template(result_t) vector<int>;
   %template(row_t) vector<pair<int, float> >;
   %template(all_row_t) vector<vector<pair<int, float> > >;
}

// Include the header file with above prototypes
%include "k_argmax.h"
