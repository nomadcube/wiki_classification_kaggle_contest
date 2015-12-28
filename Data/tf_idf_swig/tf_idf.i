%module tf_idf
%{
#include "tf_idf.h"
%}

%include "std_vector.i"
%include "std_map.i"
%include "std_string.i"
%newobject log_inverse_doc_frequency;
%newobject tf_idf;
// Instantiate templates used by example
namespace std {
    %template(term_val_t) map<int, float>;
    %template(doc_term_val_t) map<int, map<int,float> >;
}

// Include the header file with above prototypes
%include "tf_idf.h"
