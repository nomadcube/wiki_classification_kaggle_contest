#include "tf_idf.h"


float val_sum(term_val_t& term_val)
{
    float res = 0.0;
    for(auto term_val_pair: term_val)
    {
        res += term_val_pair.second;
    }
    return res;
}

doc_term_val_t term_frequency(doc_term_val_t& x)
{
    doc_term_val_t updated_x;
    for(auto doc_term_val_pair: x)
    {
        int instance_index = doc_term_val_pair.first;
        float sum_term_val = val_sum(x[instance_index]);
        for(auto term_val_pair: x[instance_index])
        {
            int term = term_val_pair.first;
            updated_x[instance_index][term] = term_val_pair.second / sum_term_val;
        }
    }
    return updated_x;
}
