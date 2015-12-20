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
