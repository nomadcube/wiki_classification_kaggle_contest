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

term_val_t log_inverse_doc_frequency(doc_term_val_t& x)
{
    term_val_t res;
    for(auto doc_term_val_pair: x)
    {
        for(auto term_val_pair: doc_term_val_pair.second)
        {
            int term = term_val_pair.first;
                res[term] += 1;
        }
    }
    int total_doc_count = x.size();
    for(auto res_pair: res)
    {
        res[res_pair.first] = std::log(total_doc_count / res_pair.second);
    }
    return res;
}

doc_term_val_t tf_idf(doc_term_val_t& x, float threshold)
{
    doc_term_val_t res;
    term_val_t idf = log_inverse_doc_frequency(x);
    for(auto doc_term_val_pair: x)
    {
        int instance_index = doc_term_val_pair.first;
        float sum_term_val = val_sum(x[instance_index]);
        for(auto term_val_pair: doc_term_val_pair.second)
        {
            int term = term_val_pair.first;
            float tf_idf_val = (x[instance_index][term] / sum_term_val)  * idf[term];
            if(tf_idf_val > threshold)
                res[instance_index][term] = tf_idf_val;
        }
    }
    return res;
}
