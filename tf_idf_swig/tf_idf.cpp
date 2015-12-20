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

term_val_t log_inverse_doc_frequency(doc_term_val_t& x)
{
    term_val_t res;
    for(auto doc_term_val_pair: x)
    {
        for(auto term_val_pair: doc_term_val_pair.second)
        {
            int term = term_val_pair.first;
            if(res.count(term) <= 0)
                res[term] = 0.0;
            else
                res[term] += 1;
        }
    }
    int total_doc_count = x.size();
    for(auto term_val_pair: res)
    {
        res[term_val_pair.first] = std::log(total_doc_count / term_val_pair.second);
    }
    return res;
}

doc_term_val_t tf_idf(doc_term_val_t& x)
{
    doc_term_val_t res;
    doc_term_val_t tf = term_frequency(x);
    term_val_t idf = log_inverse_doc_frequency(x);
    for(auto doc_term_val_pair: tf)
    {
        int instance_index = doc_term_val_pair.first;
        for(auto term_val_pair: doc_term_val_pair.second)
        {
            int term = term_val_pair.first;
            float val = term_val_pair.second;
            res[instance_index][term] = x[instance_index][term] * idf[term];
        }
    }
    return res;
}
