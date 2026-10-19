#pragma once

#include "types.h"
#include "tc_dynamic_state.h"

typedef enum algo_type {
    META, DYNAMIC_CONSTANT_CPU, DYNAMIC_REALTIME_CPU
} algo_type;

typedef struct BenchmarkingParameters {
    algo_type algos;
    int parallel_count;
    bool output_is_stdout;
    char * output_file;
    unsigned int batch_count;
    bool insert_edges;
    UINT_t add_count;
    float add_ratio;
    bool add_is_float;
    bool withdraw_edges;
    UINT_t delete_count;
    float delete_ratio;
    bool delete_is_float;
} BenchmarkingParameters;

void initialize_TCD_benchmarking(GRAPH_eTYPE **batc, GRAPH_eTYPE **batch, 
    const GRAPH_TYPE *const originalGraph, const GRAPH_eTYPE *const originalGraph_e, 
    GRAPH_TYPE **ptr_static_state, const BenchmarkingParameters* const a_param);

void free_TCD_benchmarking(GRAPH_eTYPE * batc, GRAPH_eTYPE * batch, 
    GRAPH_TYPE ** ptr_static_state, const BenchmarkingParameters * const a_param);

void initialize_TCD_stat_records(double **batch_times, double **total_times, 
    UINT_t **batch_counts, UINT_t **edge_counts, UINT_t **edge_counts2,
    const BenchmarkingParameters* const a_param, const unsigned int len);

void free_TCD_stat_records(double ** batch_times, double ** total_times,
    UINT_t ** batch_counts, UINT_t ** edge_counts, UINT_t ** edge_counts2,
    const BenchmarkingParameters* const a_param);

void reset_benchmarking_parameters(BenchmarkingParameters* in);

void check_count_consistency(const UINT_t truth, const UINT_t testee,
    const char* const filename, const UINT_t line, const char* const algo_name);

void print_TCD_results(FILE* outfile, char* infilename, const GRAPH_TYPE* const graph,
    double * total_times, double * batch_times, UINT_t * edge_counts, 
    UINT_t * edge_counts2, const char (* const name)[NAME_LEN], 
    UINT_t * batch_counts, const unsigned int loop_count,
    const BenchmarkingParameters* const a_param, 
    const unsigned int offset, const UINT_t len);

int print_trial_progress();

int print_reals_progress();