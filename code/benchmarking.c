#include "benchmarking.h"
#include "graph.h"
#include "tc_dynamic_state.h"
#include "types.h"
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

void reset_benchmarking_parameters(BenchmarkingParameters* in)
{
    in->algos = META;
    in->parallel_count = 0;
    in->output_is_stdout = true;
    in->output_file = NULL;
    in->batch_count = 0;
    in->insert_edges = false;
    in->withdraw_edges = false;
    in->add_count = 0;
    in->add_ratio = 0.0;
    in->add_is_float = false;
    in->delete_count = 0;
    in->delete_ratio = 0.0;
    in->delete_is_float = false;
}

/**
* Used for both double batches and normal batches
* Ensures they both work the same way
*
* @param a_param Details of the benchmarking run
*
* @param insert_or_withdraw Prepare the initialization for insertion or withdrawing of edges
*
* @param number_edges Emits the number of edges
*
* @param float_or_count Emits whether input has been in float or integer format
*
* @param originalGraph Initial graph passed in
*
* @param BATCH_COUNT Number of batches
*/
void common_batch_initializers(const BenchmarkingParameters* const a_param,
    const bool insert_or_withdraw, UINT_t* number_edges, float* initial_ratio,
    bool* float_or_count, const GRAPH_TYPE* const originalGraph,
    const unsigned int BATCH_COUNT)
{
    bool task_selected = false;
    if(insert_or_withdraw && a_param->insert_edges) {
        *number_edges = a_param->add_count;
        *initial_ratio = a_param->add_ratio;
        *float_or_count = a_param->add_is_float;
        task_selected = true;
    }
    if(!insert_or_withdraw && a_param->withdraw_edges) {
        *number_edges = a_param->delete_count;
        *initial_ratio = a_param->delete_ratio;
        *float_or_count = a_param->delete_is_float;
        task_selected = true;
    }
    if(!task_selected) {
        fprintf(stderr, "Neither insertion nor deletion of edges has been "
        "asked for...\n");
    }
    if(*float_or_count) {
        *number_edges = (UINT_t)(*initial_ratio * originalGraph->numEdges);
    }
    else {
        *initial_ratio = ((float)a_param->add_count) / ((float)originalGraph->numEdges);            
    }
    if(!insert_or_withdraw && a_param->withdraw_edges 
        && originalGraph->numEdges < BATCH_COUNT * *number_edges) {
        fprintf(stderr, "More edges than that which exist in the graph "
            "are requested for deletion. Capping the number of edges...\n");
        *initial_ratio = ((float)originalGraph->numEdges) / ((float)BATCH_COUNT);
        *number_edges = (UINT_t)(*initial_ratio * originalGraph->numEdges);
    }
}

UINT_t common_batch_sizers(const BenchmarkingParameters* const a_param,
    const GRAPH_TYPE* const originalGraph, const UINT_t number_edges, 
    const float initial_ratio, const bool float_or_count, 
    const UINT_t total_until_now, const bool insert_or_withdraw)
{
    switch(a_param->algos)
    {
        default:
        case DYNAMIC_CONSTANT_CPU:
            return number_edges;
            break;
        case DYNAMIC_REALTIME_CPU:
            if(float_or_count) {
                UINT_t current_graph_edge_count;
                if(insert_or_withdraw) {
                    current_graph_edge_count = originalGraph->numEdges +
                        total_until_now;
                }
                else {
                    current_graph_edge_count = originalGraph->numEdges -
                        total_until_now;
                }
                return (UINT_t)(initial_ratio * 
                    ((float)current_graph_edge_count));
            }
            else {
                return number_edges;
            }
            break;
    }
}

void common_static_state_initializer(const GRAPH_TYPE* const originalGraph,
    GRAPH_TYPE** ptr_static_state)
{
    *ptr_static_state = (GRAPH_TYPE*)malloc(sizeof(GRAPH_TYPE));
    assert_malloc(*ptr_static_state);
    GRAPH_TYPE * restrict const static_state = *ptr_static_state;
    static_state->numEdges = originalGraph->numEdges;
    static_state->numVertices = originalGraph->numVertices;
    allocate_graph(static_state);
    copy_graph(originalGraph, static_state);
}

/**
* Allocating variables used in benchmarking. Only one of insertion or 
* deletion of edges is active
*
* @param batc Pointer to an array of pointers to batches of random edges
*
* @param batch Pointer to a pointer pointing at a batch of random edges
*
* @param originalGraph Pointer to GRAPH_TYPE representation of starting graph
*
* @param originalGraph_e Pointer to GRAPH_eTYPE representation of starting graph
*
* @param ptr_static_state Pointer to a pointer pointing to the static_state
*
* @param a_param Pointer to a benchmark object
*/
void initialize_TCD_benchmarking(GRAPH_eTYPE **batc, GRAPH_eTYPE **batch, 
    const GRAPH_TYPE *const originalGraph, const GRAPH_eTYPE *const originalGraph_e, 
    GRAPH_TYPE **ptr_static_state, const BenchmarkingParameters* const a_param)
{
    const unsigned int BATCH_COUNT = a_param->batch_count;
    *batc = (GRAPH_eTYPE*)malloc(BATCH_COUNT * sizeof(GRAPH_eTYPE));
    assert_malloc(*batc);
    UINT_t number_edges;
    float initial_ratio;
    bool float_or_count;
    UINT_t total = 0;
    common_batch_initializers(a_param, a_param->insert_edges, 
        &number_edges, &initial_ratio, 
        &float_or_count, originalGraph, BATCH_COUNT);
    // Assuming either insert or withdraw edges
    UINT_t max_number_edges = 0;
    for(int i = 0; i < BATCH_COUNT; ++i) {
        const UINT_t current_number_edges = common_batch_sizers(a_param, originalGraph, 
            number_edges, initial_ratio, float_or_count, 
            total, a_param->insert_edges);
        total += current_number_edges;
        max_number_edges = max(current_number_edges, max_number_edges);
        allocateBatch(*batc + i, current_number_edges,
            originalGraph->numVertices);
    }

    *batch = malloc(sizeof(GRAPH_eTYPE));
    allocateBatch(*batch, max_number_edges, 
        originalGraph->numVertices);

    common_static_state_initializer(originalGraph, ptr_static_state);
}

void check_count_consistency(const UINT_t truth, const UINT_t testee,
    const char* const filename, const UINT_t line, const char* const algo_name)
{
    if(truth != testee)
        fprintf(stderr, "\nError with batch count. Required count %zu, "
        "current count %zu\nLocation: %s, %zu\nAlgorithm: %s\n", 
        truth, testee, filename, line, algo_name);
}

/**
* Allocating memory for storing stats
*
* @param batch_times Pointer to an array of time taken to ruin each batch
*
* @param total_times Pointer to an array of time taken to run all batches for an algorithm
*
* @param batch_counts Pointer to an array of resulting count from each batch
*
* @param edge_count Pointer to an array of number of edges in each batch
*
* @param edge_counts2 Optional count of edges. Used when there are two counts
*
* @param len Number of dynamic algorithms using ALT_DYNAMIC_STATE
*/
void initialize_TCD_stat_records(double **batch_times, double **total_times, 
    UINT_t **batch_counts, UINT_t **edge_counts, UINT_t **edge_counts2,
    const BenchmarkingParameters* const a_param, const unsigned int len)
{
    const unsigned int BATCH_COUNT = a_param->batch_count;
    const unsigned int stat_len = len + 1;
    *batch_times = (double*)calloc(BATCH_COUNT * stat_len, sizeof(double));
    assert_malloc(*batch_times);
    *batch_counts = (UINT_t*)calloc(BATCH_COUNT * stat_len, sizeof(UINT_t));
    assert_malloc(*batch_counts);
    *edge_counts = (UINT_t*)calloc(BATCH_COUNT, sizeof(UINT_t));
    assert_malloc(*edge_counts);
    if(a_param->insert_edges && a_param->withdraw_edges) {
        *edge_counts2 = calloc(BATCH_COUNT, sizeof(UINT_t));
        assert_malloc(*edge_counts2);
    }
    *total_times = (double*)calloc(stat_len, sizeof(double));
    assert_malloc(*total_times);
}

/**
* Printing results of insertion TCD algorithms
*
* @param outfile Output file stream
*
* @param infilename Name of the input file
*
* @param graph Graph read from input file
*
* @param total_times Time taken over all batches and repeated runs. Does not include trial run
*
* @param batch_times Time taken by each batch
*
* @param edge_counts Number of edges in each batch fed into algorithm
*
* @param name Name of each algorithm
*
* @param batch_counts Triangles found after adding each batch
*
* @param loop_count Number of repeat runs to get an average runtime
*
* @param a_param A list of parameters supplied for this benchmarking run
*
* @param offset to dial in the start of the relevant statistics
*
* @param len Number of algorithms being benchmarked
*/
void print_TCD_results(FILE* outfile, char* infilename, const GRAPH_TYPE* const graph,
    double * total_times, double * batch_times, UINT_t * edge_counts, 
    UINT_t * edge_counts2, const char (* const name)[NAME_LEN], 
    UINT_t * batch_counts, const unsigned int loop_count,
    const BenchmarkingParameters* const a_param, 
    const unsigned int offset, const UINT_t len)
{
    const unsigned int BATCH_COUNT = a_param->batch_count;
    const bool parallel_switch = a_param->parallel_count != 0;
    printf("\n");
    char* buf = calloc(strlen(infilename) + 7, sizeof(char));
    assert_malloc(buf);
    for(unsigned int k = 0; k < len; ++k) {
        unsigned int j = k + offset;
        fprintf(outfile,"TC\t%-30s\t%12zu\t%12zu\t%-30s\t%9.6f\n",
        infilename,
        graph->numVertices, graph->numEdges,
        name[k], total_times[j] / loop_count);
        for(unsigned int i = 0; i < BATCH_COUNT; ++i) {
            sprintf(buf, "%s-%d", infilename, i);
            if(parallel_switch) {
                fprintf(outfile, "BC\t%-30s\t%12zu\t%12zu\t%-30s\t%9.6f\t%12zu\t%12d\n", 
                buf, (size_t)0, edge_counts[i], name[k], batch_times[j * BATCH_COUNT + i] / loop_count, 
                batch_counts[j * BATCH_COUNT + i], a_param->parallel_count);    
            }
            else {
                fprintf(outfile, "BC\t%-30s\t%12zu\t%12zu\t%-30s\t%9.6f\t%12zu\n", buf, 
                (size_t)0, edge_counts[i], name[k], batch_times[j * BATCH_COUNT + i] / loop_count, 
                batch_counts[j * BATCH_COUNT + i]);
            }
        }  
    }
    free(buf);
}

/**
* Frees memory allocated for storing stats of algorithms
*
* @param batch_times Pointer to an array of times representing the time taken to process each batch
*
* @param total_times Pointer to an array of times representing the total time taken by an algorithm 
* to process batches
*
* @param batch_counts Pointer to an array of integers representing the count of triangles in a graph 
* after adding a batch
*
* @param edge_counts Pointer to an array of integers representing the count of edges
*
* @param edge_counts2 Pointer to a second array of integers representing an optional second 
* count of integers
*
* @param a_param Parameters of current run
*/
void free_TCD_stat_records(double ** batch_times, double ** total_times,
    UINT_t ** batch_counts, UINT_t ** edge_counts, UINT_t ** edge_counts2,
    const BenchmarkingParameters* const a_param)
{
    free(*batch_times);
    free(*total_times);
    free(*batch_counts);
    free(*edge_counts);
    if(a_param->insert_edges && a_param->withdraw_edges)
        free(edge_counts2);
}

/**
* Frees benchmarking resources
*
* @param batc Pointer to an array of pointers pointing to batches of edges
*
* @param batch Pointer to a pointer to a batch of edges 
*
* @param pre_static_state Pointer to a pointer pointing to static state
*
* @param a_param Parameters of the current run
*/
void free_TCD_benchmarking(GRAPH_eTYPE * batc, GRAPH_eTYPE * batch, 
    GRAPH_TYPE ** ptr_static_state, const BenchmarkingParameters * const a_param)
{
    const unsigned int BATCH_COUNT = a_param->batch_count;
    for(register UINT_t i = 0; i < BATCH_COUNT; ++i) {
        freeBatch(batc + i);
    }
    freeBatch(batch);
    free(batch);
    free(batc);
    free_graph(*ptr_static_state);
}

#define TRIAL_PROGRESS "░"
#define REAL_PROGRESS "▓"

int print_trial_progress() {
    int toret;
#ifdef _WIN32
    if (_isatty(_fileno(stdout))) {
        fflush(stdout);
        const int old_mode = _setmode(_fileno(stdout), _O_U16TEXT);
        toret = wprintf(L""TRIAL_PROGRESS);
        fflush(stdout);
        _setmode(_fileno(stdout), old_mode);
    }
    else {
#endif
        toret = printf(TRIAL_PROGRESS);
#ifdef _WIN32
    }
#endif
    return toret;
}

int print_reals_progress() {
    int toret;
#ifdef _WIN32
    if (_isatty(_fileno(stdout))) {
        fflush(stdout);
        const int old_mode = _setmode(_fileno(stdout), _O_U16TEXT);
        toret = wprintf(L""REAL_PROGRESS);
        fflush(stdout);
        _setmode(_fileno(stdout), old_mode);
    }
    else {
#endif
        toret = printf(REAL_PROGRESS);
#ifdef _WIN32
    }
#endif
    return toret;
}