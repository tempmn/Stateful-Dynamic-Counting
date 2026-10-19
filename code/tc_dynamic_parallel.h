#pragma once
#include "tc_dynamic_state.h"

/** Function takes in pointers to edge structs and compares them, first by src and then by dst */
static int compareEdge_t(const void *a, const void *b)
{
    edge_t arg1 = *(const edge_t *)a;
    edge_t arg2 = *(const edge_t *)b;
    if (arg1.src < arg2.src) return -1;
    if (arg1.src > arg2.src) return 1;
    if ((arg1.src == arg2.src) && (arg1.dst < arg2.dst)) return -1;
    if ((arg1.src == arg2.src) && (arg1.dst > arg2.dst)) return 1;
    return 0;
}

void find_dlt_base_alt2(GRAPH_eTYPE* update, edge_t** ptr_edge_list_dst,
    const UINT_t num_vertices, vertexDegree_t** ptr_vtx_count, 
    UINT_t** ptr_ptr_to_arrays, UINT_t *ptr_number_vertices, 
    const unsigned int np, const unsigned int rank);

UINT_t RepeatOld_parallel(ALT_DYNAMIC_STATE* pre_exist, GRAPH_eTYPE* update);

UINT_t W_RepeatOld_parallel(ALT_DYNAMIC_STATE* pre_exist, GRAPH_eTYPE* update);

UINT_t RelaxedDO_parallel(ALT_DYNAMIC_STATE* pre_exist, GRAPH_eTYPE* update);

UINT_t W_RelaxedDO_parallel(ALT_DYNAMIC_STATE* pre_exist, GRAPH_eTYPE* update);

UINT_t static_dynamic_parallel(GRAPH_TYPE* exist, GRAPH_eTYPE* update);

UINT_t W_static_dynamic_parallel(GRAPH_TYPE* exist, GRAPH_eTYPE* update);

UINT_t new_dynamicTC_parallel(DBTL_GRAPH* pre_exist, GRAPH_eTYPE * update);

UINT_t W_new_dynamicTC_parallel(DBTL_GRAPH* pre_exist, GRAPH_eTYPE * update);

#define STRINGIFY(x) #x,
#define INCREMENT(x) + 1
#define COUNT(x) (0 FOR_EACH(INCREMENT, x))

#define STATIC_DYNAMIC_PARALLEL_ALGS static_dynamic_parallel
#define STATIC_DYNAMIC_PARALLEL_ALGS_LEN COUNT(STATIC_DYNAMIC_PARALLEL_ALGS)
#define STATIC_DYNAMIC_PARALLEL_ALGS_STRINGS FOR_EACH(STRINGIFY, STATIC_DYNAMIC_PARALLEL_ALGS)

#define ALT_DYNAMIC_PARALLEL_ALGS RepeatOld_parallel, RelaxedDO_parallel
#define ALT_DYNAMIC_PARALLEL_ALGS_LEN COUNT(ALT_DYNAMIC_PARALLEL_ALGS)
#define ALT_DYNAMIC_PARALLEL_ALGS_STRINGS FOR_EACH(STRINGIFY,ALT_DYNAMIC_PARALLEL_ALGS)

#define DBTL_DYNAMIC_PARALLEL_ALGS new_dynamicTC_parallel
#define DBTL_DYNAMIC_PARALLEL_ALGS_LEN COUNT(DBTL_DYNAMIC_PARALLEL_ALGS)
#define DBTL_DYNAMIC_PARALLEL_ALGS_STRINGS FOR_EACH(STRINGIFY,DBTL_DYNAMIC_PARALLEL_ALGS)

#define W_STATIC_DYNAMIC_PARALLEL_ALGS W_static_dynamic_parallel
#define W_STATIC_DYNAMIC_PARALLEL_ALGS_LEN COUNT(W_STATIC_DYNAMIC_PARALLEL_ALGS)
#define W_STATIC_DYNAMIC_PARALLEL_ALGS_STRINGS FOR_EACH(STRINGIFY,W_STATIC_DYNAMIC_PARALLEL_ALGS)

#define W_ALT_DYNAMIC_PARALLEL_ALGS W_RepeatOld_parallel, W_RelaxedDO_parallel
#define W_ALT_DYNAMIC_PARALLEL_ALGS_LEN COUNT(W_ALT_DYNAMIC_PARALLEL_ALGS)
#define W_ALT_DYNAMIC_PARALLEL_ALGS_STRINGS FOR_EACH(STRINGIFY,W_ALT_DYNAMIC_PARALLEL_ALGS)

#define W_DBTL_DYNAMIC_PARALLEL_ALGS W_new_dynamicTC_parallel
#define W_DBTL_DYNAMIC_PARALLEL_ALGS_LEN COUNT(W_DBTL_DYNAMIC_PARALLEL_ALGS)
#define W_DBTL_DYNAMIC_PARALLEL_ALGS_STRINGS FOR_EACH(STRINGIFY,W_DBTL_DYNAMIC_PARALLEL_ALGS)