#pragma once

#include "types.h"
#include "graph.h"
#include <omp.h>

#define NONE 0
#define NEW_POS 1
#define OLD_POS 2
#define OLD_NEG 3

typedef struct ALT_DYNAMIC_STATE
{
    UINT_t numVertices;
    GRAPH_TYPE* g1;
} ALT_DYNAMIC_STATE;

typedef struct DBTL_GRAPH
{
    UINT_t n;
    UINT_t m;
    UINT_t **con;
    UINT_t *con_sz;
    UINT_t *oid, *nid;
    UINT_t *deg;
    UINT_t **con_dlt;
    UINT_t *con_dlt_sz;
    UINT_t *deg_dlt;
    UINT_t *list_v;
    UINT_t list_v_len;
    UINT_t list_v_sz;
    UINT_t **hash;
    UINT_t *hash_sz;
    bool *hash_col;
    UINT_t **hash_dlt;
    UINT_t *hash_dlt_sz;
    bool *hash_dlt_col;
    enum reorderDegree_t vtx_order;
    enum reorderDegree_t ngh_order;
} DBTL_GRAPH;

void allocateBatch(GRAPH_eTYPE* out, const UINT_t batch_sz, const UINT_t num_Vertices);

UINT_t choose_random_edges(const GRAPH_TYPE* const bag, GRAPH_eTYPE* batch, 
    const bool reset, const UINT_t in_state);

UINT_t generate_random_edges(const GRAPH_TYPE* check, GRAPH_eTYPE* batch,
    const bool reset, const UINT_t in_state);

UINT_t initialize_alt_state(ALT_DYNAMIC_STATE ** ptr_superstates, const GRAPH_TYPE* const originalGraph);

void set_alt_state(ALT_DYNAMIC_STATE* in, const UINT_t in2, const UINT_t in3);

void reset_alt_state(ALT_DYNAMIC_STATE* const in, const GRAPH_TYPE* const in2);

void free_alt_state(ALT_DYNAMIC_STATE* in);

void initialize_DBTL_graph(DBTL_GRAPH** ptr_out, const GRAPH_TYPE* const originalGraph);

void set_DBTL_graph(DBTL_GRAPH* in, const UINT_t n, const UINT_t m,
    enum reorderDegree_t reorder_degree, enum reorderDegree_t ngh_ordering);

void reset_DBTL_graph(DBTL_GRAPH* const in, const GRAPH_TYPE* const in2);

void free_DBTL_graph(DBTL_GRAPH* in);

void freeBatch(GRAPH_eTYPE* batch);

void clear_dlt_parallel(DBTL_GRAPH * const in, const bool clear_hash,
    const UINT_t np, const UINT_t rank);

void DBTL_insert_reorder(DBTL_GRAPH* in);

void DBTL_withdraw_reorder(DBTL_GRAPH* in);