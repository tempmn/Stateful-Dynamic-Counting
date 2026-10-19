#include "tc_dynamic_state.h"
#include "graph.h"
#include "types.h"
#include "merge.h"
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#ifdef _WIN32
#include <intrin.h>
#endif

#ifndef min
#define min(x, y) ((x) < (y) ? (x) : (y))
#define min_del 1
#endif

#ifndef max
#define max(x, y) ((x) > (y) ? (x) : (y))
#define max_del 1
#endif

#define START_STATE 0x0D0E

/** Allocating space for batches of edges */
void allocateBatch(GRAPH_eTYPE* out, const UINT_t batch_sz, const UINT_t num_Vertices)
{
    out->numVertices = num_Vertices;
    out->numEdges = batch_sz;
    out->edgeList = calloc(batch_sz, sizeof(edge_t));
    assert_malloc(out->edgeList);
}

/**
* Arranges vertices in ascending order by degree
*
* @param a For compare condition in qsort
*
* @param b For compare condition in qsort
*/
static int compareDegreeAscending(const void * a, const void * b)
{
    const vertexDegree_t* i = (const vertexDegree_t*) a;
    const vertexDegree_t* j = (const vertexDegree_t*) b;
    if (i->degree < j->degree) return -1;
    if (i->degree > j->degree) return 1;
    if (i->index < j->index) return -2;
    if (i->index > j->index) return 2;
    return 0;
}

/**
* Arranges vertices in descending order by degree
*
* @param a For compare condition in qsort
*
* @param b For compare condition in qsort
*/
static int compareDegreeDescending(const void * a, const void * b)
{
    const vertexDegree_t* i = (const vertexDegree_t*) a;
    const vertexDegree_t* j = (const vertexDegree_t*) b;
    if (i->degree > j->degree) return -1;
    if (i->degree < j->degree) return 1;
    if (i->index > j->index) return -2;
    if (i->index < j->index) return 2;
    return 0;
}

/**
* Arranges vertices in ascending order by vertex number
*
* @param a For compare condition in qsort
*
* @param b For compare condition in qsort
*/
static int compareVertexAscending(const void * a, const void * b)
{
    const UINT_t * i = (const UINT_t *) a;
    const UINT_t * j = (const UINT_t *) b;
    if(*i < *j) return -1;
    if(*i > *j) return 1;
    return 0;
}

/**
* Arranges vertices in ascending order by vertex number
*
* @param a For compare condition in qsort
*
* @param b For compare condition in qsort
*/
static int compareVertexDescending(const void * a, const void * b)
{
    const UINT_t * i = (const UINT_t *) a;
    const UINT_t * j = (const UINT_t *) b;
    if(*i > *j) return -1;
    if(*i < *j) return 1;
    return 0;
}

/** XORshift 32 */
static inline uint32_t xor_shift_32(uint32_t* state)
{
    register uint32_t x;
    x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/** XORshift 64 */
static inline uint64_t xor_shift_64(uint64_t* state)
{
    register uint64_t x;
    x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/** Moving integer randoms to double (0,1] range */
static inline double xor_random(UINT_t* state)
{
    UINT_t a;
    if (sizeof(UINT_t) == sizeof(uint32_t))
    {
        return ((double)xor_shift_32((uint32_t*)state)) / HERO;
    }
    else 
    {
        return ((double)xor_shift_64((uint64_t*)state)) / HERO;
    }
}

static inline void combine_64(UINT_t* seed, const UINT_t b)
{
    UINT_t x = 0x9e3779b97f4a7c15ull + *seed + b;
    const UINT_t m = 0xe9846af9b1a615dull;
    x ^= x >> 32;
    x *= m;
    x ^= x >> 32;
    x *= m;
    x ^= x >> 28;
    *seed = x;
}

static inline void combine_32(UINT_t* seed, const UINT_t b)
{
    UINT_t x = 0x9e3779b9 + *seed + b;
    const UINT_t m1 = 0x21f0aaad;
    const UINT_t m2 = 0x735a2d97;
    x ^= x >> 16;
    x *= m1;
    x ^= x >> 15;
    x *= m2;
    x ^= x >> 15;
    *seed = x;
}

static inline void combine_hash(UINT_t * seed, const UINT_t b)
{
    switch(sizeof(UINT_t))
    {
        case sizeof(uint32_t):
            combine_32(seed, b);
            break;
        case sizeof(uint64_t):
            combine_64(seed, b);
            break;
        default:
            *seed ^= b;
    }
}

static inline UINT_t combine_fields(const UINT_t seed, 
    const UINT_t a, const UINT_t b)
{
    UINT_t to_ret = seed;
    combine_hash(&to_ret, a);
    combine_hash(&to_ret, b);
    return to_ret;
}

UINT_t edge_present(const UINT_t check1, const UINT_t check2, const UINT_t * chk,
    const UINT_t * chk2, const UINT_t hashlen)
{
    UINT_t i;
    UINT_t seed = (sizeof(UINT_t) == sizeof(uint64_t) ? 0xB7E151628AED2A6Aull : 0xB7E15162);
    for(i = combine_fields(seed, check1, check2) % (hashlen + 1); 
        (chk[i] != HERO || chk2[i] != HERO) && (chk[i] != check1 || chk2[i] != check2); 
        i = ((i + 1) & hashlen));
    return i;
}

/** Choose random edges in existing state for deletion 
*
* @param bag pointer to graph which is the source of all valid edges that may be emitted
*
* @param batch pointer to batch - output here. numEdges member gives the input
*
* @param reset resets random number generator
*
* @param in_state Carries over state of PRNG from different invocations
*
* @returns state of PRNG
*/
UINT_t choose_random_edges(const GRAPH_TYPE* const bag, GRAPH_eTYPE* batch, 
    const bool reset, const UINT_t in_state) {
    UINT_t state = in_state;
    if(reset)
        state = START_STATE;
    register UINT_t hashlen, src, dst, check1, check2;
    for(hashlen = 0; batch->numEdges > (1 << hashlen); ++hashlen);
    hashlen = (1 << (hashlen + 4)) - 1;
    UINT_t * restrict chk = (UINT_t*)malloc((hashlen + 1) * 
        sizeof(UINT_t));
    assert_malloc(chk);
    UINT_t * restrict chk2 = malloc((hashlen + 1) * 
        sizeof(UINT_t));
    assert_malloc(chk2);
    for(UINT_t i = 0; i <= hashlen; ++i) {
        chk[i] = HERO;
        chk2[i] = HERO;
    }
    for(register UINT_t gencount = 0, i; gencount < batch->numEdges; gencount++) {
        bool no_dupli = true;
        do {
            src = (UINT_t) (xor_random(&state) * bag->numVertices);
            const UINT_t src_deg = bag->rowPtr[src + 1] - bag->rowPtr[src];
            if(src_deg == 0)
                continue;
            const UINT_t dst_idx = (UINT_t) (xor_random(&state) * src_deg);
            dst = bag->colInd[bag->rowPtr[src] + dst_idx];
            check1 = min(src, dst);
            check2 = max(src, dst);
            i = edge_present(check1, check2, chk, chk2, hashlen);
            no_dupli &= !((chk[i] != check1) & (chk2[i] != check2));
        } while(no_dupli);
        chk[i] = check1;
        chk2[i] = check2;
        batch->edgeList[gencount] = (edge_t){.src=src,
        .dst=dst};
    }
    free(chk);
    free(chk2);
    return state;
#undef LINEAR_SEEK_C
}

/** Generates random edges between existing vertices.
*
* @param check pointer to starting graph
*
* @param batch pointer to batch - output here. numEdges member has the input
*
* @param reset resets random generator 
*
* @param in_state carries over state of PRNG from different invocations
*
* @returns state of PRNG
*/
UINT_t generate_random_edges(const GRAPH_TYPE* check, GRAPH_eTYPE* batch,
    const bool reset, const UINT_t in_state) {
    UINT_t state = in_state;
    if(reset)
        state = START_STATE;
    register bool no_dupli;
    register UINT_t hashlen;
    for(hashlen = 0; batch->numEdges > (1 << hashlen); ++hashlen);
    hashlen = (1 << (hashlen + 4)) - 1;
    UINT_t * restrict chk = calloc(hashlen + 1, sizeof(UINT_t));
    assert_malloc(chk);
    UINT_t * restrict chk2 = calloc(hashlen + 1, sizeof(UINT_t));
    assert_malloc(chk2);
    for(UINT_t i = 0; i <= hashlen; ++i) {
        chk[i] = HERO;
        chk2[i] = HERO;
    }
    for(register UINT_t gencount = 0, i; gencount < batch->numEdges; ) {
        const UINT_t src = (UINT_t) (xor_random(&state) * batch->numVertices);
        const UINT_t dst = (UINT_t) (xor_random(&state) * batch->numVertices);
        if(src == dst)
            continue;
        no_dupli = true;
        for(i = check->rowPtr[src]; i < check->rowPtr[src + 1]; ++i) {
            if(check->colInd[i] == dst) {
                no_dupli = false;
                break;
            }
        }
        if(!no_dupli)
            continue;
        const UINT_t check1 = min(src, dst);
        const UINT_t check2 = max(src, dst);
        i = edge_present(check1, check2, chk, chk2, hashlen);
        no_dupli &= (chk[i] != check1) & (chk2[i] != check2);
        if(no_dupli) {
            batch->edgeList[gencount].src = src;
            batch->edgeList[gencount++].dst = dst;
            chk[i] = check1;
            chk2[i] = check2;
        }
    }
    free(chk);
    free(chk2);
    return state;
}

/**
* Allocates and Initializes ALT_DYNAMIC_STATE
*
* @param ptr_superstates Pointer to a pointer to an ALT_DYNAMIC_STATEs
*
* @param originalGraph Pointer to graph. Left unmodified.
*
* @returns size of independentSetHash
*/
UINT_t initialize_alt_state(ALT_DYNAMIC_STATE ** ptr_superstates, const GRAPH_TYPE* const originalGraph)
{
    *ptr_superstates = (ALT_DYNAMIC_STATE*)malloc(sizeof(ALT_DYNAMIC_STATE));
    assert_malloc(*ptr_superstates);
    ALT_DYNAMIC_STATE* const superstates = *ptr_superstates;
    const UINT_t hash_len_vtx = originalGraph->numVertices;
    set_alt_state(superstates, originalGraph->numVertices, originalGraph->numEdges);
    reset_alt_state(superstates, originalGraph);
    return hash_len_vtx;
}

/**
* Sets up fields of ALT_DYNAMIC_STATE by allocating
*
* @param in State to allocate fields
*
* @param in2 Number of vertices in graph
*
* @param in3 Number of edges in graph
*/
void set_alt_state(ALT_DYNAMIC_STATE* in, const UINT_t in2, const UINT_t in3)
{
    in->g1 = (GRAPH_TYPE*)malloc(sizeof(GRAPH_TYPE));
    assert_malloc(in->g1);
    in->g1->numVertices = in2;
    in->g1->numEdges = in3;
    allocate_graph(in->g1);
}

/**
* Resets ALT_DYNAMIC_STATE to values consistent with input GRAPH_TYPE
* 
* @param in State to reset
*
* @param in2 Graph providing initial state after reset
*/
void reset_alt_state(ALT_DYNAMIC_STATE* const in, const GRAPH_TYPE* const in2)
{
    in->numVertices = in2->numVertices;
    in->g1->numVertices = in2->numVertices;
    in->g1->numEdges = in2->numEdges;
    reallocate_graph(in->g1);
    copy_graph(in2, in->g1);
}

/**
* Frees individual ALT_DYNAMIC_STATES
*
* @param in State to be freed
*/
void free_alt_state(ALT_DYNAMIC_STATE* in)
{
    free_graph(in->g1);
}

/**
* Allocates and initializes DBTL_GRAPH
*
* @param ptr_out Pointer to a pointer to a DBTL_GRAPH state
*
* @param originalGraph Graph setting initial state
*/
void initialize_DBTL_graph(DBTL_GRAPH** ptr_out, const GRAPH_TYPE* const originalGraph)
{
    *ptr_out = (DBTL_GRAPH*)malloc(sizeof(DBTL_GRAPH));
    assert_malloc(*ptr_out);
    DBTL_GRAPH* out = *ptr_out;
    set_DBTL_graph(out, originalGraph->numVertices, originalGraph->numEdges, 
            REORDER_HIGHEST_DEGREE_FIRST, REORDER_LOWEST_DEGREE_FIRST);
    reset_DBTL_graph(out, originalGraph);
}

/**
* Allocates memory for DBTL_GRAPH
*
* @param in Pointer to state
*
* @param n Number of vertices
*
* @param m Number of edges
*/
void set_DBTL_graph(DBTL_GRAPH* in, const UINT_t n, const UINT_t m,
    enum reorderDegree_t reorder_degree, enum reorderDegree_t ngh_ordering)
{
    in->n = n;
    in->m = m;
    in->con = (UINT_t**)calloc(n, sizeof(UINT_t*));
    assert_malloc(in->con);
    in->con_sz = (UINT_t*)calloc(n, sizeof(UINT_t));
    assert_malloc(in->con_sz);
    for(UINT_t i = 0; i < in->n; ++i) {
        in->con[i] = NULL;
        in->con_sz[i] = 0;
    }
    in->deg = (UINT_t*)calloc(n, sizeof(UINT_t));
    assert_malloc(in->deg);
    in->con_dlt = (UINT_t**)calloc(n, sizeof(UINT_t*));
    assert_malloc(in->con_dlt);
    in->con_dlt_sz = (UINT_t*)calloc(n, sizeof(UINT_t));
    assert_malloc(in->con_dlt_sz);
    for(UINT_t i = 0; i < in->n; ++i) {
        in->con_dlt[i] = NULL;
        in->con_dlt_sz[i] = 0;
    }
    in->deg_dlt = (UINT_t*)calloc(n, sizeof(UINT_t));
    assert_malloc(in->deg_dlt);
    in->nid = (UINT_t*)calloc(n, sizeof(UINT_t));
    assert_malloc(in->nid);
    in->oid = (UINT_t*)calloc(n, sizeof(UINT_t));
    assert_malloc(in->oid);
    in->hash = (UINT_t**)calloc(n, sizeof(UINT_t*));
    assert_malloc(in->hash);
    in->hash_sz = (UINT_t*)calloc(n, sizeof(UINT_t));
    assert_malloc(in->hash_sz);
    in->hash_col = (bool*)calloc(n, sizeof(bool));
    assert_malloc(in->hash_col);
    in->hash_dlt = (UINT_t**)calloc(n, sizeof(UINT_t*));
    assert_malloc(in->hash_dlt);
    in->hash_dlt_sz = (UINT_t*)calloc(n, sizeof(UINT_t));
    assert_malloc(in->hash_dlt_sz);
    in->hash_dlt_col = (bool*)calloc(n, sizeof(bool));
    assert_malloc(in->hash_dlt_col);
    in->list_v = (UINT_t*)malloc(sizeof(UINT_t));
    assert_malloc(in->list_v);
    in->list_v_sz = 1;
    in->ngh_order = ngh_ordering;
    in->vtx_order = reorder_degree;
}

/**
* Resets state to match passed in graph
*
* @param in State to be reset
*
* @param in2 Graph used in setting state
*/
void reset_DBTL_graph(DBTL_GRAPH* const in, const GRAPH_TYPE* const in2)
{
#define LINEAR_SET(var, array, mask, key, collide) for(var = (mask) & (key); \
                                                    array[var] != HERO; \
                                                    var = ((var + 1) & (mask)), collide++)
#define UINT_CACHE_LEN (TC_CACHE_LINE_SIZE / sizeof(UINT_t))
    omp_set_num_threads(omp_get_num_procs() / 2);
    const comparator_cls comparator = (in->vtx_order == REORDER_LOWEST_DEGREE_FIRST) ? 
        compareDegreeAscending : compareDegreeDescending;
    const comparator_cls comparator_ngh = (in->ngh_order == REORDER_LOWEST_DEGREE_FIRST) ? 
        compareVertexAscending : compareVertexDescending;
    #pragma omp parallel default(shared)
    {
        register UINT_t j, k, l;
        vertexDegree_t * ordering, *ordering_dest;
        size_t * state;
        
        const unsigned int np = omp_get_num_threads();
        const unsigned int rank = omp_get_thread_num();

        const UINT_t rounded_up_num_vertices = (in2->numVertices / np) + 
            ((in2->numVertices % np > 0) ? 1 : 0);
        const UINT_t align_cache = (1 + ((rounded_up_num_vertices - 1) / UINT_CACHE_LEN)) 
            * UINT_CACHE_LEN;
        
        #pragma omp single copyprivate(ordering,state,ordering_dest)
        {
            ordering = (vertexDegree_t*)alloc_aligned(TC_CACHE_LINE_SIZE,
                np * align_cache * sizeof(vertexDegree_t));
            assert_malloc(ordering);
            ordering_dest = (vertexDegree_t*)alloc_aligned(TC_CACHE_LINE_SIZE,
                np * align_cache * sizeof(vertexDegree_t));
            assert_malloc(ordering_dest);
            state = (size_t*)malloc(np * SIZE_CACHE_LEN * sizeof(size_t));
            assert_malloc(state);
        }

        #pragma omp for schedule(static, UINT_CACHE_LEN)
        for(UINT_t i = 0; i < in2->numVertices; ++i) 
            ordering[i] = (vertexDegree_t) { .index=i, .degree=in2->rowPtr[i + 1] - in2->rowPtr[i]};

        const UINT_t my_start = (rank * in2->numVertices) / np;
        const UINT_t my_end = ((rank + 1) * in2->numVertices) / np;
        state[rank * SIZE_CACHE_LEN] = my_end - my_start;
        vertexDegree_t* dest_place = ordering_dest + rank * align_cache;

        memcpy(dest_place, ordering + my_start, 
            state[rank * SIZE_CACHE_LEN] * sizeof(vertexDegree_t));
        qsort(dest_place, state[rank * SIZE_CACHE_LEN],
            sizeof(vertexDegree_t), comparator);
        #pragma omp barrier
        mergepath_sorted_parallel(align_cache, 
            ordering, ordering_dest, 
            sizeof(vertexDegree_t), state, np, comparator, np, rank);
        #pragma omp barrier
        #pragma omp for schedule(static, UINT_CACHE_LEN)
        for(UINT_t i = 0; i < in->n; ++i) 
            in->oid[i] = ordering[i].index, in->nid[in->oid[i]] = i;
        #pragma omp sections nowait
        {
            #pragma omp section
            {
                free_aligned(ordering_dest);
            }
            #pragma omp section
            {
                free_aligned(ordering);
            }
            #pragma omp section
            {
                free(state);
            }
        }
        #pragma omp for schedule(dynamic, UINT_CACHE_LEN)
        for(UINT_t i = 0; i < in2->numVertices; ++i) {
            if(in->con_dlt[i] != NULL)
                signal_free((void **)in->con_dlt + i);
            if(in->hash_dlt[i] != NULL)
                signal_free((void **)in->hash_dlt + i);
            in->con_dlt_sz[i] = 0;
            in->deg_dlt[i] = 0;
            const UINT_t old_i = in->oid[i];
            const UINT_t local_degree = in2->rowPtr[old_i + 1] - in2->rowPtr[old_i];
            in->deg[i] = local_degree;
            if(in->con[i] == NULL) {
                in->con[i] = (UINT_t *)calloc(local_degree, sizeof(UINT_t));
                assert_malloc(in->con[i]);
            }
            else {
                if(local_degree == 0) {
                    signal_free((void **)in->con + i);
                }
                else {
                    in->con[i] = (UINT_t *)realloc(in->con[i], 
                        local_degree * sizeof(UINT_t));
                    assert_malloc(in->con[i]);
                }
            }
            for(j = 0; (1 << j) < local_degree; ++j);
            const UINT_t requisition = (1 << (j + 4)) - 1;
            if(in->hash[i] == NULL) {
                in->hash[i] = (UINT_t*)malloc((requisition + 1) * sizeof(UINT_t));
            }
            else {
                in->hash[i] = (UINT_t*)realloc(in->hash[i],
                    (requisition + 1) * sizeof(UINT_t));
            }
            assert_malloc(in->hash[i]);
            for(j = 0; j < requisition + 1; ++j)
                in->hash[i][j] = HERO;
            in->hash_sz[i] = requisition;
            in->con_sz[i] = local_degree;
            for(j = 0, l = 0; j < local_degree; ++j) {
                const UINT_t graph_j = j + in2->rowPtr[old_i];
                const UINT_t v = in->nid[in2->colInd[graph_j]];
                in->con[i][j] = v;
                LINEAR_SET(k, in->hash[i], requisition, v, l);
                in->hash[i][k] = v;
            }
            if(l > 0)
                in->hash_col[i] = true;
            qsort(in->con[i], local_degree, 
                sizeof(UINT_t), comparator_ngh);
        }
        in->list_v_len = 0;
    }
#undef UINT_CACHE_LEN
#undef LINEAR_SET
}

/**
* Frees DBTL state
*
* @param in State to be freed
*/
void free_DBTL_graph(DBTL_GRAPH* in)
{
    for(UINT_t i = 0; i < in->n; ++i) {
        free(in->con[i]);
        free(in->hash[i]);
        if(in->hash_dlt[i] != NULL)
            free(in->hash_dlt[i]);
        if(in->con_dlt[i] != NULL)
            free(in->con_dlt[i]);
    }
    free(in->con);
    free(in->con_sz);
    free(in->con_dlt);
    free(in->con_dlt_sz);
    free(in->deg_dlt);
    free(in->hash);
    free(in->hash_col);
    free(in->hash_sz);
    free(in->hash_dlt);
    free(in->hash_dlt_col);
    free(in->hash_dlt_sz);
    free(in->deg);
    free(in->list_v);
    free(in->oid);
    free(in->nid);
}

/** Deletes a generated batch */
void freeBatch(GRAPH_eTYPE* batch)
{
    if(batch != NULL)
    {
        if(batch->edgeList != NULL)
            free(batch->edgeList);
    }
}

/**
* Clears delta (new batch) for dynamic triangle counting
*
* @param pre_exist State of dynamic triangle counting
*
* @param update Arriving new edges / deleted edges
*
* @param np Number of threads in parallel region
*
* @param rank Rank of thread executing the region
*/
void clear_dlt_parallel(DBTL_GRAPH * const in, const bool clear_hash,
    const UINT_t np, const UINT_t rank)
{
    register UINT_t i;
    const UINT_t split_len = in->list_v_len / np;
    const UINT_t plus_one_check = in->list_v_len % np;
    const UINT_t split_len_max = split_len + 1;
    const UINT_t my_len = (rank < plus_one_check) ? 
        split_len_max : split_len;
    const UINT_t my_start = rank * split_len_max - 
        (rank > plus_one_check ? rank - plus_one_check : 0);
    const UINT_t* const local_list_v = in->list_v + my_start;
    #pragma omp barrier
    for(i = 0; i < my_len; ++i) {
        const UINT_t v = local_list_v[i];
        if (in->con_dlt[v] != NULL)
            signal_free((void **)in->con_dlt + v);
        if (clear_hash)
            if(in->hash_dlt[v] != NULL)
                signal_free((void**)in->hash_dlt + v);
        in->con_dlt_sz[v] = 0;
        in->deg_dlt[v] = 0;
    }
    #pragma omp barrier
    #pragma omp single
    {
        in->list_v_len = 0;
    }
}

static inline UINT_t DBTL_insert_rorder_split_hellper(const UINT_t norm_vi_len, const UINT_t chk_vi_len, 
    UINT_t* const xdeg_dlt, UINT_t* const psums, const UINT_t psums_split, const UINT_t nvtx, const unsigned int rank, 
    const unsigned int nthreads)
{
    UINT_t vi, ti, target, counter;
    ti = nthreads - 1;
        
    while(ti > 0 && psums[ti] > psums_split)
        ti--;
    if(psums_split > psums[ti])
        ti++;
    const UINT_t max_vi_len = norm_vi_len + 1;
    /* We want to hit the lowest psums that is greater than psums_split/_end. 
    * In case of equality, we stick with the lower psum
    * because the xdeg_dlt we want will be in the lower bucket in that case
    */
    const UINT_t provisional_vi_start = ti * max_vi_len
        - (ti > chk_vi_len ? (ti - chk_vi_len) : 0);
    const UINT_t provisional_vi_start_len = (ti < chk_vi_len) ? 
        max_vi_len : norm_vi_len;

    target = psums_split - (ti > 0 ? psums[ti - 1] : 0);

    const UINT_t provisional_vi_start_search_end = provisional_vi_start + provisional_vi_start_len;

    for(vi = provisional_vi_start, counter = 0; 
        vi < provisional_vi_start_search_end && counter < target; 
        ++vi)
        counter += xdeg_dlt[vi];
    
    if(vi >= nvtx) {
        vi--;
    }
    else { 
        counter += xdeg_dlt[vi];
    }// vi = min(vi, provisional_vi_start_search_end - 1); 
    const UINT_t candidate_1_vi = vi;
    const UINT_t candidate_1_psums = counter;

    for(; 
        vi >= provisional_vi_start && (counter > target || (counter == target && xdeg_dlt[vi] == 0));
        --vi)
        counter -= xdeg_dlt[vi];
    
    const UINT_t candidate_2_vi = vi;
    const UINT_t candidate_2_psums = counter;

    return ((candidate_1_psums - target < target - candidate_2_psums) ?
        candidate_1_vi : candidate_2_vi) + 1;
}

void DBTL_insert_reorder(DBTL_GRAPH* in)
{
    UINT_t vi, nvtxs, nthreads, maxdegree=0, csrange=0;
    UINT_t *xdeg, *nxdeg, *psums, *nid, *oid, *xdeg_dlt, *nx_con_sz, *con_sz;
    UINT_t **adjncy, **nadjncy, **adjncy_dlt, *perm=NULL, *iperm=NULL, *chunkptr=NULL;
    UINT_t *gcounts;
    const bool comparator = (in->vtx_order == REORDER_LOWEST_DEGREE_FIRST);
    const comparator_cls comparator_ngh = (in->ngh_order == REORDER_LOWEST_DEGREE_FIRST) ? 
        compareVertexAscending : compareVertexDescending;

    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    nvtxs  = in->n;
    xdeg   = in->deg;
    adjncy = in->con;
    con_sz = in->con_sz;
    xdeg_dlt = in->deg_dlt;
    adjncy_dlt = in->con_dlt;
    nid    = in->nid;
    oid    = in->oid;

    /* Insertion phase*/
    #pragma omp parallel default(none) shared(nthreads, psums, nvtxs, xdeg, xdeg_dlt, \
        adjncy, adjncy_dlt, in)
    {
        #pragma omp single
        {
            nthreads = omp_get_num_threads();
            psums  = (UINT_t *)malloc(nthreads * sizeof(UINT_t));
            assert_malloc(psums);
        }
        
        const unsigned int rank = omp_get_thread_num();

        UINT_t vi, ti, counter, target;
        const UINT_t init_norm_vi_len = nvtxs / nthreads;
        const UINT_t init_max_vi_len  = init_norm_vi_len + 1;
        const UINT_t init_chk_vi_len  = nvtxs % nthreads;
        const UINT_t init_my_vi_len   = (rank < init_chk_vi_len) ? 
            init_max_vi_len : init_norm_vi_len;
        const UINT_t init_my_vi_start = rank * init_max_vi_len 
            - (rank > init_chk_vi_len ? (rank - init_chk_vi_len) : 0);
        const UINT_t init_my_vi_end   = init_my_vi_start + init_my_vi_len; 

        /* Split vertices amongst threads and count insertions */
        for(vi = init_my_vi_start, counter = 0; vi < init_my_vi_end; ++vi) {
            counter += xdeg_dlt[vi];
        }
        psums[rank] = counter;

        #pragma omp barrier
        #pragma omp single
        {
            for(ti = 1; ti < nthreads; ti++)
                psums[ti] += psums[ti - 1];
        }

        const UINT_t norm_psums_len = psums[nthreads - 1] / nthreads;
        const UINT_t max_psums_len = norm_psums_len + 1;
        const UINT_t chk_psums_len = psums[nthreads - 1] % nthreads;
        const UINT_t my_psums_len = (rank < chk_psums_len) ? max_psums_len : norm_psums_len;
        const UINT_t psums_start = rank * max_psums_len 
            - (rank > chk_psums_len ? (rank - chk_psums_len) : 0);
        const UINT_t psums_end = psums_start + my_psums_len;

        const UINT_t final_vi_start = (rank == 0) ? 0 : 
            DBTL_insert_rorder_split_hellper(init_norm_vi_len, init_chk_vi_len, 
            xdeg_dlt, psums, psums_start, nvtxs, rank, nthreads);
        
        const UINT_t final_vi_end = min(nvtxs, DBTL_insert_rorder_split_hellper(init_norm_vi_len, 
            init_chk_vi_len, xdeg_dlt, psums, psums_end, nvtxs, rank, nthreads));
        
        for(vi = final_vi_start; vi < final_vi_end; ++vi) {
            if(xdeg_dlt[vi] > 0) {
                DBTL_vector_insert_test(adjncy + vi, xdeg_dlt[vi],
                    xdeg + vi, in->con_sz + vi);
                for(counter = 0; counter < xdeg_dlt[vi]; ++counter) {
                    adjncy[vi][counter + xdeg[vi]] = adjncy_dlt[vi][counter];
                }
                xdeg[vi] += xdeg_dlt[vi];
            }
        }
        
        clear_dlt_parallel(in, true, nthreads, rank);
    }
    
    in->m += psums[nthreads - 1];

    nxdeg  = (UINT_t*)malloc(nvtxs * sizeof(UINT_t));
    assert_malloc(nxdeg);
    nadjncy = (UINT_t**)malloc(nvtxs * sizeof(UINT_t*));
    assert_malloc(nadjncy);
    nx_con_sz = (UINT_t*) malloc(nvtxs * sizeof(UINT_t));
    assert_malloc(nx_con_sz);
  
    perm  = (UINT_t *)malloc(nvtxs * sizeof(UINT_t));
    assert_malloc(perm);
    iperm = (UINT_t *)malloc(nvtxs * sizeof(UINT_t));
    assert_malloc(iperm);

    /* Determine maxdegree/csrange */
    #pragma omp parallel for schedule(static,4096) default(none) \
        shared(nvtxs, xdeg) \
        reduction(max: maxdegree)
    for (vi=0; vi<nvtxs; vi++) 
        maxdegree = max(maxdegree, xdeg[vi]);

    csrange = maxdegree+1;
    csrange = 16*((csrange+15)/16); /* get the per thread arrays to be alligned 
                                     at the start of the cache line */

    gcounts = (UINT_t *)malloc((nthreads*csrange) * sizeof(UINT_t));
    assert_malloc(gcounts);



    #pragma omp parallel default(none) \
        shared(nvtxs, nthreads, maxdegree, csrange, xdeg, adjncy, nxdeg, nadjncy, \
        perm, iperm, gcounts, psums, chunkptr, comparator, comparator_ngh, nid, oid, \
        nx_con_sz, con_sz) 
    {
        UINT_t vi, vistart, viend, vj, nedges, nchunks;
        UINT_t ti, di, ci, dstart, dend, counter, counter2;
        UINT_t *counts, *buffer;

        UINT_t ej, ejend, psum, chunksize;
        const int mytid = omp_get_thread_num();

        vistart = min(nvtxs, mytid*((nvtxs+nthreads-1)/nthreads));
        viend   = min(nvtxs, (mytid+1)*((nvtxs+nthreads-1)/nthreads));

        dstart = min(csrange, mytid*((csrange+nthreads-1)/nthreads));
        dend   = min(csrange, (mytid+1)*((csrange+nthreads-1)/nthreads));

        /* Compute the local counts */
        counts = gcounts + mytid*csrange;
        for (int i = 0 ; i < csrange ; i++)
            counts[i] = 0;

        if(comparator) {
            for (vi=vistart; vi<viend; vi++) 
                counts[xdeg[vi]]++;
        }
        else {
            for (vi=vistart; vi<viend; vi++) 
                counts[maxdegree - xdeg[vi]]++;
        }
        #pragma omp barrier

        /* Compute the partial sum of the range assigned to each thread */
        for (psum=0, ti=0; ti<nthreads; ti++) {
            counts = gcounts + ti*csrange;
            for (di=dstart; di<dend; di++) 
                psum += counts[di];
        }
        psums[mytid] = psum;
        #pragma omp barrier

        #pragma omp single
        for (ti=1; ti<nthreads; ti++)
            psums[ti] += psums[ti-1];
        #pragma omp barrier

        /* Compute the actual prefix sums of the range assigned to each thread.
        This is done from right to left to get it into the desired exclusive
        prefix sum stage. */
        psum = psums[mytid];
        for (counter = 1; counter <= dend - dstart; ++counter) { 
            di = dend - counter;
            counts = gcounts + (nthreads-1)*csrange;
            // ti=nthreads-1; ti>=0; ti--
            for (counter2 = 1; counter2 <= nthreads; ++counter2) {
                ti = nthreads - counter2;
                psum -= counts[di];
                counts[di] = psum;
                counts -= csrange;
            }
        }
        #pragma omp barrier

        /* Create the perm/iperm arrays and the nxadj array of the re-ordered graph */
        counts = gcounts + mytid*csrange;

        /* TODO: This can be optimized by pre-sorting the per-thread vertices according 
            to their degree and processing them in increasing degree order */
        if(comparator) {
            for (vi=vistart; vi<viend; vi++) {
                perm[vi] = counts[xdeg[vi]]++;
                nxdeg[perm[vi]] = xdeg[vi];
                iperm[perm[vi]] = vi;
            }
        }
        else {
            for (vi=vistart; vi<viend; vi++) {
                perm[vi] = counts[maxdegree - xdeg[vi]]++;
                nxdeg[perm[vi]] = xdeg[vi];
                iperm[perm[vi]] = vi;
            }
        }
        #pragma omp barrier

        #pragma omp barrier
        /* compute the local sums and their prefix sums */
        for (psum=0, vi=vistart; vi<viend; vi++)
            psum += nxdeg[vi];
        psums[mytid] = psum;
        #pragma omp barrier

        #pragma omp single
        for (ti=1; ti<nthreads; ti++)
            psums[ti] += psums[ti-1];
        #pragma omp barrier

        /* Compute the actual prefix sums of the nxadj[] array assigned to each thread.
        This is done from right to left to get it into the desired exclusive
        prefix sum stage. */
        // psum = psums[mytid];
        // if (mytid == nthreads-1)
        //   nxadj[nvtxs] = psum;
        // for (vi=viend-1; vi>=vistart; vi--) { 
        //   psum -= nxadj[vi];
        //   nxadj[vi] = psum;
        // }
        // #pragma omp barrier
      
        /* Compute the chunk-based partitioning of the work for the reordered/sorted graph */
        chunksize = 1+psums[nthreads-1]/(100*nthreads);
        for (nchunks=0, psum=0, vi=vistart; vi<viend; vi++) {
            if ((psum += nxdeg[vi]) >= chunksize) {
                nchunks++;
                psum = 0;
            }
        }
        psums[mytid] = nchunks+1;

        #pragma omp barrier
        #pragma omp single
        for (ti=1; ti<nthreads; ti++)
            psums[ti] += psums[ti-1];
        #pragma omp barrier

        #pragma omp single
        {
            chunkptr = (UINT_t *)malloc((psums[nthreads-1]+1) * sizeof(UINT_t));
            assert_malloc(chunkptr);
        }
        #pragma omp barrier

        nchunks = psums[mytid];
        chunkptr[nchunks] = viend;
        for (psum=0, counter = 1; counter <= viend - vistart; counter++) {
            vi = viend - counter;
            if ((psum += nxdeg[vi]) >= chunksize) {
                chunkptr[--nchunks] = vi;
                psum = 0;
            }
        }
        if (mytid == 0)
            chunkptr[0] = 0;
        #pragma omp barrier

        nchunks = psums[nthreads-1]; /* this is the total # of chunks */
        /*
        #pragma omp single
        {
        for (vi=0; vi<nchunks; vi++) {
            printf("%4d: %6d - %6d [%5d: %zd]\n", vi, chunkptr[vi], chunkptr[vi+1], 
                chunkptr[vi+1]-chunkptr[vi], nxadj[chunkptr[vi+1]]-nxadj[chunkptr[vi]]);
        }
        }
        #pragma omp barrier
        */

        /* create the reordered/sorted graph by processing the chunks in parallel */
        #pragma omp for schedule(dynamic, 1)
        for (counter = 1; counter <= nchunks; ++counter) {
            ci = nchunks - counter;
            for (vi=chunkptr[ci]; vi<chunkptr[ci+1]; vi++) {
                vj = iperm[vi];
                nid[vi] = oid[vj];
                nadjncy[vi] = adjncy[vj];
                nx_con_sz[vi] = con_sz[vj];
                buffer = nadjncy[vi];
                for (nedges=0; nedges < nxdeg[vi]; nedges++)  {
                    buffer[nedges] = perm[buffer[nedges]];
                }
                if (nedges > 1)
                    qsort(buffer, nedges, sizeof(UINT_t), comparator_ngh);
            }
        }

        #pragma omp for schedule(static)
        for(vi = 0; vi < nvtxs; ++vi) {
            oid[nid[vi]] = vi;
        }
    }

    in->oid = nid;
    in->nid = oid;
    in->con = nadjncy;
    in->deg = nxdeg;
    in->con_sz = nx_con_sz;
    free(con_sz);
    free(adjncy);
    free(xdeg);
    free(perm);
    free(iperm);
    free(gcounts);
    free(psums);
    free(chunkptr);
}

void DBTL_withdraw_reorder(DBTL_GRAPH* in)
{
    UINT_t vi, nvtxs, nthreads, maxdegree=0, csrange=0;
    UINT_t *xdeg, *nxdeg, *psums, *nid, *oid, *xdeg_dlt, *nx_con_sz, *con_sz;
    UINT_t **adjncy, **nadjncy, **adjncy_dlt, *perm=NULL, *iperm=NULL, *chunkptr=NULL;
    UINT_t *gcounts;
    const bool comparator = (in->vtx_order == REORDER_LOWEST_DEGREE_FIRST);
    const comparator_cls comparator_ngh = (in->ngh_order == REORDER_LOWEST_DEGREE_FIRST) ? 
        compareVertexAscending : compareVertexDescending;

    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    nvtxs  = in->n;
    xdeg   = in->deg;
    adjncy = in->con;
    con_sz = in->con_sz;
    xdeg_dlt = in->deg_dlt;
    adjncy_dlt = in->con_dlt;
    nid    = in->nid;
    oid    = in->oid;

    /* Insertion phase*/
    #pragma omp parallel default(none) shared(nthreads, psums, nvtxs, xdeg, xdeg_dlt, \
        adjncy, adjncy_dlt, in)
    {
        #pragma omp single
        {
            nthreads = omp_get_num_threads();
            psums  = (UINT_t *)malloc(nthreads * sizeof(UINT_t));
            assert_malloc(psums);
        }
        
        const unsigned int np = omp_get_num_threads();
        const unsigned int rank = omp_get_thread_num();

        UINT_t vi, ti, counter, target;
        const UINT_t init_norm_vi_len = nvtxs / nthreads;
        const UINT_t init_max_vi_len  = init_norm_vi_len + 1;
        const UINT_t init_chk_vi_len  = nvtxs % nthreads;
        const UINT_t init_my_vi_len   = (rank < init_chk_vi_len) ? 
            init_max_vi_len : init_norm_vi_len;
        const UINT_t init_my_vi_start = rank * init_max_vi_len 
            - (rank > init_chk_vi_len ? (rank - init_chk_vi_len) : 0);
        const UINT_t init_my_vi_end   = init_my_vi_start + init_my_vi_len; 

        /* Split vertices amongst threads and count insertions */
        for(vi = init_my_vi_start, counter = 0; vi < init_my_vi_end; ++vi) {
            counter += xdeg_dlt[vi];
        }
        psums[rank] = counter;

        #pragma omp barrier
        #pragma omp single
        {
            for(ti = 1; ti < nthreads; ti++)
                psums[ti] += psums[ti - 1];
        }

        const UINT_t norm_psums_len = psums[nthreads - 1] / nthreads;
        const UINT_t max_psums_len = norm_psums_len + 1;
        const UINT_t chk_psums_len = psums[nthreads - 1] % nthreads;
        const UINT_t my_psums_len = (rank < chk_psums_len) ? max_psums_len : norm_psums_len;
        const UINT_t psums_start = rank * max_psums_len 
            - (rank > chk_psums_len ? (rank - chk_psums_len) : 0);
        const UINT_t psums_end = psums_start + my_psums_len;

        const UINT_t final_vi_start = (rank == 0) ? 0 : 
            DBTL_insert_rorder_split_hellper(init_norm_vi_len, init_chk_vi_len, 
            xdeg_dlt, psums, psums_start, nvtxs, rank, nthreads);
        
        const UINT_t final_vi_end = min(nvtxs, DBTL_insert_rorder_split_hellper(init_norm_vi_len, 
            init_chk_vi_len, xdeg_dlt, psums, psums_end, nvtxs, rank, nthreads));
        
        for(vi = final_vi_start; vi < final_vi_end; ++vi) {
            if(xdeg_dlt[vi] > 0) {
                DBTL_vector_withdraw(adjncy + vi, xdeg + vi, 
                    in->con_sz + vi, adjncy_dlt[vi], 
                    xdeg_dlt[vi], in->ngh_order);
            }
        }

        #pragma omp barrier

        clear_dlt_parallel(in, false, np, rank);
    }
    
    in->m += psums[nthreads - 1];

    nxdeg  = (UINT_t*)malloc(nvtxs * sizeof(UINT_t));
    assert_malloc(nxdeg);
    nadjncy = (UINT_t**)malloc(nvtxs * sizeof(UINT_t*));
    assert_malloc(nadjncy);
    nx_con_sz = (UINT_t*) malloc(nvtxs * sizeof(UINT_t));
    assert_malloc(nx_con_sz);
  
    perm  = (UINT_t *)malloc(nvtxs * sizeof(UINT_t));
    assert_malloc(perm);
    iperm = (UINT_t *)malloc(nvtxs * sizeof(UINT_t));
    assert_malloc(iperm);

    /* Determine maxdegree/csrange */
    #pragma omp parallel for schedule(static,4096) default(none) \
        shared(nvtxs, xdeg) \
        reduction(max: maxdegree)
    for (vi=0; vi<nvtxs; vi++) 
        maxdegree = max(maxdegree, xdeg[vi]);

    csrange = maxdegree+1;
    csrange = 16*((csrange+15)/16); /* get the per thread arrays to be alligned 
                                     at the start of the cache line */

    gcounts = (UINT_t *)malloc((nthreads*csrange) * sizeof(UINT_t));
    assert_malloc(gcounts);



    #pragma omp parallel default(none) \
        shared(nvtxs, nthreads, maxdegree, csrange, xdeg, adjncy, nxdeg, nadjncy, \
        perm, iperm, gcounts, psums, chunkptr, comparator, comparator_ngh, nid, oid, \
        nx_con_sz, con_sz) 
    {
        UINT_t vi, vistart, viend, vj, nedges, nchunks;
        UINT_t ti, di, ci, dstart, dend, counter, counter2;
        UINT_t *counts, *buffer;

        UINT_t ej, ejend, psum, chunksize;
        const int mytid = omp_get_thread_num();

        vistart = min(nvtxs, mytid*((nvtxs+nthreads-1)/nthreads));
        viend   = min(nvtxs, (mytid+1)*((nvtxs+nthreads-1)/nthreads));

        dstart = min(csrange, mytid*((csrange+nthreads-1)/nthreads));
        dend   = min(csrange, (mytid+1)*((csrange+nthreads-1)/nthreads));

        /* Compute the local counts */
        counts = gcounts + mytid*csrange;
        for (int i = 0 ; i < csrange ; i++)
            counts[i] = 0;

        if(comparator) {
            for (vi=vistart; vi<viend; vi++) 
                counts[xdeg[vi]]++;
        }
        else {
            for (vi=vistart; vi<viend; vi++) 
                counts[maxdegree - xdeg[vi]]++;
        }
        #pragma omp barrier

        /* Compute the partial sum of the range assigned to each thread */
        for (psum=0, ti=0; ti<nthreads; ti++) {
            counts = gcounts + ti*csrange;
            for (di=dstart; di<dend; di++) 
                psum += counts[di];
        }
        psums[mytid] = psum;
        #pragma omp barrier

        #pragma omp single
        for (ti=1; ti<nthreads; ti++)
            psums[ti] += psums[ti-1];
        #pragma omp barrier

        /* Compute the actual prefix sums of the range assigned to each thread.
        This is done from right to left to get it into the desired exclusive
        prefix sum stage. */
        psum = psums[mytid];
        for (counter = 1; counter <= dend - dstart; ++counter) { 
            di = dend - counter;
            counts = gcounts + (nthreads-1)*csrange;
            // ti=nthreads-1; ti>=0; ti--
            for (counter2 = 1; counter2 <= nthreads; ++counter2) {
                ti = nthreads - counter2;
                psum -= counts[di];
                counts[di] = psum;
                counts -= csrange;
            }
        }
        #pragma omp barrier

        /* Create the perm/iperm arrays and the nxadj array of the re-ordered graph */
        counts = gcounts + mytid*csrange;

        /* TODO: This can be optimized by pre-sorting the per-thread vertices according 
            to their degree and processing them in increasing degree order */
        if(comparator) {
            for (vi=vistart; vi<viend; vi++) {
                perm[vi] = counts[xdeg[vi]]++;
                nxdeg[perm[vi]] = xdeg[vi];
                iperm[perm[vi]] = vi;
            }
        }
        else {
            for (vi=vistart; vi<viend; vi++) {
                perm[vi] = counts[maxdegree - xdeg[vi]]++;
                nxdeg[perm[vi]] = xdeg[vi];
                iperm[perm[vi]] = vi;
            }
        }
        #pragma omp barrier

        #pragma omp barrier
        /* compute the local sums and their prefix sums */
        for (psum=0, vi=vistart; vi<viend; vi++)
            psum += nxdeg[vi];
        psums[mytid] = psum;
        #pragma omp barrier

        #pragma omp single
        for (ti=1; ti<nthreads; ti++)
            psums[ti] += psums[ti-1];
        #pragma omp barrier

        /* Compute the actual prefix sums of the nxadj[] array assigned to each thread.
        This is done from right to left to get it into the desired exclusive
        prefix sum stage. */
        // psum = psums[mytid];
        // if (mytid == nthreads-1)
        //   nxadj[nvtxs] = psum;
        // for (vi=viend-1; vi>=vistart; vi--) { 
        //   psum -= nxadj[vi];
        //   nxadj[vi] = psum;
        // }
        // #pragma omp barrier
      
        /* Compute the chunk-based partitioning of the work for the reordered/sorted graph */
        chunksize = 1+psums[nthreads-1]/(100*nthreads);
        for (nchunks=0, psum=0, vi=vistart; vi<viend; vi++) {
            if ((psum += nxdeg[vi]) >= chunksize) {
                nchunks++;
                psum = 0;
            }
        }
        psums[mytid] = nchunks+1;

        #pragma omp barrier
        #pragma omp single
        for (ti=1; ti<nthreads; ti++)
            psums[ti] += psums[ti-1];
        #pragma omp barrier

        #pragma omp single
        {
            chunkptr = (UINT_t *)malloc((psums[nthreads-1]+1) * sizeof(UINT_t));
            assert_malloc(chunkptr);
        }
        #pragma omp barrier

        nchunks = psums[mytid];
        chunkptr[nchunks] = viend;
        for (psum=0, counter = 1; counter <= viend - vistart; counter++) {
            vi = viend - counter;
            if ((psum += nxdeg[vi]) >= chunksize) {
                chunkptr[--nchunks] = vi;
                psum = 0;
            }
        }
        if (mytid == 0)
            chunkptr[0] = 0;
        #pragma omp barrier

        nchunks = psums[nthreads-1]; /* this is the total # of chunks */
        /*
        #pragma omp single
        {
        for (vi=0; vi<nchunks; vi++) {
            printf("%4d: %6d - %6d [%5d: %zd]\n", vi, chunkptr[vi], chunkptr[vi+1], 
                chunkptr[vi+1]-chunkptr[vi], nxadj[chunkptr[vi+1]]-nxadj[chunkptr[vi]]);
        }
        }
        #pragma omp barrier
        */

        /* create the reordered/sorted graph by processing the chunks in parallel */
        #pragma omp for schedule(dynamic, 1)
        for (counter = 1; counter <= nchunks; ++counter) {
            ci = nchunks - counter;
            for (vi=chunkptr[ci]; vi<chunkptr[ci+1]; vi++) {
                vj = iperm[vi];
                nid[vi] = oid[vj];
                nadjncy[vi] = adjncy[vj];
                nx_con_sz[vi] = con_sz[vj];
                buffer = nadjncy[vi];
                for (nedges=0; nedges < nxdeg[vi]; nedges++)  {
                    buffer[nedges] = perm[buffer[nedges]];
                }
                if (nedges > 1)
                    qsort(buffer, nedges, sizeof(UINT_t), comparator_ngh);
            }
        }

        #pragma omp for schedule(static)
        for(vi = 0; vi < nvtxs; ++vi) {
            oid[nid[vi]] = vi;
        }
    }

    in->oid = nid;
    in->nid = oid;
    in->con = nadjncy;
    in->deg = nxdeg;
    in->con_sz = nx_con_sz;
    free(con_sz);
    free(adjncy);
    free(xdeg);
    free(perm);
    free(iperm);
    free(gcounts);
    free(psums);
    free(chunkptr);
}