#include "tc_dynamic_parallel.h"
#include "graph.h"
#include "tc_dynamic_state.h"
#include "types.h"
#include "merge.h"
#include "benchmarking.h"
#include <omp.h>
#include <stdio.h>
#ifdef _WIN32
#include <intrin.h>
#endif
#include <string.h>
#include "tc_parallel.h"

#define EDGE_CACHE_LEN (TC_CACHE_LINE_SIZE / sizeof(edge_t))
#define UINT_CACHE_LEN (TC_CACHE_LINE_SIZE / sizeof(UINT_t))
#define VDGX_CACHE_LEN (TC_CACHE_LINE_SIZE / sizeof(vertexDegree_t))
#define BOOL_CACHE_LEN (TC_CACHE_LINE_SIZE / sizeof(bool))

typedef struct two_counts {
    UINT_t tc;
    UINT_t tc2;
} two_counts;

static int compare_vdgx_indices(const void * in1, const void * in2)
{
    const vertexDegree_t* a = (const vertexDegree_t*)in1;
    const vertexDegree_t* b = (const vertexDegree_t*)in2;
    if (a->index < b->index) return -1;
    if (a->index > b->index) return 1;
    return 0;
}

static void merge_vdgx_dedup(void * in1, const void* in2)
{
    vertexDegree_t* a = (vertexDegree_t*) in1;
    const vertexDegree_t* b = (const vertexDegree_t*) in2;
    a->degree += b->degree;
}

void find_dlt_base_alt2(GRAPH_eTYPE* update, edge_t** ptr_edge_list_dst,
    const UINT_t num_vertices, vertexDegree_t** ptr_vtx_count, 
    UINT_t** ptr_ptr_to_arrays, UINT_t *ptr_number_vertices, 
    const unsigned int np, const unsigned int rank)
{
    register UINT_t i, j, k, l;
    vertexDegree_t * idx_count, * proxy_idx_count;
    edge_t* edge_list_src, *edge_list_dst;
    UINT_t * reductor;
    size_t *state;
    UINT_t * ptr_to_arrays;
    UINT_t number_vertices;
    const UINT_t normal_len = update->numEdges / np;
    const UINT_t plus_one_check = update->numEdges % np;
    const UINT_t max_len = normal_len + 1;
    const UINT_t my_len = (rank < plus_one_check) ? max_len : normal_len;
    const UINT_t stride = (1 + ((2 * normal_len + 1) / EDGE_CACHE_LEN)) * EDGE_CACHE_LEN;
    #pragma omp single copyprivate(edge_list_src, edge_list_dst)
    {
        edge_list_src = (edge_t*)alloc_aligned(TC_CACHE_LINE_SIZE, 
            stride * np * sizeof(edge_t));
        assert_malloc(edge_list_src);
        edge_list_dst = (edge_t*)alloc_aligned(TC_CACHE_LINE_SIZE,
            stride * np * sizeof(edge_t));
        assert_malloc(edge_list_dst);
    }
    *ptr_edge_list_dst = edge_list_dst;
    edge_t* const assigner = edge_list_src + rank * stride;
    j = rank * max_len - (rank > plus_one_check ? rank - plus_one_check : 0);
    for(i = 0; i < my_len; ++i) {
        assigner[2 * i].dst = update->edgeList[j].src;
        assigner[2 * i].src = update->edgeList[j].dst;
        assigner[2 * i + 1] = update->edgeList[j];
        ++j;
    }
    qsort(assigner, my_len * 2, 
        sizeof(edge_t), compareEdge_t);
    const UINT_t max_counts = (1 + ((2 * normal_len + 1) / VDGX_CACHE_LEN))
        * VDGX_CACHE_LEN;
    #pragma omp single copyprivate(ptr_to_arrays,proxy_idx_count,state)
    {
        ptr_to_arrays = (UINT_t*)calloc(num_vertices, sizeof(UINT_t));
        assert_malloc(ptr_to_arrays);
        proxy_idx_count = (vertexDegree_t*)alloc_aligned(TC_CACHE_LINE_SIZE, 
            (np + 2) * max_counts * sizeof(vertexDegree_t));
        assert_malloc(proxy_idx_count);
        state = (size_t*)malloc(np * SIZE_CACHE_LEN * sizeof(size_t));
        assert_malloc(state);
    }
    *ptr_ptr_to_arrays = ptr_to_arrays;
    *ptr_vtx_count = proxy_idx_count;
    idx_count = proxy_idx_count + VDGX_CACHE_LEN;

    state[rank * SIZE_CACHE_LEN] = my_len * 2;

    #pragma omp barrier

    mergepath_sorted_parallel(stride, 
        edge_list_dst, edge_list_src, 
        sizeof(edge_t), state, np, 
        compareEdge_t, np, rank);
    
    vertexDegree_t* const local_idx_count = idx_count + rank * max_counts;
    const UINT_t contiguous_start = 2 * (rank * max_len - 
        (rank > plus_one_check ? rank - plus_one_check : 0));
    edge_t* const local_edges = edge_list_dst + contiguous_start;
    for(i = 0, k = 0; i < 2 * my_len; i = l, k++) {
        l = i;
        j = local_edges[i].src;
        while(l < 2 * my_len && local_edges[l].src == j)
            l++;
        local_idx_count[k].index = j;
        local_idx_count[k].degree = l - i;
    }
    state[rank * SIZE_CACHE_LEN] = k;
    while(k < max_counts) {
        local_idx_count[k].index = HERO;
        local_idx_count[k++].degree = HERO;
    }
    #pragma omp single copyprivate(reductor)
    {
        reductor = (UINT_t*)alloc_aligned(TC_CACHE_LINE_SIZE, 
            np * UINT_CACHE_LEN * sizeof(UINT_t));
        assert_malloc(reductor);
    }
    merge_buckets_parallel2(max_counts, 
        idx_count, sizeof(vertexDegree_t), 
        state, np, compare_vdgx_indices, 
        merge_vdgx_dedup, np, rank);
    number_vertices = state[0];
    *ptr_number_vertices = number_vertices;
    const UINT_t reduction_length = number_vertices / np;
    const UINT_t max_reduction_length = reduction_length + 1;
    const UINT_t reduction_check = number_vertices % np;
    const UINT_t my_reduction_len = (rank < reduction_check) ? 
        max_reduction_length : reduction_length;
    const UINT_t my_reduction_start = rank * max_reduction_length - 
        (rank > reduction_check ? rank - reduction_check : 0);
    vertexDegree_t* const local_reduction_div = idx_count + my_reduction_start;
    for(i = 1; i < my_reduction_len; ++i) {
        local_reduction_div[i].degree += local_reduction_div[i - 1].degree;
    }
    reductor[rank * UINT_CACHE_LEN] = local_reduction_div[i - 1].degree;
    #pragma omp barrier
    #pragma omp single
    {
        for(i = 1; i < np; ++i) {
            reductor[i * UINT_CACHE_LEN] += reductor[(i - 1) * UINT_CACHE_LEN];
        }
        idx_count[number_vertices].index = HERO;
        idx_count[number_vertices].degree = update->numEdges * 2;
        idx_count[-1].index = ZERO;
        idx_count[-1].degree = 0;
    }
    if(rank > 0) {
        for(i = 0; i < my_reduction_len; ++i) {
            local_reduction_div[i].degree += reductor[(rank - 1) * UINT_CACHE_LEN];
        }
    }
    #pragma omp single
    {
        free_aligned(edge_list_src);
    }
    for(i = 0; i < my_reduction_len; ++i) {
        const intmax_t prev_ind = ((intmax_t)i) - 1;
        ptr_to_arrays[local_reduction_div[i].index] = local_reduction_div[prev_ind].degree;
    }
    #pragma omp sections
    {
        #pragma omp section
        {
            free_aligned(reductor);
        }
        #pragma omp section
        {
            free(state);
        }
    }
}

UINT_t alt_task2u_parallel(ALT_DYNAMIC_STATE* pre_exist, 
    GRAPH_eTYPE* update, const vertexDegree_t* proxy_idx_count,
    const UINT_t* ptr_to_arrays, const UINT_t number_vertices, 
    const unsigned int np, const unsigned int rank)
{
    register UINT_t i, j, k; 
    register bool_t l;
    
    const UINT_t hashlen_stride = MAX_BITS(pre_exist->numVertices, bool_t);
    const UINT_t aligned_hashlen_stride = (1 + (hashlen_stride - 1) / BOLT_CACHE_LEN) *
        BOLT_CACHE_LEN;
    bool_t * restrict global_hashpad, * local_hashpad, * all_hashpad;
    const vertexDegree_t* idx_count = proxy_idx_count + VDGX_CACHE_LEN;
    const vertexDegree_t* prev_idx_count = idx_count - 1;
    register const UINT_t * const traversi = pre_exist->g1->rowPtr;
    register const UINT_t * const traversj = pre_exist->g1->colInd;
    const edge_t* edge_list_dst = update->edgeList;
    const UINT_t edge_list_len = update->numEdges;
    UINT_t * state, * proxy_state, max_nonuniq;

    #pragma omp single copyprivate(proxy_state,global_hashpad,all_hashpad)
    {
        proxy_state = (UINT_t*)malloc((np + 1) * UINT_CACHE_LEN * sizeof(UINT_t));
        assert_malloc(proxy_state);
        proxy_state[0] = 0;
        global_hashpad = (bool_t*)calloc(hashlen_stride, 
            sizeof(bool_t));
        assert_malloc(global_hashpad);
        all_hashpad = (bool_t*)alloc_aligned(TC_CACHE_LINE_SIZE, 
            np * aligned_hashlen_stride * sizeof(bool_t));
        assert_malloc(all_hashpad);
    }
    
    local_hashpad = all_hashpad + rank * aligned_hashlen_stride;

    for(i = 0; i < hashlen_stride; ++i) {
        local_hashpad[i] = ZERO;
    }
    
    #pragma omp for schedule(static)
    for(UINT_t counter = 0; counter < number_vertices; ++counter) {
        const UINT_t update_vtx = idx_count[counter].index;
        SET_BIT(update_vtx, local_hashpad, bool_t);
    }

    #pragma omp for schedule(static)
    for(UINT_t counter = 0; counter < hashlen_stride; ++counter) {
        global_hashpad[counter] = 0;
        for(i = 0; i < np; ++i) {
            global_hashpad[counter] |= all_hashpad[i * aligned_hashlen_stride
                + counter];
        }
    }

    for(i = 0; i < hashlen_stride; ++i) {
        local_hashpad[i] = ZERO;
    }
    
    #pragma omp for schedule(static)
    for(UINT_t counter = 0; counter < number_vertices; ++counter) {
        const UINT_t src = idx_count[counter].index;
        for(i = traversi[src]; i < traversi[src + 1]; ++i) {
            const UINT_t old_ngh = traversj[i];
            CHECK_BIT(l, old_ngh, global_hashpad, bool_t);
            if(l == 0) {
                SET_BIT(old_ngh, local_hashpad, bool_t);
            }
        }
    }

    UINT_t * restrict uniq_old;
    state = proxy_state + UINT_CACHE_LEN;
    state[rank * UINT_CACHE_LEN] = 0;

    #pragma omp for schedule(static) 
    for(UINT_t counter = 0; counter < hashlen_stride; ++counter) {
        global_hashpad[counter] = 0;
        for(i = 0; i < np; ++i) {
            global_hashpad[counter] |= all_hashpad[i * aligned_hashlen_stride
                + counter];
        }
        state[rank * UINT_CACHE_LEN] += BOLT_COUNTER(global_hashpad[counter]);
    }

    #pragma omp barrier
    #pragma omp single copyprivate(uniq_old)
    {
        for(i = 1; i < np; ++i) {
            state[i * UINT_CACHE_LEN] += state[(i - 1) * UINT_CACHE_LEN];
        }
        uniq_old = (UINT_t*)malloc(state[(np - 1) * UINT_CACHE_LEN] *
            sizeof(UINT_t));
        assert_malloc(uniq_old);
    }

    i = 0;
    const UINT_t offset = proxy_state[rank * UINT_CACHE_LEN];
    const UINT_t uniq_len = state[(np - 1) * UINT_CACHE_LEN];

    #pragma omp for schedule(static)
    for(UINT_t counter = 0; counter < hashlen_stride; ++counter) {
        bool_t read_val = global_hashpad[counter];
        while(read_val > 0) {
            BOLT_EXTRACTOR(read_val, j);
            k = j + counter * BYTE_SZ(read_val);
            uniq_old[offset + i++] = k;
            read_val &= (read_val - 1);
        }
    }

    for(i = 0; i < hashlen_stride; ++i) {
        local_hashpad[i] = ZERO;
    }
    UINT_t all_count = 0;
    
    #pragma omp for schedule(dynamic)
    for(UINT_t counter = 0; counter < number_vertices; ++counter) {
        UINT_t tc = 0;
        const UINT_t src = idx_count[counter].index;
        const edge_t* const local_edge_list = edge_list_dst + 
            prev_idx_count[counter].degree;
        const UINT_t n_V = idx_count[counter].degree - prev_idx_count[counter].degree;
        for(i = 0; i < n_V; ++i) {
            const UINT_t to_set = local_edge_list[i].dst;
            SET_BIT(to_set, local_hashpad, bool_t);
        }
        for(char state = 0; state != 7; ++state) {
            if(state == 4) {
                for(j = 0; j < n_V; ++j) {
                    const UINT_t ngh = local_edge_list[j].dst;
                    CLEAR_ALL_BIT(ngh, local_hashpad, bool_t);
                }
                for(j = traversi[src]; j < traversi[src + 1]; ++j) {
                    const UINT_t ngh = traversj[j];
                    SET_BIT(ngh, local_hashpad, bool_t);
                }
            }
            if((state & 2) == 0) {
                for(j = 0; local_edge_list[j].dst <= src && j < n_V; j++);
                for( ; j < n_V; ++j) {
                    const UINT_t ksrc = local_edge_list[j].dst;
                    if((state & 1) == 0) {
                        k = ptr_to_arrays[ksrc];
                        for(; k < edge_list_len && 
                            edge_list_dst[k].src == ksrc && 
                            edge_list_dst[k].dst <= ksrc; k++);
                        for(; k < edge_list_len && 
                            edge_list_dst[k].src == ksrc; ++k) {
                            const UINT_t chk = edge_list_dst[k].dst;
                            CHECK_BIT(l, chk, local_hashpad, bool_t);
                            if(l > 0)
                                tc++;
                        }
                    }
                    else {
                        for(k = traversi[ksrc]; k < traversi[ksrc + 1] && 
                            traversj[k] <= ksrc; ++k);
                        for(; k < traversi[ksrc + 1]; ++k)
                        {
                            const UINT_t chk = traversj[k];
                            CHECK_BIT(l, chk, local_hashpad, bool_t);
                            if(l > 0)
                                tc++;
                        }
                    }             
                }
            }
            else {
                for(j = traversi[src] ; j < traversi[src + 1] && 
                        traversj[j] <= src; ++j);
                for( ; j < traversi[src + 1]; ++j) {
                    const UINT_t ksrc = traversj[j];
                    if((state & 1) == 0) {
                        k = ptr_to_arrays[ksrc];
                        for(; k < edge_list_len && edge_list_dst[k].src == ksrc && 
                            edge_list_dst[k].dst <= ksrc; ++k);
                        for(; k < edge_list_len && edge_list_dst[k].src == ksrc; ++k)
                        {
                            const UINT_t chk = edge_list_dst[k].dst;
                            CHECK_BIT(l, chk, local_hashpad, bool_t);
                            if(l > 0)
                                tc++;
                        }
                    }
                    else {
                        for(k = traversi[ksrc]; k < traversi[ksrc + 1] && 
                            traversj[k] <= ksrc; ++k);
                        for(; k < traversi[ksrc + 1]; ++k)
                        {
                            const UINT_t chk = traversj[k];
                            CHECK_BIT(l, chk, local_hashpad, bool_t);
                            if(l > 0)
                                tc++;
                        }
                    }
                }
            }
        }
        for(j = traversi[src]; j < traversi[src + 1]; ++j) {
            const UINT_t ngh = traversj[j];
            CLEAR_ALL_BIT(ngh, local_hashpad, bool_t);
        }
        all_count += tc;
    }
    #pragma omp for schedule(dynamic)
    for(UINT_t counter = 0; counter < uniq_len; ++counter) {
        const UINT_t src = uniq_old[counter];
        UINT_t tc = 0;
        for(j = traversi[src]; j < traversi[src + 1]; ++j) {
            const UINT_t ngh = traversj[j];
            SET_BIT(ngh, local_hashpad, bool_t);
        }
        for(j = traversi[src] ; j < traversi[src + 1] && 
                traversj[j] <= src; ++j);
        for( ; j < traversi[src + 1]; ++j) {
            const UINT_t ksrc = traversj[j];
            k = ptr_to_arrays[ksrc];
            if(edge_list_dst[k].src == ksrc) {
                for(; k < edge_list_len && edge_list_dst[k].src == ksrc && 
                    edge_list_dst[k].dst <= ksrc; ++k);
                for(; k < edge_list_len && edge_list_dst[k].src == ksrc; ++k) {
                    const UINT_t chk = edge_list_dst[k].dst;
                    CHECK_BIT(l, chk, local_hashpad, bool_t);
                    if(l > 0)
                        tc++;
                }
            }
        }
        for(j = traversi[src]; j < traversi[src + 1]; ++j) {
            const UINT_t ngh = traversj[j];
            CLEAR_ALL_BIT(ngh, local_hashpad, bool_t);
        }
        all_count += tc;
    }
    #pragma omp sections
    {
        #pragma omp section
        {
            free_aligned(all_hashpad);
        }
        #pragma omp section
        {
            free(uniq_old);
        }
        #pragma omp section
        {
            free(global_hashpad);
        }
        #pragma omp section
        {
            free(proxy_state);
        }
    }
    return all_count;
}

#define BASE_ALT_TC_PARALLEL_STRUCTURE(task,overhaul) \
    register UINT_t i, j; \
    UINT_t tc = 0; \
    GRAPH_eTYPE container; \
    const UINT_t edge_list_len = update->numEdges * 2; \
    _Pragma ("omp parallel default(shared)") \
    { \
        UINT_t counter; \
        vertexDegree_t * idx_count, * proxy_idx_count; \
        edge_t* edge_list_dst; \
        UINT_t * ptr_to_arrays; \
        UINT_t number_vertices; \
        UINT_t* my_counts; \
        const UINT_t np = omp_get_num_threads(); \
        const UINT_t rank = omp_get_thread_num(); \
        find_dlt_base_alt2(update, &edge_list_dst, \
            pre_exist->numVertices, \
            &proxy_idx_count, &ptr_to_arrays, \
            &number_vertices, np, rank); \
        _Pragma ("omp single nowait") \
        { \
            container.edgeList = edge_list_dst; \
            container.numEdges = edge_list_len; \
            container.numVertices = number_vertices; \
        } \
        _Pragma ("omp single copyprivate(my_counts)") \
        { \
            my_counts = (UINT_t*)malloc(np * sizeof(UINT_t)); \
            assert_malloc(my_counts); \
            for(unsigned int i = 0; i < np; ++i) { \
                my_counts[i] = 0; \
            } \
        } \
        my_counts[rank] = task; \
        _Pragma ("omp barrier") \
        _Pragma ("omp for schedule(static) reduction(+:tc)") \
        for(counter = 0; counter < np; ++counter) { \
            tc += my_counts[counter]; \
        } \
        _Pragma ("omp barrier") \
        idx_count = proxy_idx_count + VDGX_CACHE_LEN; \
        overhaul; \
        _Pragma ("omp sections nowait") \
        { \
            _Pragma ("omp section") \
            { \
                free(ptr_to_arrays); \
            } \
            _Pragma ("omp section") \
            { \
                free_aligned(edge_list_dst); \
            } \
            _Pragma ("omp section") \
            { \
                free_aligned(proxy_idx_count); \
            } \
            _Pragma ("omp section") \
            { \
                free(my_counts); \
            } \
        } \
    } \
    return tc;

UINT_t RepeatOld_parallel(ALT_DYNAMIC_STATE* pre_exist, GRAPH_eTYPE* update)
{
    BASE_ALT_TC_PARALLEL_STRUCTURE(
        alt_task2u_parallel(pre_exist, 
            &container, proxy_idx_count,
            ptr_to_arrays, number_vertices, 
            np, rank), 
        merge_GRAPH_TYPE__GRAPH_eTYPE_parallel(pre_exist->g1,
            &container, idx_count - 1, 
            number_vertices, np, rank)
    );
}

UINT_t W_RepeatOld_parallel(ALT_DYNAMIC_STATE* pre_exist, GRAPH_eTYPE* update)
{
    register UINT_t i, j;
    UINT_t tc = 0;
    GRAPH_eTYPE container;
    const UINT_t edge_list_len = update->numEdges * 2;
    #pragma omp parallel default(shared)
    {
        UINT_t counter;
        vertexDegree_t * idx_count, * proxy_idx_count;
        edge_t* edge_list_dst;
        UINT_t * ptr_to_arrays;
        UINT_t number_vertices;
        UINT_t* my_counts;
        const UINT_t np = omp_get_num_threads();
        const UINT_t rank = omp_get_thread_num();
        find_dlt_base_alt2(update, &edge_list_dst,
            pre_exist->numVertices,
            &proxy_idx_count, &ptr_to_arrays,
            &number_vertices, np, rank);
        #pragma omp single nowait
        {
            container.edgeList = edge_list_dst;
            container.numEdges = edge_list_len;
            container.numVertices = number_vertices;
        }
        #pragma omp single copyprivate(my_counts)
        {
            my_counts = (UINT_t*)malloc(np * sizeof(UINT_t));
            assert_malloc(my_counts);
            for(unsigned int i = 0; i < np; ++i) {
                my_counts[i] = 0;
            }
        }
        idx_count = proxy_idx_count + VDGX_CACHE_LEN;
        demerge_GRAPH_TYPE__GRAPH_eTYPE_parallel(pre_exist->g1,
            &container, idx_count - 1, 
            number_vertices, np, rank);
        #pragma omp barrier
        my_counts[rank] = alt_task2u_parallel(pre_exist, 
            &container, proxy_idx_count,
            ptr_to_arrays, number_vertices, 
            np, rank);
        #pragma omp barrier
        #pragma omp for schedule(static) reduction(+:tc)
        for(counter = 0; counter < np; ++counter) {
            tc += my_counts[counter];
        }
        #pragma omp sections nowait
        {
            #pragma omp section
            {
                free(ptr_to_arrays);
            }
            #pragma omp section
            {
                free_aligned(edge_list_dst);
            }
            #pragma omp section
            {
                free_aligned(proxy_idx_count);
            }
            #pragma omp section
            {
                free(my_counts);
            }
        }
    }
    return tc;
}

void alt_task2_parallel(ALT_DYNAMIC_STATE* pre_exist, 
    GRAPH_eTYPE* update, const vertexDegree_t* proxy_idx_count,
    const UINT_t* ptr_to_arrays, const UINT_t number_vertices, 
    two_counts* out, const unsigned int np, const unsigned int rank)
{
    register UINT_t k, i, j;
    register bool_t l;
    UINT_t counter;
    UINT_t collide_prefix, nxti;
    const UINT_t edge_list_len = update->numEdges;
    const edge_t* edge_list_dst = update->edgeList;
    const UINT_t hashlen_stride = MAX_BITS(pre_exist->numVertices, bool_t);
    bool_t * restrict local_hashpad = (bool_t*)calloc(hashlen_stride, 
        sizeof(bool_t));
    assert_malloc(local_hashpad);
    const vertexDegree_t * const idx_count = proxy_idx_count + VDGX_CACHE_LEN;
    const vertexDegree_t * const prev_idx_count = idx_count - 1;

    register const UINT_t * const traversi = pre_exist->g1->rowPtr;
    register const UINT_t * const traversj = pre_exist->g1->colInd; 
    // Just convenience variable to reduce length of expressions
    
    #pragma omp for schedule(dynamic)
    for(counter = 0; counter < number_vertices; ++counter) {
        UINT_t tc = 0, tc2 = 0;
        const UINT_t src = idx_count[counter].index;
        const edge_t* const local_edge_list = edge_list_dst + prev_idx_count[counter].degree;
        const UINT_t n_V = idx_count[counter].degree - prev_idx_count[counter].degree;
        for(i = 0; i < n_V; ++i) {
            const UINT_t to_set = local_edge_list[i].dst;
            SET_BIT(to_set, local_hashpad, bool_t);
        }
        for(char state = 0; state != 6; ++state) {
            if(state == 4) {
                for(j = 0; j < n_V; ++j) {
                    const UINT_t ngh = local_edge_list[j].dst;
                    CLEAR_ALL_BIT(ngh, local_hashpad, bool_t);
                }
                for(j = traversi[src]; j < traversi[src + 1]; ++j) {
                    const UINT_t ngh = traversj[j];
                    SET_BIT(ngh, local_hashpad, bool_t);
                }
            }
            if((state & 2) == 0) {
                if((state & 1) + ((state & 2) >> 1) + ((state & 4) >> 2) != 2) {
                    for(j = 0; j < n_V && local_edge_list[j].dst <= src; j++);
                }
                else
                    j = 0;
                for( ; j < n_V; ++j) {
                    const UINT_t ksrc = local_edge_list[j].dst;
                    if((state & 1) == 0) {
                        if((state & 1) + ((state & 2) >> 1) + ((state & 4) >> 2) == 2) {
                            if(src == ksrc)
                                continue;
                            k = ptr_to_arrays[ksrc];
                            for(; k < edge_list_len && edge_list_dst[k].src == ksrc && edge_list_dst[k].dst <= ksrc; k++);
                            for(; k < edge_list_len && edge_list_dst[k].src == ksrc; ++k) {
                                const UINT_t chk = edge_list_dst[k].dst;
                                CHECK_BIT(l, chk, local_hashpad, bool_t);
                                if(l > 0)
                                    tc2++;
                            }
                        }
                        else {
                            k = ptr_to_arrays[ksrc];
                            for(; k < edge_list_len && edge_list_dst[k].src == ksrc && edge_list_dst[k].dst <= ksrc; k++);
                            for(; k < edge_list_len && edge_list_dst[k].src == ksrc; ++k) {
                                const UINT_t chk = edge_list_dst[k].dst;
                                CHECK_BIT(l, chk, local_hashpad, bool_t);
                                if(l > 0)
                                    tc++; 
                            }
                        }
                    }
                    else {
                        if((state & 1) + ((state & 2) >> 1) + ((state & 4) >> 2) == 2) {
                            for(k = traversi[ksrc]; k < traversi[ksrc + 1] && traversj[k] <= ksrc; ++k);
                            for(; k < traversi[ksrc + 1]; ++k) {
                                const UINT_t chk = traversj[k];
                                CHECK_BIT(l, chk, local_hashpad, bool_t);
                                if(l > 0)
                                    tc2++;
                            }
                        }
                        else {
                            for(k = traversi[ksrc]; k < traversi[ksrc + 1] && traversj[k] <= ksrc; ++k);
                            for(; k < traversi[ksrc + 1]; ++k) {
                                const UINT_t chk = traversj[k];
                                CHECK_BIT(l, chk, local_hashpad, bool_t);
                                if(l > 0)
                                    tc++;
                            }
                        }
                    }             
                }
            }
            else {
                if((state & 1) + ((state & 2) >> 1) + ((state & 4) >> 2) != 2)
                    for(j = traversi[src] ; j < traversi[src + 1] && traversj[j] <= src; ++j);
                else
                    j = traversi[src];
                for(; j < traversi[src + 1]; ++j) {
                    const UINT_t ksrc = traversj[j];
                    if((state & 1) == 0) {
                        if((state & 1) + ((state & 2) >> 1) + ((state & 4) >> 2) == 2) {
                            if(src == ksrc)
                                continue;
                            k = ptr_to_arrays[ksrc];
                            for(; k < edge_list_len && edge_list_dst[k].src == ksrc && edge_list_dst[k].dst <= ksrc; ++k);
                            for(; k < edge_list_len && edge_list_dst[k].src == ksrc; ++k)
                            {
                                const UINT_t chk = edge_list_dst[k].dst;
                                CHECK_BIT(l, chk, local_hashpad, bool_t);
                                if(l > 0)
                                    tc2++;
                            }
                        }
                        else {       
                            k = ptr_to_arrays[ksrc];
                            for(; k < edge_list_len && edge_list_dst[k].src == ksrc && edge_list_dst[k].dst <= ksrc; ++k);
                            for(; k < edge_list_len && edge_list_dst[k].src == ksrc; ++k) {
                                const UINT_t chk = edge_list_dst[k].dst;
                                CHECK_BIT(l, chk, local_hashpad, bool_t);
                                if(l > 0)
                                    tc++;
                            }
                        }
                    }
                    else 
                        if((state & 1) + ((state & 2) >> 1) + ((state & 4) >> 2) == 2) {
                            for(k = traversi[ksrc]; k < traversi[ksrc + 1] && traversj[k] <= ksrc; ++k);
                            for(; k < traversi[ksrc + 1]; ++k) {
                                const UINT_t chk = traversj[k];
                                CHECK_BIT(l, chk, local_hashpad, bool_t);
                                if(l > 0)
                                    tc2++;
                            }
                        }
                        else {
                            for(k = traversi[ksrc]; k < traversi[ksrc + 1] && traversj[k] <= ksrc; ++k);
                            for(; k < traversi[ksrc + 1]; ++k) {
                                const UINT_t chk = traversj[k];
                                CHECK_BIT(l, chk, local_hashpad, bool_t);
                                if(l > 0)
                                    tc++;
                            }
                        }
                }
            }
        }
        for(j = traversi[src]; j < traversi[src + 1]; ++j) {
            const UINT_t ngh = traversj[j];
            CLEAR_ALL_BIT(ngh, local_hashpad, bool_t);
        }
        out->tc += tc;
        out->tc2 += tc2;
    }
    free(local_hashpad);
}

#define BASE_ALT_TC_TC2_PARALLEL_STRUCTURE(task,overhaul) \
    register UINT_t i, j; \
    UINT_t tc = 0, tc2 = 0; \
    GRAPH_eTYPE container; \
    const UINT_t edge_list_len = update->numEdges * 2; \
    _Pragma ("omp parallel default(shared)") \
    { \
        UINT_t counter; \
        vertexDegree_t * idx_count, * proxy_idx_count; \
        edge_t* edge_list_dst; \
        UINT_t * ptr_to_arrays; \
        UINT_t number_vertices; \
        two_counts* my_counts; \
        const UINT_t np = omp_get_num_threads(); \
        const UINT_t rank = omp_get_thread_num(); \
        find_dlt_base_alt2(update, &edge_list_dst, \
            pre_exist->numVertices, \
            &proxy_idx_count, &ptr_to_arrays, \
            &number_vertices, np, rank); \
        _Pragma ("omp single nowait") \
        { \
            container.edgeList = edge_list_dst; \
            container.numEdges = edge_list_len; \
            container.numVertices = number_vertices; \
        } \
        _Pragma ("omp single copyprivate(my_counts)") \
        { \
            my_counts = (two_counts*)malloc(np * sizeof(two_counts)); \
            assert_malloc(my_counts); \
            for(unsigned int i = 0; i < np; ++i) { \
                my_counts[i] = (two_counts){.tc=0, .tc2=0}; \
            } \
        } \
        task; \
        _Pragma ("omp barrier") \
        _Pragma ("omp for schedule(static) reduction(+:tc,tc2)") \
        for(counter = 0; counter < np; ++counter) { \
            tc += my_counts[counter].tc; \
            tc2 += my_counts[counter].tc2; \
        } \
        _Pragma ("omp barrier") \
        idx_count = proxy_idx_count + VDGX_CACHE_LEN; \
        overhaul; \
        _Pragma ("omp sections nowait") \
        { \
            _Pragma ("omp section") \
            { \
                free(ptr_to_arrays); \
            } \
            _Pragma ("omp section") \
            { \
                free_aligned(edge_list_dst); \
            } \
            _Pragma ("omp section") \
            { \
                free_aligned(proxy_idx_count); \
            } \
            _Pragma ("omp section") \
            { \
                free(my_counts); \
            } \
        } \
    } \
    return tc + tc2 / 2;

UINT_t RelaxedDO_parallel(ALT_DYNAMIC_STATE* pre_exist, GRAPH_eTYPE* update) {
    BASE_ALT_TC_TC2_PARALLEL_STRUCTURE(
        alt_task2_parallel(pre_exist, &container, proxy_idx_count,
            ptr_to_arrays, number_vertices, 
            my_counts + rank, np, rank), 
        merge_GRAPH_TYPE__GRAPH_eTYPE_parallel(pre_exist->g1,
            &container, idx_count - 1, number_vertices, np, rank));
}
    

UINT_t W_RelaxedDO_parallel(ALT_DYNAMIC_STATE* pre_exist, GRAPH_eTYPE* update) {
    register UINT_t i, j;
    UINT_t tc = 0, tc2 = 0;
    GRAPH_eTYPE container;
    const UINT_t edge_list_len = update->numEdges * 2;
    #pragma omp parallel default(shared)
    {
        UINT_t counter;
        vertexDegree_t * idx_count, * proxy_idx_count;
        edge_t* edge_list_dst;
        UINT_t * ptr_to_arrays;
        UINT_t number_vertices;
        two_counts* my_counts;
        const UINT_t np = omp_get_num_threads();
        const UINT_t rank = omp_get_thread_num();
        find_dlt_base_alt2(update, &edge_list_dst,
            pre_exist->numVertices,
            &proxy_idx_count, &ptr_to_arrays,
            &number_vertices, np, rank);
        #pragma omp single nowait
        {
            container.edgeList = edge_list_dst;
            container.numEdges = edge_list_len;
            container.numVertices = number_vertices;
        }
        #pragma omp single copyprivate(my_counts)
        {
            my_counts = (two_counts*)malloc(np * sizeof(two_counts));
            assert_malloc(my_counts);
            for(unsigned int i = 0; i < np; ++i) {
                my_counts[i] = (two_counts){.tc=0, .tc2=0};
            }
        }
        idx_count = proxy_idx_count + VDGX_CACHE_LEN;
        demerge_GRAPH_TYPE__GRAPH_eTYPE_parallel(pre_exist->g1,
            &container, idx_count - 1, number_vertices, np, rank);
        #pragma omp barrier
        alt_task2_parallel(pre_exist, &container, proxy_idx_count,
            ptr_to_arrays, number_vertices, 
            my_counts + rank, np, rank);
        #pragma omp barrier
        #pragma omp for schedule(static) reduction(+:tc,tc2)
        for(counter = 0; counter < np; ++counter) {
            tc += my_counts[counter].tc;
            tc2 += my_counts[counter].tc2;
        }
        #pragma omp sections nowait
        {
            #pragma omp section
            {
                free(ptr_to_arrays);
            }
            #pragma omp section
            {
                free_aligned(edge_list_dst);
            }
            #pragma omp section
            {
                free_aligned(proxy_idx_count);
            }
            #pragma omp section
            {
                free(my_counts);
            }
        }
    }
    return tc + tc2 / 2;
}

UINT_t static_dynamic_parallel(GRAPH_TYPE* exist, GRAPH_eTYPE* update)
{
    GRAPH_eTYPE sorted_update;
    #pragma omp parallel default(shared)
    {
        edge_t * edge_list_dst;
        vertexDegree_t* proxy_idx_count;
        UINT_t * ptr_to_arrays;
        UINT_t number_vertices;
        const unsigned int np = omp_get_num_threads();
        const unsigned int rank = omp_get_thread_num();
        find_dlt_base_alt2(update, &edge_list_dst, 
            exist->numVertices,  
            &proxy_idx_count, 
            &ptr_to_arrays, 
            &number_vertices, 
            np, rank);
        #pragma omp sections
        {
            #pragma omp section
            {
                free(ptr_to_arrays);
            }
            #pragma omp section
            {
                sorted_update.edgeList = edge_list_dst;
                sorted_update.numEdges = 2 * update->numEdges;
                sorted_update.numVertices = update->numVertices;
            }
        }
        merge_GRAPH_TYPE__GRAPH_eTYPE_parallel(exist, &sorted_update, 
            proxy_idx_count + VDGX_CACHE_LEN - 1, number_vertices, np, rank);
        #pragma omp sections nowait
        {
            #pragma omp section
            {
                free_aligned(edge_list_dst);
            }
            #pragma omp section
            {
                free_aligned(proxy_idx_count);
            }
        }
    }
    const UINT_t tc = tc_MapJIK_P(exist);
    return tc;
}

UINT_t W_static_dynamic_parallel(GRAPH_TYPE* exist, GRAPH_eTYPE* update)
{
    GRAPH_eTYPE sorted_update;
    #pragma omp parallel default(shared)
    {
        edge_t * edge_list_dst;
        vertexDegree_t* proxy_idx_count;
        UINT_t * ptr_to_arrays;
        UINT_t number_vertices;
        const unsigned int np = omp_get_num_threads();
        const unsigned int rank = omp_get_thread_num();
        find_dlt_base_alt2(update, &edge_list_dst, 
            exist->numVertices,  
            &proxy_idx_count, 
            &ptr_to_arrays, 
            &number_vertices, 
            np, rank);
        #pragma omp sections
        {
            #pragma omp section
            {
                free(ptr_to_arrays);
            }
            #pragma omp section
            {
                sorted_update.edgeList = edge_list_dst;
                sorted_update.numEdges = 2 * update->numEdges;
                sorted_update.numVertices = update->numVertices;
            }
        }
        demerge_GRAPH_TYPE__GRAPH_eTYPE_parallel(exist, &sorted_update, 
            proxy_idx_count + VDGX_CACHE_LEN - 1, number_vertices, np, rank);
        #pragma omp sections nowait
        {
            #pragma omp section
            {
                free_aligned(edge_list_dst);
            }
            #pragma omp section
            {
                free_aligned(proxy_idx_count);
            }
        }
    }
    const UINT_t tc = tc_MapJIK_P(exist);
    return tc;
}

/**
* Organizes delta (new batch) for dynamic triangle counting
* Parallel version of get_dlt
*
* @param pre_exist State of dynamic triangle counting
*
* @param update Arriving new edges / deleted edges
*
* @param np Number of threads executing the parallel region
*
* @param rank The rank of the thread executing this function
*/
void get_dlt_parallel(DBTL_GRAPH* const pre_exist, const GRAPH_eTYPE* const update,
    const UINT_t np, const UINT_t rank)
{
#define EDGE_LEN (TC_CACHE_LINE_SIZE / sizeof(edge_t))
#define UINT_LEN (TC_CACHE_LINE_SIZE / sizeof(UINT_t))
#define BOOL_LEN (TC_CACHE_LINE_SIZE / sizeof(bool))
    register UINT_t i, j, k, l;
    edge_t* update_sorted, *update_working;
    const UINT_t normal_len = update->numEdges / np;
    const UINT_t plus_one_check = update->numEdges % np;
    const UINT_t individual_cache_len = (1 + ((normal_len) / EDGE_LEN))
        * EDGE_LEN;
    const UINT_t max_len = normal_len + 1;
    const UINT_t my_len = (rank < plus_one_check) ? max_len : normal_len;
    const UINT_t contiguous_start = rank * max_len - 
        (rank > plus_one_check ? rank - plus_one_check : 0);
    #pragma omp single copyprivate(update_sorted,update_working)
    {
        pre_exist->list_v_len = 0;
        pre_exist->list_v_sz = 1;
        update_sorted = (edge_t*)alloc_aligned(TC_CACHE_LINE_SIZE, 
            2 * np * individual_cache_len * sizeof(edge_t));
        assert_malloc(update_sorted);
        update_working = (edge_t*)alloc_aligned(TC_CACHE_LINE_SIZE,
            2 * np * individual_cache_len * sizeof(edge_t));
        assert_malloc(update_working);
    }
    const edge_t* update_e = update->edgeList + contiguous_start;
    const UINT_t current_start = 2 * rank * individual_cache_len;
    edge_t* const local_start = update_working + current_start;
    for(i = 0; i < my_len; ++i)
        local_start[i] = (edge_t) {.src=pre_exist->nid[update_e[i].src], 
            .dst=pre_exist->nid[update_e[i].dst]},
        local_start[my_len + i] = (edge_t){.src=pre_exist->nid[update_e[i].dst],
            .dst=pre_exist->nid[update_e[i].src]};
    qsort(local_start, 2 * my_len, sizeof(edge_t), 
        compareEdge_t);
    UINT_t * balancer, * counter, * reductor, *reductor2;
    #pragma omp single copyprivate(counter,reductor,reductor2)
    {
        counter = (UINT_t*)malloc(sizeof(UINT_t) * UINT_LEN * np);
        assert_malloc(counter);
        reductor = (UINT_t*)malloc(sizeof(UINT_t) * UINT_LEN * (np + 1));
        assert_malloc(reductor);
        reductor2 = (UINT_t*)malloc(sizeof(UINT_t) * UINT_LEN * (np + 1));
        assert_malloc(reductor2);
    }
    #pragma omp single
    {
        #pragma omp task untied default(shared) priority(5)
        {
            merge_sorted_parallel(2 * normal_len, 
                update_sorted, update_working, sizeof(edge_t), 
                plus_one_check, 2, 0, np - 1, 
                false, 2 * individual_cache_len, compareEdge_t);    
        }
    }
    edge_t* const local_duplicated_sorted = update_sorted + 2 * contiguous_start;
    for(i = 0, l = 0, j = HERO, k = 0; i < 2 * my_len; ++i, ++k) {
        if(i == 0 || j != local_duplicated_sorted[i].src) {
            if(i != 0) {
                local_start[l].src = j;
                local_start[l++].dst = k;
            }
            j = local_duplicated_sorted[i].src;
            k = 0;
        }
    }
    local_start[l].src = j;
    local_start[l].dst = k;
    counter[rank * UINT_LEN] = l;
    const UINT_t reduction_rank = rank + 1;
    if(my_len == 0) 
        local_start[0].src = HERO,
        local_start[0].dst = ZERO,
        reductor2[reduction_rank * UINT_LEN] = counter[rank * UINT_LEN];
    else
        reductor2[reduction_rank * UINT_LEN] = counter[rank * UINT_LEN] + 1;
    #pragma omp barrier
    // Deduplicate across buckets
    if(np > 1) {
        if(rank != 0) {
            const UINT_t prev_start = 2 * (rank - 1) * individual_cache_len;
            const UINT_t prev_len = counter[(rank - 1) * UINT_LEN];
            const edge_t* const left = update_working + prev_start + prev_len;
            const UINT_t disqualified = left->src;
            const UINT_t testee = local_start->src;
            if(disqualified == testee) {
                local_start->src = HERO;
            }
        }
    }
    UINT_t m;
    #pragma omp single copyprivate(m,balancer)
    {
        j = 0;
        for(i = 0; i < np; ++i) {
            j = max(j, counter[i * UINT_LEN]);
        }
        m = (1 + ((j + 1) / UINT_LEN)) * UINT_LEN;
        balancer = (UINT_t*)alloc_aligned(TC_CACHE_LINE_SIZE, 
            sizeof(UINT_t) * np * m);
        assert_malloc(balancer);
        reductor2[0] = 0;
    }
    const UINT_t cache_dedup_len = m;
    if(rank < np - 1) {
        i = rank;
        k = current_start + 2 * individual_cache_len;
        l = 0;
        if(
            (reductor2[(i + 1) * UINT_LEN] == 1 && 
            local_start[0].src != HERO) || 
            reductor2[(i + 1) * UINT_LEN] > 1
        ) {
            while(i < np - 1 && update_working[k].src == HERO) {
                
                i++;
                l += update_working[k].dst;                
                if(reductor2[(i + 1) * UINT_LEN] != 0)
                    break;
                k += 2 * individual_cache_len;
            }
            local_start[counter[rank * UINT_LEN]].dst += l;
        }
    }
    #pragma omp barrier
    if(local_start[0].src == HERO) {
        if(reductor2[reduction_rank * UINT_LEN] > 1) {
            memmove(local_start, local_start + 1, 
                sizeof(edge_t) * 
                (reductor2[reduction_rank * UINT_LEN] - 1));
        }
        reductor2[reduction_rank * UINT_LEN] = (
            reductor2[reduction_rank * UINT_LEN] == 0 ? 0 :
            reductor2[reduction_rank * UINT_LEN] - 1
        );
        counter[rank * UINT_LEN] = (
            counter[rank * UINT_LEN] == 0 ? 0 :
            counter[rank * UINT_LEN] - 1);
    }
    #pragma omp barrier
    UINT_t* const local_balancer = balancer + rank * cache_dedup_len;
    for(i = 0, j = 0, local_balancer[0] = 0; 
        i < reductor2[reduction_rank * UINT_LEN]; ++i) {
        j += local_start[i].dst;
        local_balancer[i + 1] = j;
    }
    reductor[reduction_rank * UINT_LEN] = local_balancer[i];
    #pragma omp barrier
    #pragma omp single
    {
        reductor[0] = 0;
        for(i = 1; i <= np; ++i) {
            reductor[UINT_LEN * i] += reductor[UINT_LEN * (i - 1)];
            reductor2[UINT_LEN * i] += reductor2[UINT_LEN * (i - 1)];
        }
        if(pre_exist->list_v_sz < reductor2[np * UINT_LEN]) {
            pre_exist->list_v = (UINT_t*)realloc(pre_exist->list_v, 
                reductor2[np * UINT_LEN] * sizeof(UINT_t));
            assert_malloc(pre_exist->list_v);
            pre_exist->list_v_sz = reductor2[np * UINT_LEN];
        }
        pre_exist->list_v_len = reductor2[np * UINT_LEN];
    }
    // for(i = 0; i <= counter[rank * UINT_LEN] + 1; ++i) {
    //     local_balancer[i] += reductor[rank * UINT_LEN];
    // }
    // #pragma omp barrier
    const UINT_t double_edges = 2 * update->numEdges;
    const UINT_t plus_one_check_outer = double_edges % np;
    const UINT_t plus_one_factor_outer = double_edges / np;
    const UINT_t plus_one_max_outer = plus_one_factor_outer + 1;
    const UINT_t candidate_start = plus_one_max_outer * rank - 
        (rank > plus_one_check_outer ? rank - plus_one_check_outer : 0);
    const UINT_t candidate_end =  plus_one_max_outer * (rank + 1) - 
        (rank + 1> plus_one_check_outer ? 
            rank + 1 - plus_one_check_outer : 0);
    i = 0;
    while(i < np &&
        reductor[i * UINT_LEN] <= candidate_start) {
        i++;
    }
    i -= 1;
    const UINT_t start_bucket = i;
    const UINT_t* const start_balancer = balancer + start_bucket * cache_dedup_len;
    k = reductor[start_bucket * UINT_LEN];
    j = 0;
    while(start_balancer[j] < candidate_start - k) {
        j++;
    }
    const UINT_t start_index = j;
    i = 0;
    while(i < np &&
        reductor[i * UINT_LEN] <= candidate_end) {
        i++;
    }
    i -= 1;
    const UINT_t end_bucket = i;
    const UINT_t* const end_balancer = balancer + end_bucket * cache_dedup_len;
    k = reductor[end_bucket * UINT_LEN];
    j = 0;
    while(end_balancer[j] < candidate_end - k) {
        j++;
    }
    const UINT_t end_index = j;
    for(i = start_bucket, j = start_index, l = start_index; 
        i <= end_bucket; i++) {
        for(; (i < end_bucket && j < reductor2[UINT_LEN * (i + 1)] - reductor2[UINT_LEN * i]) 
            || (i == end_bucket && j < end_index); j++) {
            if(update_working[2 * i * individual_cache_len + j].src != HERO) {
                const UINT_t v = update_working[2 * i * individual_cache_len + j].src;
                const UINT_t v_len = update_working[2 * i * individual_cache_len + j].dst;
                if(pre_exist->con_dlt[v] == NULL) {
                    pre_exist->con_dlt[v] = (UINT_t*)malloc(sizeof(UINT_t) * v_len);
                    assert_malloc(pre_exist->con_dlt[v]);
                    pre_exist->con_dlt_sz[v] = v_len;
                    pre_exist->deg_dlt[v] = 0;
                }
                else {
                    DBTL_vector_insert_test(pre_exist->con_dlt + v, 
                        v_len, pre_exist->deg_dlt + v, 
                        pre_exist->con_dlt_sz + v);
                }
                const UINT_t skip_length = reductor[i * UINT_LEN] 
                    + balancer[i * cache_dedup_len + j]; // if prev comment is coded, look here
                if(pre_exist->ngh_order == REORDER_LOWEST_DEGREE_FIRST) {
                    for(k = 0; k < v_len; ++k) {
                        pre_exist->con_dlt[v][k + pre_exist->deg_dlt[v]] = 
                            update_sorted[skip_length + k].dst;
                    }    
                }
                else {
                    for(k = 0; k < v_len; ++k) {
                        pre_exist->con_dlt[v][k + pre_exist->deg_dlt[v]] = 
                            update_sorted[skip_length + v_len - 1 - k].dst;
                    }
                }
                pre_exist->deg_dlt[v] += v_len;
                pre_exist->list_v[reductor2[i * UINT_LEN] + l] = v;
                l++;
            }
        }
        j = l = 0;
    }
    #pragma omp barrier
    #pragma omp single nowait
    {
        free_aligned(update_sorted);
        free_aligned(update_working);
        free_aligned(balancer);
        free(reductor);
        free(reductor2);
        free(counter);
    }
#undef BOOL_LEN
#undef UINT_LEN
#undef EDGE_LEN
}

/**
* Compares degree of vertices
*
* @param u Vertex u to be compared
*
* @param du Degree of vertex u
*
* @param v Vertex v to be compared
*
* @param dv Degree of vertex v
* @returns  1 if degree of u > degree of v else
*           -1 if degree of u < degree of v
* Tie breaks with labels, 0 if equal in all
*/
static inline int compare_aot_dbtl(const UINT_t u, const UINT_t du, 
    const UINT_t v, const UINT_t dv)
{
    if(du > dv) return 1;
    if(du < dv) return -1;
    if(u > v) return -1;
    if(u < v) return 1;
    return 0;
}

/**
* Compares degree of vertices (combining degree in update and degree in state)
*
* @param state DBTL state which had dlt members filled
*
* @param u Vertex u to be compared
*
* @param v Vertex v to be compared
*
* @returns  1 if degree of u > degree of v else
*           -1 if degree of u < degree of v
* Tie breaks with labels, 0 if equal in all
*/
static inline int shortcut_compare_aot_dbtl(const DBTL_GRAPH* const state,
    const UINT_t u, const UINT_t v)
{
    return compare_aot_dbtl(u, state->deg[u] + state->deg_dlt[u], 
        v, state->deg[v] + state->deg_dlt[v]);
}

UINT_t new_task_parallel(DBTL_GRAPH* pre_exist)
{
    const UINT_t n = pre_exist->n;
    char * restrict status = (char*)calloc(n, sizeof(char));
    assert_malloc(status);

    const UINT_t * restrict const deg = pre_exist->deg;
    const UINT_t * restrict const deg_dlt = pre_exist->deg_dlt;
    UINT_t * restrict const * restrict const con = pre_exist->con;
    UINT_t * restrict const * restrict const con_dlt = pre_exist->con_dlt;

    UINT_t tcount = 0;
    register UINT_t i, j, k, w;

    for(i = 0; i < n; ++i)
        status[i] = NONE;

    #pragma omp for schedule(dynamic)
    for(i = 0; i < pre_exist->list_v_len; ++i) {
        const UINT_t u = pre_exist->list_v[i];
        for(j = 0; j < deg[u]; ++j) {
            status[con[u][j]] = shortcut_compare_aot_dbtl(pre_exist, u, con[u][j]) > 0 ? 
                OLD_POS :
                OLD_NEG;
        }
        for(j = 0; j < deg_dlt[u]; ++j) {
            if(shortcut_compare_aot_dbtl(pre_exist, u, con_dlt[u][j]) > 0) {
                status[con_dlt[u][j]] = NEW_POS;
            }
        }
        for(j = 0; j < deg_dlt[u]; ++j) {
            const UINT_t v = con_dlt[u][j];
            if(shortcut_compare_aot_dbtl(pre_exist, u, v) < 0) continue;
            for(k = 0; k < deg[v]; ++k) {
                w = con[v][k];
                if(status[w] == NONE) continue;
                if(status[w] != NEW_POS) {
                    ++tcount;
                }
                else if (v > w) {
                    ++tcount;
                }
            }
            for(k = 0; k < deg_dlt[v]; ++k) {
                w = con_dlt[v][k];
                if(status[w] == OLD_POS) {
                    ++tcount;
                }
                else if (status[w] == NEW_POS && v > w) {
                    ++tcount;
                }
            }
        }

        for(j = 0; j < deg[u]; ++j) status[con[u][j]] = NONE;
        for(j = 0; j < deg_dlt[u]; ++j) status[con_dlt[u][j]] = NONE;
    }
    free(status);
    return tcount;
}

UINT_t new_dynamicTC_parallel(DBTL_GRAPH* pre_exist, GRAPH_eTYPE * update)
{
    UINT_t * counts, len;
    #pragma omp parallel default(shared)
    {
        const UINT_t np = omp_get_num_threads();
        const UINT_t rank = omp_get_thread_num();
        #pragma omp single nowait
        {
            counts = (UINT_t*)malloc(np * sizeof(UINT_t));
            assert_malloc(counts);
            len = np;
        }
        get_dlt_parallel(pre_exist, update, np, rank);
        counts[rank] = new_task_parallel(pre_exist);
    }
    DBTL_insert_reorder(pre_exist);
    UINT_t tcount = 0;
    for(register UINT_t i = 0; i < len; ++i)
        tcount += counts[i];
    free(counts);
    return tcount;
}

int compare_UINT_UINT(const void * a, const void * b)
{
    const UINT_t in1 = *(const UINT_t*)a;
    const UINT_t in2 = *(const UINT_t*)b;
    if (in1 < in2) return -1;
    if (in1 > in2) return 1;
    return 0;
}

UINT_t W_new_dynamicTC_parallel(DBTL_GRAPH* pre_exist, GRAPH_eTYPE * update)
{
    UINT_t * counts, len;
    #pragma omp parallel default(shared)
    {
        const UINT_t np = omp_get_num_threads();
        const UINT_t rank = omp_get_thread_num();
        #pragma omp single nowait
        {
            counts = (UINT_t*)malloc(np * sizeof(UINT_t));
            assert_malloc(counts);
            len = np;
        }
        get_dlt_parallel(pre_exist, update, np, rank);
    }
    GRAPH_TYPE* examiner = (GRAPH_TYPE*)malloc(sizeof(GRAPH_TYPE));
    assert_malloc(examiner);
    examiner->numVertices = pre_exist->n;
    UINT_t j = 0;
    for(UINT_t i = 0; i < pre_exist->n; ++i) {
        j += pre_exist->deg[i];
    }
    examiner->numEdges = j;
    allocate_graph(examiner);
    examiner->rowPtr[0] = 0;
    for(UINT_t i = 0; i < pre_exist->n; ++i) {
        const UINT_t mydeg = pre_exist->deg[pre_exist->nid[i]];
        examiner->rowPtr[i + 1] = mydeg + examiner->rowPtr[i];
        for(UINT_t k = 0; k < mydeg; ++k) {
            examiner->colInd[examiner->rowPtr[i] + k] = pre_exist->oid[pre_exist->con[pre_exist->nid[i]][k]];
        }
        qsort(examiner->colInd + examiner->rowPtr[i], 
            mydeg, sizeof(UINT_t), compare_UINT_UINT);
    }
    DBTL_withdraw_reorder(pre_exist);
    #pragma omp parallel default(shared)
    {
        const unsigned int np = omp_get_num_threads();
        const unsigned int rank = omp_get_thread_num();
        get_dlt_parallel(pre_exist, update, np, rank);
        #pragma omp barrier
        counts[rank] = new_task_parallel(pre_exist);
        #pragma omp barrier
        clear_dlt_parallel(pre_exist, false, np, rank);
    }
    UINT_t tcount = 0;
    for(register UINT_t i = 0; i < len; ++i)
        tcount += counts[i];
    free(counts);
    return tcount;
}