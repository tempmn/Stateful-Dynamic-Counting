#include "merge.h"
#include "graph.h"
#include "tc_dynamic_state.h"
#include "types.h"
#include <omp.h>
#include <stdint.h>
#include <string.h>


void merge_sorted(void* const start_a, const UINT_t a_sz,
    void* const start_b, const UINT_t b_sz, void* const start_c, 
    const size_t step, comparator_cls orderer)
{
    UINT_t i, j, k;
    const char* const a = (const char* const)start_a;
    const char* const b = (const char* const)start_b;
    char* const c = (char* const)start_c;
    for (k = i = j = 0; i < a_sz; ++k) {
        if (j < b_sz &&
            orderer(a + step * i, b + step * j) == 1) {
            memcpy(c + k * step, b + j++ * step, step);
        }
        else {
            memcpy(c + k * step, a + i++ * step, step);
        }
    }
    if (j < b_sz) {
        memcpy(c + k * step, b + j * step, (b_sz - j) * step);
    }
}

/**
* Merging in parallel without tasks
* @note pre-req - the pieces are broken apart in divide/modulo fashion, not cyclical 
* divide fashion
*
* @param cache_individual_len The size of the entire partition which is the 
*          size of the sorted piece and size of padding combined
*
* @param dest_ordering Destination array to write entire sorted array to
*
* @param src_ordering Source array to take sorted pieces of array from
*
* @param step Size of each element in array
*
* @param state Length of each portion of array, placed in the first position of every
*          cache line. There must be partition entries
*
* @param partitions Number of pieces of array
*
* @param np Number of threads
*
* @param rank Rank of this thread
*
* @param orderer A comparison function that can perform total ordering
*
* @note threads are synchronized at the end of loop, all work is done by the 
*       time any thread exits. The parameters must be set completely by all threads
*       before any thread enters
*/
void mergepath_sorted_parallel(const UINT_t cache_individual_len, 
    void * const dest_ordering, void * const src_ordering, 
    const size_t step, size_t* const state, const UINT_t partition, 
    comparator_cls const orderer, const unsigned int np, 
    const unsigned int rank)
{
    unsigned long index;
#ifdef _WIN32
    if(sizeof(partition) == sizeof(uint64_t)) {
        _BitScanReverse64(&index, partition - 1);
    }
    else {
        _BitScanReverse(&index, partition - 1);
    }
    const UINT_t max_levels = index + 1;
#else
    if(sizeof(partition) == sizeof(uint64_t)) {
        index = sizeof(partition) * 8 - __builtin_clzll(partition - 1);
    }
    else {
        index = sizeof(partition) * 8 - __builtin_clz(partition - 1);
    }
    const UINT_t max_levels = index;
#endif
    void* starting_side;
    void* ending_side;
    if(!ODD(max_levels)) {
        #pragma omp for schedule(static)
        for(size_t i = 0; i < partition; ++i) {
            const size_t my_len = state[i * SIZE_CACHE_LEN];
            memcpy(dest_ordering + i * step * cache_individual_len,
                src_ordering + i * step * cache_individual_len,
                my_len * step);
        }
        starting_side = dest_ordering;
        ending_side = src_ordering;
    }
    else {
        starting_side = src_ordering;
        ending_side = dest_ordering;
    }
    UINT_t merges_this_level = partition / 2;
    UINT_t passes_this_level = partition % 2;
    for(size_t i = 0; i < max_levels; ++i) {
        const size_t multiples_of_cur_base = 1 << i;
        const size_t multiples_of_next_base = 1 << (i + 1);
        const size_t next_level_len = multiples_of_next_base;
        #pragma omp for schedule(static)
        for(size_t j = 0; j < merges_this_level + passes_this_level; ++j) {
            const size_t start_first = j * multiples_of_next_base;
            const size_t start_second = start_first + multiples_of_cur_base;
            if(j == merges_this_level) {
                memcpy(ending_side + start_first * cache_individual_len * step,
                    starting_side + start_first * cache_individual_len * step, 
                    state[start_first * SIZE_CACHE_LEN] * step);
            }
            else {
                merge_sorted(starting_side + 
                    start_first * cache_individual_len * step, 
                state[start_first * SIZE_CACHE_LEN], 
                starting_side + 
                    start_second * cache_individual_len * step, 
                state[start_second * SIZE_CACHE_LEN], 
                ending_side + start_first * 
                    step * cache_individual_len, 
                step, orderer);
                state[start_first * SIZE_CACHE_LEN] += state[start_second * SIZE_CACHE_LEN];
                state[start_second * SIZE_CACHE_LEN] = 0;
            }
        }
        const UINT_t temp_merges_this_level = merges_this_level;
        merges_this_level = (merges_this_level + passes_this_level) / 2;
        passes_this_level = (temp_merges_this_level + passes_this_level) % 2;
        void * temp = starting_side;
        starting_side = ending_side;
        ending_side = temp; 
    }
}

UINT_t merge_sorted_parallel(const UINT_t normal_len,
    void* const dest_ordering, void* const src_ordering,
    const size_t step, const UINT_t plus_one_check, const UINT_t plus_one_excess,
    const UINT_t a_rank, const UINT_t b_rank, const bool dest_is_src,
    const UINT_t cache_individual_len, comparator_cls const orderer)
{
    UINT_t a_sz, b_sz;
    if(a_rank == b_rank) {
        if(dest_is_src) {
            memcpy(src_ordering + a_rank * step * cache_individual_len, 
                dest_ordering + a_rank * step * cache_individual_len, 
                step * cache_individual_len);
        }
        else {
            memcpy(dest_ordering + a_rank * step * cache_individual_len, 
                src_ordering + a_rank * step * cache_individual_len, 
                step * cache_individual_len);
        }
        a_sz = (a_rank < plus_one_check) ?
            normal_len + plus_one_excess :
            normal_len;
        return a_sz;
    }
    const UINT_t half_1_1 = a_rank;
    const UINT_t half_1_2 = a_rank + (b_rank - a_rank) / 2;
    const UINT_t half_2_1 = half_1_2 + 1;
    const UINT_t half_2_2 = b_rank;
    if(a_rank + 1 == b_rank) {
        if(dest_is_src) {
            memcpy(src_ordering + a_rank * step * cache_individual_len, 
                dest_ordering + a_rank * step * cache_individual_len,
                step * 2 * cache_individual_len);
        }
        a_sz = (a_rank < plus_one_check) ?
                normal_len + plus_one_excess :
                normal_len;
        b_sz = (b_rank < plus_one_check) ?
                normal_len + plus_one_excess :
                normal_len;
    }
    else {
        #pragma omp task untied default(shared) priority(5) depend(out: src_ordering,a_sz)
        {
            a_sz = merge_sorted_parallel(normal_len, src_ordering, 
                dest_ordering, step, plus_one_check, plus_one_excess, 
                half_1_1, half_1_2, !dest_is_src, 
                cache_individual_len, orderer);
        }
        if(half_2_2 > half_2_1) {
            #pragma omp task untied default(shared) priority(5) depend(out: src_ordering,b_sz)
            {
                b_sz = merge_sorted_parallel(normal_len, src_ordering, 
                    dest_ordering, step, plus_one_check, plus_one_excess, 
                    half_2_1, half_2_2, !dest_is_src, 
                    cache_individual_len, orderer);
            }
        }
        else {
            if(dest_is_src) {
                memcpy(src_ordering + b_rank * step * cache_individual_len, 
                dest_ordering + b_rank * step * cache_individual_len,
                step * cache_individual_len);
            }
            b_sz = (b_rank < plus_one_check) ? 
                    normal_len + plus_one_excess :
                    normal_len;
        }
        #pragma omp taskwait
    }
    #pragma omp task untied default(shared) priority(5) depend(in: src_ordering,a_sz,b_sz)
    {
        merge_sorted(src_ordering + a_rank * step * cache_individual_len, a_sz, 
        src_ordering + half_2_1 * step * cache_individual_len, b_sz, 
        dest_ordering + a_rank * step * cache_individual_len, step, orderer);
    }
    #pragma omp taskwait
    return a_sz + b_sz;
}

/**
* Used for combining results from multiple threads each working on its own
* bucket. The overall goal is converting to frequency domain. This function
* helps in removing common items between threads.
* The items are sorted in a particular order which applies over all buckets
* The buckets are collapsed into one contiguous array by the end
*
* @param cache_individual_len Buckets are spaced apart by this amount
*
* @param src_dedup Array with buckets
*
* @param step Size of the object
*
* @param state contains size of each bucket at start and number of 
*       vertices in update by the end
*
* @param partition Number of buckets
*
* @param is_equal Testing for equality between items
*
* @param merge_fn Merging items between buckets in case of collision
*
* @param np Number of threads
*
* @param rank Rank of this thread
*
* @note Needs a barrier before but not after function
*/
void merge_buckets_parallel2(const UINT_t cache_individual_len, 
    void * const src_dedup, const size_t step, 
    size_t* const state, const UINT_t partition,
    comparator_cls const is_equal, merger_cls merge_fn,
    const unsigned int np, const unsigned int rank)
{
    unsigned long index;
#ifdef _WIN32
    if(sizeof(partition) == sizeof(uint64_t)) {
        _BitScanReverse64(&index, partition - 1);
    }
    else {
        _BitScanReverse(&index, partition - 1);
    }
    const UINT_t max_levels = index + 1;
#else
    if(sizeof(partition) == sizeof(uint64_t)) {
        index = sizeof(partition) * 8 - __builtin_clzll(partition - 1);
    }
    else {
        index = sizeof(partition) * 8 - __builtin_clz(partition - 1);
    }
    const UINT_t max_levels = index;
#endif
    const int old_active_levels = omp_get_max_active_levels();
    omp_set_max_active_levels(2);
    UINT_t merges_this_level = partition / 2;
    UINT_t passes_this_level = partition % 2;
    for(UINT_t i = 0; i < max_levels; ++i) {
        const UINT_t multiples_of_cur_base = 1 << i;
        const UINT_t multiples_of_next_base = 1 << (i + 1);
        const UINT_t next_level_len = multiples_of_next_base;
        #pragma omp for schedule(static)
        for(UINT_t j = 0; j < merges_this_level; ++j) {
            const UINT_t start_first = j * multiples_of_next_base;
            const UINT_t start_second = start_first + multiples_of_cur_base;
            void * last_of_first = src_dedup + 
                (start_first * cache_individual_len + 
                    state[start_first * SIZE_CACHE_LEN] - 1)
                * step;
            void * const first_of_second = src_dedup +
                start_second * cache_individual_len * step;
            if(state[start_first * SIZE_CACHE_LEN] > 0 && 
                state[start_second * SIZE_CACHE_LEN] > 0 && 
                is_equal(last_of_first, first_of_second) == 0) {
                merge_fn(first_of_second, last_of_first);
                state[start_first * SIZE_CACHE_LEN]--;
                last_of_first -= step;
            }
            memmove(last_of_first + step, first_of_second,
                step * state[start_second * SIZE_CACHE_LEN]);
            state[start_first * SIZE_CACHE_LEN] += 
                state[start_second * SIZE_CACHE_LEN];
        }
        const UINT_t temp_merges_this_level = merges_this_level;
        merges_this_level = (merges_this_level + passes_this_level) / 2;
        passes_this_level = (temp_merges_this_level + passes_this_level) % 2;
    }
    omp_set_max_active_levels(old_active_levels);
}

/**
* Helper function in implementing deduplication movement 
*
* @param my_partition_start Start of partition/piece of array
*
* @param start Starting index of range to move
*
* @param end Ending index of range to move
*
* @param shift Number of places to shift
*
* @param step Size of each element
*/
static void inline deduplicate_helper(void* my_partition_start, const size_t start,
    const size_t end, const size_t shift, const size_t step)
{
    void * copy_from = my_partition_start + start * step;
    void * copy_to = my_partition_start + (start - shift) * step;
    memmove(copy_to, copy_from, (end - start) * step);
}

/**
* Deduplicates sorted array in parallel 
*
* @param cache_individual_len The size of the entire partition which is the 
*          size of the piece and size of padding combined
*
* @param src_array Source/starting sorted array
*
* @param step Size of each element
*
* @param state Length of each portion/partition of sorted array
*
* @param partition Number of partitions
*
* @param is_equal Function to test for equality
*
* @param merge_fn Function to merge two items of array
*
* @param np Number of processes
*
* @param rank Rank of the thread executing this function
*
* @note threads are synchronized at the end of loop, all work is done by the 
*       time any thread exits. The parameters must be set completely by all threads
*       before any thread enters
*/
void pairwise_deduplicate_parallel(const UINT_t cache_individual_len, 
    void * const src_array, const size_t step, 
    size_t* const state, const UINT_t partition, 
    comparator_cls const is_equal, merger_cls merge_fn, const unsigned int np, 
    const unsigned int rank)
{
    #pragma omp for schedule(static)
    for(UINT_t i = 0; i < partition; ++i) {
        const UINT_t my_len = state[i * SIZE_CACHE_LEN];
        if(my_len > 1) {
            void* my_partition_start = src_array + i * 
                cache_individual_len * step;
            size_t j, k, start_move;
            bool immediate_prev_copy;
            start_move = 0;
            void* me, *follower;
            for(j = 0; j < my_len - 1; ++j) {
                me = my_partition_start + j * step;
                follower = my_partition_start + (j + 1) * step;
                if(is_equal(me, follower) == 0) {
                    merge_fn(follower, me);
                    immediate_prev_copy = true;
                    start_move = ++j;
                    break;
                }
            }
            for(k = 1; j < my_len - 1; ) {
                me = my_partition_start + j * step;
                follower = my_partition_start + (j + 1) * step;
                if(is_equal(me, follower) == 0) {
                    merge_fn(follower, me);
                    if(immediate_prev_copy) {
                        start_move++;
                    }
                    else{
                        deduplicate_helper(my_partition_start, 
                            start_move, j, k, step);
                        start_move = j + 1;
                    }
                    k++;
                }
                else {
                    immediate_prev_copy = false;
                }
                j++;
            }
            if(start_move != 0) {
                deduplicate_helper(my_partition_start, 
                    start_move, j + 1, k, step);
                state[i * SIZE_CACHE_LEN] -= k;
            }
        }
    }
    #pragma omp for schedule(static)
    for(UINT_t i = 0; i < partition - 1; ++i) {
        void * my_partition_end = src_array + (i * cache_individual_len 
            + state[i * SIZE_CACHE_LEN] - 1) * step;
        void * next_partition_start = src_array + (i + 1) * 
            cache_individual_len * step;
        if(is_equal(my_partition_end, next_partition_start) == 0) {
            merge_fn(next_partition_start, my_partition_end);
            state[i * SIZE_CACHE_LEN]--;
        }
    }
}

/**
* Finds the target vertex or higher in list
*
* @param list List of sorted elements
*
* @param start Lowest allowed entry
*
* @param end Highest allowed entry
*
* @param target Target element
*
* @returns Target vertex's entry or the higher entry
*/
size_t binary_search_partition(const void* list,
    const size_t start, const size_t end, const size_t step, 
    const void* target, comparator_cls orderer)
{
    register UINT_t s, e, mid;
    s = start;
    e = end;
    while(s < e) {
        mid = s + (e - s) / 2;
        if(orderer(list + mid * step, target) == 0)
            return mid;
        else if(orderer(list + mid * step, target) < 0)
            s =  mid + 1;
        else
            e = mid;
    }
    return s;
}

static int compare_vtxs_uint_vertexDegree(const void * a, const void * b)
{
    const vertexDegree_t* arg1 = (const vertexDegree_t*)a;
    const UINT_t* arg2 = (const UINT_t*)b;
    if(arg1->index < *arg2) return -1;
    if(arg1->index > *arg2) return 1;
    return 0;
}

static int compare_UINT_UINT_t(const void* a, const void* b)
{
    const UINT_t arg1 = *((const UINT_t*)a);
    const UINT_t arg2 = *((const UINT_t*)b);
    if(arg1 < arg2) return -1;
    if(arg1 > arg2) return 1;
    return 0;
}

/**
* Helps in merging/demerging GRAPH_eTYPE in GRAPH_TYPE
* Smashes update down into row part of CSR form.
*
* @param ptr_new_row New row to be generated. It is assumed to 
*       be unallocated
*
* @param numV Number of vertices in the graph represented by CSR
*
* @param update Update of edges
*
* @param idx_count Map of unique vertices in update to starting position
*
* @param number_vertices Number of unique vertices in update
*
* @param np Number of threads
*
* @param rank ID of this thread
*
* @note does not need barrier before or after function call
*/
static inline void flatten_smear_update(UINT_t** const ptr_new_row, 
    const UINT_t numV, const GRAPH_eTYPE * const update,  
    const vertexDegree_t * const idx_count, 
    const UINT_t number_vertices, const int np, const int rank)
{
    register UINT_t i, j;
    UINT_t * new_row;
    const vertexDegree_t* const idx_proper = idx_count + 1;
    const UINT_t range_start = (rank * (numV + 1)) / np;
    const UINT_t range_end = ((rank + 1) * (numV + 1)) / np;

    #pragma omp single copyprivate(new_row)
    {
        new_row = (UINT_t*)malloc((numV + 1) * sizeof(UINT_t));
        assert_malloc(new_row);
    }
    *ptr_new_row = new_row;

    const UINT_t in_update_start = binary_search_partition(
        idx_proper, 0, 
        number_vertices, sizeof(vertexDegree_t), 
        &range_start, 
        compare_vtxs_uint_vertexDegree);
    const UINT_t in_update_end = binary_search_partition(
        idx_proper, 0, 
        number_vertices, sizeof(vertexDegree_t), 
        &range_end, 
        compare_vtxs_uint_vertexDegree);
    // Find how the ranges of copying map to the update ranges
    // Maps to target or higher index

    const UINT_t prev_idx_deg = idx_count[in_update_start].degree;
    const UINT_t starter_lim = idx_proper[in_update_start].index == HERO ? 
        range_end : idx_proper[in_update_start].index;
    for(i = range_start; i < starter_lim; ++i) {
        new_row[i] = prev_idx_deg;
    }
    if(i < range_end) {
        new_row[i] = prev_idx_deg;
        for(j = in_update_start, i++; j < in_update_end; ++j) {
            const UINT_t new_prev_idx_deg = idx_proper[j].degree;
            for(; i <= min(range_end - 1, idx_proper[j + 1].index); ++i) {
                new_row[i] = new_prev_idx_deg;
            }
        }
        for(; i < range_end; ++i) {
            new_row[i] = new_row[i - 1];
        }
    }

    #pragma omp barrier
}

/** Merges GRAPH_TYPE and GRAPH_eTYPE in parallel
*
* @param exist Existing state of graph
*
* @param update New sequence of edges to add
*
* @param idx_count idx_count is sorted vertex-sum of degrees until vertex pairs
*
* @param number_vertices Number of unique vertices in update
*
* @param np Number of processes
*
* @param rank Rank of current process
*/
void merge_GRAPH_TYPE__GRAPH_eTYPE_parallel(GRAPH_TYPE * exist, 
    const GRAPH_eTYPE * const update,  
    const vertexDegree_t * const idx_count, 
    const UINT_t number_vertices, const int np, const int rank)
{
#define UINT_CACHE_LEN (TC_CACHE_LINE_SIZE/sizeof(UINT_t))
    register UINT_t i, j, k;
    const UINT_t edge_list_len = update->numEdges;
    const edge_t * const edge_list = update->edgeList;
    const UINT_t numV = exist->numVertices;
    const vertexDegree_t* idx_proper = idx_count + 1;
    UINT_t* new_row, * new_col;
    

    flatten_smear_update(&new_row, numV, 
        update, idx_count,
        number_vertices, np, rank);

    #pragma omp for schedule(static)
    for(UINT_t i = 0; i < numV + 1; ++i) {
        new_row[i] += exist->rowPtr[i];
    }

    #pragma omp single copyprivate(new_col)
    {
        new_col = (UINT_t*)malloc(new_row[numV] * sizeof(UINT_t));
        assert_malloc(new_col);
    }

    const UINT_t work_split_minlen_prov = new_row[numV] / np;
    const UINT_t work_split_lendec_prov = new_row[numV] % np;
    const UINT_t work_split_maxlen_prov = work_split_minlen_prov + 1;
    const UINT_t work_split_len_prov = (rank < work_split_lendec_prov) ? work_split_maxlen_prov : 
        work_split_minlen_prov;
    const UINT_t work_split_start_prov = (rank * work_split_maxlen_prov) - (rank > work_split_lendec_prov ? 
        rank - work_split_lendec_prov : 0);//(rank * new_row[numV]) / np;
    const UINT_t work_split_end_prov = work_split_start_prov + work_split_len_prov;
        //((rank + 1) * new_row[numV]) / np;

    // printf("%d: Work split boundaries: %d, %d\n", rank , work_split_start_prov, work_split_end_prov);
    // fflush(stdout);

    // #pragma omp barrier
    
    const UINT_t work_split_start = (rank > 0) ? 
        binary_search_partition(new_row, 0, 
            numV + 1, sizeof(UINT_t), 
            &work_split_start_prov, 
            compare_UINT_UINT_t) :
        0;
    const UINT_t work_split_end = (rank < np - 1) ?
        binary_search_partition(new_row, 0, 
            numV + 1, sizeof(UINT_t), 
            &work_split_end_prov, 
            compare_UINT_UINT_t) :
        numV;
    
    // printf("%d: Work split adjusted boundaries: %d, %d\n", rank, work_split_start, work_split_end);
    // fflush(stdout);
    // #pragma omp barrier
    
    const UINT_t rounded_work_split_start = (
        work_split_start > 0 &&
        new_row[work_split_start] - work_split_start_prov >
        work_split_start_prov - new_row[work_split_start - 1]) ?
        work_split_start - 1 : work_split_start;
    const UINT_t rounded_work_split_end = (
        work_split_end > 0 &&
        new_row[work_split_end] - work_split_end_prov >
        work_split_end_prov - new_row[work_split_end - 1]) ?
        work_split_end - 1 : work_split_end;
    
    // printf("%d: Rounded work split boundaries: %d, %d\n", rank, rounded_work_split_start, rounded_work_split_end);
    // fflush(stdout);
    // #pragma omp barrier
    
    const UINT_t update_worksplit_start = binary_search_partition(
        idx_proper, 0, number_vertices, 
        sizeof(vertexDegree_t), 
        &rounded_work_split_start, 
        compare_vtxs_uint_vertexDegree);
    
    const UINT_t starting_end = min(idx_proper[update_worksplit_start].index, 
        rounded_work_split_end);
    
    memcpy(new_col + new_row[rounded_work_split_start],
        exist->colInd + exist->rowPtr[rounded_work_split_start],
        (exist->rowPtr[starting_end] - exist->rowPtr[rounded_work_split_start]) * 
        sizeof(UINT_t));
    
    #pragma omp barrier
    
    for(i = starting_end, j = update_worksplit_start; 
        i < rounded_work_split_end; i = k) {
        for(UINT_t dummy;
            i < rounded_work_split_end &&
            i == idx_proper[j].index;
            i++, j++) {
            dummy = idx_count[j].degree;
            merge_list_dynamicTC(exist->colInd + exist->rowPtr[i], 
                update, 
                new_col + new_row[i], 
                &dummy, exist->rowPtr[i + 1] - exist->rowPtr[i]);
        }
        k = min(idx_proper[j].index, rounded_work_split_end);
        memcpy(new_col + new_row[i],
            exist->colInd + exist->rowPtr[i],
            (exist->rowPtr[k] - exist->rowPtr[i]) * 
            sizeof(UINT_t));
    }
    #pragma omp barrier
    #pragma omp sections
    {
        #pragma omp section
        {
            free(exist->rowPtr);
            exist->rowPtr = new_row;
            exist->numEdges = new_row[numV];
        }
        #pragma omp section
        {
            free(exist->colInd);
            exist->colInd = new_col;
        }
    }
#undef UINT_CACHE_LEN
}

/** DeMerges GRAPH_TYPE and GRAPH_eTYPE in parallel
*
* @param exist Existing state of graph
*
* @param update New sequence of edges to remove
*
* @param idx_count idx_count is sorted vertex-sum of degrees until vertex pairs
*
* @param number_vertices Number of unique vertices in update
*
* @param np Number of processes
*
* @param rank Rank of current process
*/
void demerge_GRAPH_TYPE__GRAPH_eTYPE_parallel(GRAPH_TYPE * exist, 
    const GRAPH_eTYPE * const update,  
    const vertexDegree_t * const idx_count, 
    const UINT_t number_vertices, const int np, const int rank)
{
#define UINT_CACHE_LEN (TC_CACHE_LINE_SIZE/sizeof(UINT_t))
    register UINT_t i, j, k;
    const UINT_t edge_list_len = update->numEdges;
    const edge_t * const edge_list = update->edgeList;
    const UINT_t numV = exist->numVertices;
    const vertexDegree_t* idx_proper = idx_count + 1;
    UINT_t* new_row, * new_col;
    

    flatten_smear_update(&new_row, numV, 
        update, idx_count,
        number_vertices, np, rank);

    #pragma omp for schedule(static)
    for(UINT_t i = 0; i < numV + 1; ++i) {
        new_row[i] = exist->rowPtr[i] - new_row[i];
    }

    #pragma omp single copyprivate(new_col)
    {
        new_col = (UINT_t*)malloc(new_row[numV] * sizeof(UINT_t));
        assert_malloc(new_col);
    }

    const UINT_t work_split_minlen_prov = new_row[numV] / np;
    const UINT_t work_split_lendec_prov = new_row[numV] % np;
    const UINT_t work_split_maxlen_prov = work_split_minlen_prov + 1;
    const UINT_t work_split_len_prov = (rank < work_split_lendec_prov) ? work_split_maxlen_prov : 
        work_split_minlen_prov;
    const UINT_t work_split_start_prov = (rank * work_split_maxlen_prov) - (rank > work_split_lendec_prov ? 
        rank - work_split_lendec_prov : 0);//(rank * new_row[numV]) / np;
    const UINT_t work_split_end_prov = work_split_start_prov + work_split_len_prov;
        //((rank + 1) * new_row[numV]) / np;

    // printf("%d: Work split boundaries: %d, %d\n", rank , work_split_start_prov, work_split_end_prov);
    // fflush(stdout);

    // #pragma omp barrier
    
    const UINT_t work_split_start = (rank > 0) ? 
        binary_search_partition(new_row, 0, 
            numV + 1, sizeof(UINT_t), 
            &work_split_start_prov, 
            compare_UINT_UINT_t) :
        0;
    const UINT_t work_split_end = (rank < np - 1) ?
        binary_search_partition(new_row, 0, 
            numV + 1, sizeof(UINT_t), 
            &work_split_end_prov, 
            compare_UINT_UINT_t) :
        numV;
    
    const UINT_t rounded_work_split_start = (
        work_split_start > 0 &&
        new_row[work_split_start] - work_split_start_prov >
        work_split_start_prov - new_row[work_split_start - 1]) ?
        work_split_start - 1 : work_split_start;
    const UINT_t rounded_work_split_end = (
        work_split_end > 0 &&
        new_row[work_split_end] - work_split_end_prov >
        work_split_end_prov - new_row[work_split_end - 1]) ?
        work_split_end - 1 : work_split_end;
    
    const UINT_t update_worksplit_start = binary_search_partition(
        idx_proper, 0, number_vertices, 
        sizeof(vertexDegree_t), 
        &rounded_work_split_start, 
        compare_vtxs_uint_vertexDegree);
    
    const UINT_t starting_end = min(idx_proper[update_worksplit_start].index, 
        rounded_work_split_end);
    
    memcpy(new_col + new_row[rounded_work_split_start],
        exist->colInd + exist->rowPtr[rounded_work_split_start],
        (exist->rowPtr[starting_end] - exist->rowPtr[rounded_work_split_start]) * 
        sizeof(UINT_t));
    
    #pragma omp barrier
    
    for(i = starting_end, j = update_worksplit_start; 
        i < rounded_work_split_end; i = k) {
        for(UINT_t dummy;
            i < rounded_work_split_end &&
            i == idx_proper[j].index;
            i++, j++) { 
            dummy = idx_count[j].degree;
            remove_ngh_GRAPH_TYPE__GRAPH_eTYPE(exist->colInd + exist->rowPtr[i], 
                update, 
                new_col + new_row[i], 
                &dummy, exist->rowPtr[i + 1] - exist->rowPtr[i]);
        }
        k = min(idx_proper[j].index, rounded_work_split_end);
        memcpy(new_col + new_row[i],
            exist->colInd + exist->rowPtr[i],
            (exist->rowPtr[k] - exist->rowPtr[i]) * 
            sizeof(UINT_t));
    }
    #pragma omp barrier
    #pragma omp sections
    {
        #pragma omp section
        {
            free(exist->rowPtr);
            exist->rowPtr = new_row;
            exist->numEdges = new_row[numV];
        }
        #pragma omp section
        {
            free(exist->colInd);
            exist->colInd = new_col;
        }
    }
#undef UINT_CACHE_LEN
}