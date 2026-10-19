#pragma once

#include "graph.h"
#include "types.h"
#include "tc_dynamic_state.h"

static void keepOneEdge_t(void *a, const void *b)
{
    edge_t* arg1 = (edge_t*)a;
    const edge_t* arg2 = (const edge_t*)b;
    *arg1 = *arg2;
}

void mergepath_sorted_parallel(const UINT_t cache_individual_len, 
    void * const dest_ordering, void * const src_ordering, 
    const size_t step, size_t* const state, const UINT_t partition, 
    comparator_cls const orderer, const unsigned int np, 
    const unsigned int rank);

UINT_t merge_sorted_parallel(const UINT_t normal_len,
    void* const dest_ordering, void* const src_ordering,
    const size_t step, const UINT_t plus_one_check, const UINT_t plus_one_excess,
    const UINT_t a_rank, const UINT_t b_rank, const bool dest_is_src,
    const UINT_t cache_individual_len, comparator_cls const orderer);

void merge_buckets_parallel2(const UINT_t cache_individual_len, 
    void * const src_dedup, const size_t step, 
    size_t* const state, const UINT_t partition,
    comparator_cls const is_equal, merger_cls merge_fn,
    const unsigned int np, const unsigned int rank);

void pairwise_deduplicate_parallel(const UINT_t cache_individual_len, 
    void * const src_array, const size_t step, 
    size_t* const state, const UINT_t partition, 
    comparator_cls const is_equal, merger_cls merge_fn, const unsigned int np, 
    const unsigned int rank);

/**
* Merges two sequences of vertices that form the neighbourhood of a vertex
*
* @param u_in1 Sequence of vertices from existing state
*
* @param in2 Sequence of edges added
*
* @param u_in3 Output sequence of neighbourhood vertices
*
* @param in2_idx Index for location of processing in in2. 
*    Takes start of relevant sequences of edges and returns start of next relevant sequence of edges
*
* @param in1_length Length of neighbourhood already existing 
*
* @returns final length of neighbourhood of vertex after operations
*/
static inline UINT_t merge_list_dynamicTC(const UINT_t * u_in1, 
    const GRAPH_eTYPE * const in2, UINT_t * u_in3, 
    UINT_t * in2_idx, const UINT_t in1_length)
{
    const UINT_t * __restrict in1 = u_in1;
    const edge_t * __restrict const edgeList = in2->edgeList;
    UINT_t * __restrict in3 = u_in3;
    register UINT_t i = 0, j = *in2_idx, k = 0;
    const UINT_t l = edgeList[j].src;
    while(i < in1_length && j < in2->numEdges && edgeList[j].src == l)
    {
        if(in1[i] > edgeList[j].dst)
        {
            in3[k++] = edgeList[j++].dst;
        }
        else
        {
            in3[k++] = in1[i++];
        }
    }
    if(i < in1_length)
    {
        while(i < in1_length)
        {
            in3[k++] = in1[i++];
        }
    }
    else 
    {
        while(j < in2->numEdges && edgeList[j].src == l)
        {
            in3[k++] = edgeList[j++].dst;
        }
    }
    *in2_idx = j;
    return k;
}

void merge_GRAPH_TYPE__GRAPH_eTYPE_parallel(GRAPH_TYPE * exist, 
    const GRAPH_eTYPE * const update,  
    const vertexDegree_t * const idx_count, 
    const UINT_t number_vertices, const int np, const int rank);

/**
* Used for reserving space for inserting multiple elements
*
* @param array1D Vector to add an element to
*
* @param to_insert_length Number of elements to be inserted
*
* @param length Pointer to number of elements currently in array
*
* @param existing_size Pointer to number of spaces available for elements
*/
static inline void DBTL_vector_insert_test(UINT_t ** array1D, 
    const UINT_t to_insert_length, UINT_t * length, UINT_t * existing_size)
{
    register UINT_t i;
    i = (*existing_size == 0 ? 1 : *existing_size);
    while(*length + to_insert_length > i) {
        i *= 2;
    }
    if (i != *existing_size) {
        if(*existing_size == 0) {
            *array1D = (UINT_t*)malloc(i * sizeof(UINT_t));
        }
        else {
            *array1D = (UINT_t*)realloc(*array1D, i * sizeof(UINT_t));
        }
        assert_malloc(*array1D);
        *existing_size = i;
    }
}

static inline void DBTL_vector_withdraw(UINT_t ** array1D, 
    UINT_t * length, UINT_t * existing_size,
    UINT_t * withdraw_array, const UINT_t to_withdraw_length, 
    enum reorderDegree_t ngh_ordering)
{
    UINT_t k = 0;
    UINT_t * temp_ref = *array1D;
    if(ngh_ordering == REORDER_LOWEST_DEGREE_FIRST) {
        UINT_t i = 0;
        UINT_t j = 0;
        while (i < *length) {
            while (j < to_withdraw_length && withdraw_array[j] < temp_ref[i]) {
                j++;
            }
            if (j == to_withdraw_length || withdraw_array[j] != temp_ref[i]) {
                temp_ref[k++] = temp_ref[i];
            }
            i++;
        }
    }
    else {
        UINT_t i = 0;
        UINT_t j = 0;
        UINT_t l = to_withdraw_length - 1;
        while (i < *length) {
            while (j < to_withdraw_length && withdraw_array[l] > temp_ref[i]) {
                j++;
                l--;
            }
            if (j == to_withdraw_length || withdraw_array[l] != temp_ref[i]) {
                temp_ref[k++] = temp_ref[i];
            }
            i++;
        }
    }
    UINT_t i = *existing_size;
    while(i >= 2 && i > k) {
        i /= 2;
    }
    i *= 2;
    if(i == 0) {
        i += 1;
    }
    *array1D = (UINT_t*)realloc(*array1D, sizeof(UINT_t) * i);
    assert_malloc(*array1D);
    *existing_size = i;
    *length = k;
}

void demerge_GRAPH_TYPE__GRAPH_eTYPE_parallel(GRAPH_TYPE * exist, 
    const GRAPH_eTYPE * const update,  
    const vertexDegree_t * const idx_count, 
    const UINT_t number_vertices, const int np, const int rank);

static inline void remove_ngh_GRAPH_TYPE__GRAPH_eTYPE(const UINT_t * u_in1, 
    const GRAPH_eTYPE * const in2, UINT_t * u_in3, 
    UINT_t * in2_idx, const UINT_t in1_length)
{
    const UINT_t * __restrict in1 = u_in1;
    const edge_t * __restrict const edgeList = in2->edgeList;
    UINT_t * __restrict in3 = u_in3;
    register UINT_t i = 0, j = *in2_idx, k = 0;
    const UINT_t l = edgeList[j].src;
    while(i < in1_length) {
        while(j < in2->numEdges 
            && edgeList[j].src == l && edgeList[j].dst < in1[i]) {
            j++;
        }
        if (!(j < in2->numEdges 
            && edgeList[j].src == l) || edgeList[j].dst != in1[i]) {
            in3[k++] = in1[i];
        }
        i++;
    }
}