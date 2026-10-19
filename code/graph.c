#include "types.h"
#include <string.h>
#include "graph.h"


void allocate_graph(GRAPH_TYPE* graph) {
  graph->rowPtr = (UINT_t*)calloc((graph->numVertices + 1), sizeof(UINT_t));
  assert_malloc(graph->rowPtr);
  graph->colInd = (UINT_t*)calloc(graph->numEdges, sizeof(UINT_t));
  assert_malloc(graph->colInd);
}

/**
* Reallocates the graph's memory. Useful when you want to keep allocation costs in timing
*
* @param graph Modifies graph in place. New number of vertices and edges must be set
*/
void reallocate_graph(GRAPH_TYPE* graph) {
  graph->rowPtr = (UINT_t*)realloc(graph->rowPtr, (graph->numVertices + 1) *
    sizeof(UINT_t));
  assert_malloc(graph->rowPtr);
  graph->colInd = (UINT_t*)realloc(graph->colInd, graph->numEdges * 
    sizeof(UINT_t));
  assert_malloc(graph->colInd);
}

void copy_graph(const GRAPH_TYPE *srcGraph, GRAPH_TYPE *dstGraph) {
  dstGraph->numVertices = srcGraph->numVertices;
  dstGraph->numEdges = srcGraph->numEdges;
  memcpy(dstGraph->rowPtr, srcGraph->rowPtr, (srcGraph->numVertices + 1) * sizeof(UINT_t));
  memcpy(dstGraph->colInd, srcGraph->colInd, srcGraph->numEdges * sizeof(UINT_t));
}

static int compareInt_t(const void *a, const void *b) {
    UINT_t arg1 = *(const UINT_t *)a;
    UINT_t arg2 = *(const UINT_t *)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

void convert_edges_to_graph(const edge_t* edges, GRAPH_TYPE* graph) {
  const UINT_t n = graph->numVertices;
  const UINT_t m = graph->numEdges;
  UINT_t* Ap = graph->rowPtr;
  UINT_t* Ai = graph->colInd;

  for (UINT_t i = 0 ; i < n + 1 ; i++)
    Ap[i] = 0;

  // Count the number of edges incident to each vertex
  for (UINT_t i = 0; i < m; i++) {
    UINT_t vertex = edges[i].src;
    Ap[vertex + 1]++;
  }

  // Compute the prefix sum of the rowPtr array
  for (UINT_t i = 1; i <= n; i++) {
    Ap[i] += Ap[i - 1];
  }

  // Populate the col_idx array with the destination vertices
  UINT_t *current_row = (UINT_t *)calloc(n, sizeof(UINT_t));
  assert_malloc(current_row);
  for (UINT_t i = 0; i < m; i++) {
    UINT_t src_vertex = edges[i].src;
    UINT_t dst_vertex = edges[i].dst;
    UINT_t index = Ap[src_vertex] + current_row[src_vertex];
    Ai[index] = dst_vertex;
    current_row[src_vertex]++;
  }

  // Sort the column indices within each row
  for (UINT_t i = 0; i < graph->numVertices; i++) {
    UINT_t s = Ap[i];
    UINT_t e = Ap[i + 1];
    UINT_t size = e - s;
    UINT_t *row_indices = &Ai[s];
    qsort(row_indices, size, sizeof(UINT_t), compareInt_t);
  }

  free(current_row);

}

static inline UINT_t convert_edge_to_graph_parallel_boundary(const edge_t* edges, 
  const UINT_t edges_len, const UINT_t rough_boundary) {
  register UINT_t i;
  if(rough_boundary > 0) {
    for(i = rough_boundary - 1; i > 0 && 
      edges[rough_boundary].src == edges[i].src; i--);
  }
  else {
    i = 0;
  }
  const UINT_t lesser_boundary = i == 0 ? 0 : i + 1;
  if(rough_boundary < edges_len) {
    for(i = rough_boundary + 1; i < edges_len &&
      edges[rough_boundary].src == edges[i].src; ++i);
  }
  else {
    i = edges_len;
  }
  const UINT_t greater_boundary = i;
  return (greater_boundary - rough_boundary > rough_boundary - lesser_boundary) ?
    lesser_boundary : greater_boundary;
}

void convert_edge_to_graph_parallel(const edge_t* edges, GRAPH_TYPE* graph,
  const unsigned int np, const unsigned int rank) {
#define UINT_CACHE_LEN (TC_CACHE_LINE_SIZE / sizeof(UINT_t))
  const UINT_t n = graph->numVertices;
  const UINT_t m = graph->numEdges;
  UINT_t * Ap = graph->rowPtr;
  UINT_t* Ai = graph->colInd;

  UINT_t * prefix_help;

  #pragma omp for schedule(static)
  for(UINT_t i = 0; i < n + 1; ++i) {
    Ap[i] = 0;
  }

  #pragma omp single copyprivate(prefix_help)
  {
    prefix_help = (UINT_t*)malloc((np + 1) 
      * sizeof(UINT_t));
    assert_malloc(prefix_help);
  }

  const UINT_t rough_edge_start = (rank * m) / np;
  const UINT_t rough_edge_end = ((rank + 1) * m) / np;

  UINT_t i;
  UINT_t* private_Ap;
  if(rank > 0) {
    i = convert_edge_to_graph_parallel_boundary(edges, m, 
      rough_edge_start);
  }
  else {
    i = 0;
  }
  const UINT_t edge_start = i;
  if(rank < np - 1) {
    i = convert_edge_to_graph_parallel_boundary(edges, m, 
      rough_edge_end);
  }
  else {
    i = m;
  }
  const UINT_t edge_end = i;
  
  const UINT_t edge_offset = edges[edge_start].src;
  const UINT_t edge_offset_end = (edge_end == m) ? edges[edge_end - 1].src :
    edges[edge_end].src;
  const UINT_t vlength_under_edgeb = (edge_offset == edge_offset_end && 
    edge_start != edge_end) ? 1 : edge_offset_end - edge_offset + (edge_end == m ? 1 : 0);
  prefix_help[rank] = vlength_under_edgeb;
  
  #pragma omp barrier
  #pragma omp single copyprivate(private_Ap,i)
  {
    i = 0;
    for(UINT_t j = 0; j < np; ++j) {
      i = max(prefix_help[j], i);
    }
    const UINT_t rounded_to_cache = (1 + (i - 1) / UINT_CACHE_LEN) *
      UINT_CACHE_LEN;
    private_Ap = (UINT_t*)alloc_aligned(TC_CACHE_LINE_SIZE, 
      np * rounded_to_cache * sizeof(UINT_t));
    assert_malloc(private_Ap);
    i = rounded_to_cache;
  }

  const UINT_t individual_cache_len = i;
  UINT_t* const my_Ap = private_Ap + rank * individual_cache_len;

  for(i = 0; i < vlength_under_edgeb; ++i) {
    my_Ap[i] = 0;
  }

  for(i = edge_start; i < edge_end; ++i) {
    my_Ap[edges[i].src - edge_offset]++;
  }
  
  memcpy(Ap + edge_offset + 1, my_Ap, 
    vlength_under_edgeb * sizeof(UINT_t));

  #pragma omp barrier

  const UINT_t vtx_start = (rank * (n + 1)) / np;
  const UINT_t vtx_end = ((rank + 1) * (n + 1)) / np;

  if(vtx_end - vtx_start > 1) {
    for(UINT_t i = vtx_start; i < vtx_end - 1; ++i) {
      Ap[i + 1] += Ap[i];
    }
  }

  prefix_help[rank + 1] = Ap[vtx_end - 1];

  #pragma omp barrier
  #pragma omp single
  {
    prefix_help[0] = 0;
    for(UINT_t i = 0; i < np; ++i) {
      prefix_help[i + 1] += prefix_help[i];
    }
  }

  for(UINT_t i = vtx_start; i < vtx_end; ++i) {
    Ap[i] += prefix_help[rank];
  }

  #pragma omp barrier

  #pragma omp for schedule(dynamic,UINT_CACHE_LEN)
  for(UINT_t i = 0; i < n; ++i) {
    const UINT_t my_length = Ap[i + 1] - Ap[i];
    for(UINT_t j = 0; j < my_length; ++j) {
      const UINT_t common_offset = Ap[i] + j;
      Ai[common_offset] = edges[common_offset].dst;
    }
    qsort(Ap[i] + Ai, my_length, 
      sizeof(UINT_t), compareInt_t);
  }
  #pragma omp sections nowait
  {
    #pragma omp section
    {
      free(prefix_help);
    }
    #pragma omp section
    {
      free_aligned(private_Ap);
    }
  }
#undef UINT_CACHE_LEN
}

void free_graph(GRAPH_TYPE* graph) {
    free(graph->rowPtr);
    free(graph->colInd);
    free(graph);
}

void allocate_graphe(GRAPH_eTYPE* graph) {
  graph->edgeList = (edge_t*)calloc(graph->numEdges, sizeof(edge_t));
  assert_malloc(graph->edgeList);
}

void copy_graphe(const GRAPH_eTYPE *srcGraph, GRAPH_eTYPE *dstGraph) {
  dstGraph->numVertices = srcGraph->numVertices;
  dstGraph->numEdges = srcGraph->numEdges;
  memcpy(dstGraph->edgeList, srcGraph->edgeList, srcGraph->numEdges * sizeof(edge_t));
}

void free_graphe(GRAPH_eTYPE* graph) {
  free(graph->edgeList);
  free(graph);
}