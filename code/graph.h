#ifndef _GRAPH_H
#define _GRAPH_H

#include "types.h"

typedef struct {
  UINT_t degree;
  UINT_t index;
} vertexDegree_t;

void allocate_graph(GRAPH_TYPE*);
void reallocate_graph(GRAPH_TYPE* graph);
void copy_graph(const GRAPH_TYPE *, GRAPH_TYPE *);
void convert_edges_to_graph(const edge_t*, GRAPH_TYPE*);
void convert_edge_to_graph_parallel(const edge_t* edges, GRAPH_TYPE* graph, 
  const unsigned int np, const unsigned int rank);
void free_graph(GRAPH_TYPE*);

void allocate_graphe(GRAPH_eTYPE* );
void copy_graphe(const GRAPH_eTYPE *, GRAPH_eTYPE *);
void free_graphe(GRAPH_eTYPE*);

enum reorderDegree_t { REORDER_HIGHEST_DEGREE_FIRST = 0, REORDER_LOWEST_DEGREE_FIRST };

#endif