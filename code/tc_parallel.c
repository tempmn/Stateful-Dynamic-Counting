#include "types.h"
#include "graph.h"
#include <omp.h>


int intCompare(const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}


/* MapJIK is the triangle counting algorithm in this paper:

   "Exploring Optimizations on Shared-memory Platforms for Parallel Triangle Counting Algorithms." Ancy Sarah Tom, Narayanan Sundaram, Nesreen K. Ahmed, Shaden Smith, Stijn Eyerman, Midhunchandra Kodiyath, Ibrahim Hur, Fabrizio Petrini, and George Karypis. IEEE High Performance Extreme Computing Conference (HPEC), 2017.

   The source code is originally adapted from: https://github.com/KarypisLab/TriangleCounting

   The source code is adopted from: https://github.com/Bader-Research/triangle-counting
*/


#define GK_SBSIZE  64
#define GK_DBSIZE  8

/*************************************************************************/
/*! Reorders the vertices in the graph in inc degree order and returns
    the re-ordered graph in which the adjancency lists are sorted in 
    increasing order. In addition, a diagonal entry is added to each
    row to make the code that follows tighter.
*/
/*************************************************************************/
GRAPH_TYPE *ptc_Preprocess(const GRAPH_TYPE *originalGraph)
{
  INT_t vi, nvtxs, nthreads, maxdegree=0, csrange=0;
#if 0
  ssize_t *xadj, *nxadj, *psums;
#else
  INT_t *xadj, *nxadj, *psums;
#endif
  INT_t *adjncy, *nadjncy, *perm=NULL, *iperm=NULL, *chunkptr=NULL;
  INT_t *gcounts;
  GRAPH_TYPE *graph;

#pragma omp parallel
  {
    #pragma omp single
    nthreads = omp_get_num_threads();
  }

  nvtxs  = (INT_t)originalGraph->numVertices;
#if 0
  xadj   = (ssize_t *)originalGraph->rowPtr;
#else
  xadj   = (INT_t *)originalGraph->rowPtr;
#endif
  adjncy = (INT_t *)originalGraph->colInd;

  graph = (GRAPH_TYPE *)malloc(sizeof(GRAPH_TYPE));
  assert_malloc(graph);
  graph->numVertices = nvtxs;
  graph->numEdges = /* nvtxs+xadj[nvtxs] */ xadj[nvtxs];
  /* graph->rowPtr   = nxadj = (ssize_t *)malloc((nvtxs+1) * sizeof(ssize_t)); */
  nxadj = (INT_t *)malloc((nvtxs+1) * sizeof(INT_t));
  assert_malloc(nxadj);
  graph->rowPtr = (UINT_t *)nxadj;
  nadjncy = (INT_t *)malloc((nvtxs+xadj[nvtxs]) * sizeof(INT_t));
  assert_malloc(nadjncy);
  graph->colInd = (UINT_t *)nadjncy;
  
  perm  = (INT_t *)malloc(nvtxs * sizeof(INT_t));
  assert_malloc(perm);
  iperm = (INT_t *)malloc(nvtxs * sizeof(INT_t));
  assert_malloc(iperm);

  /* Determine maxdegree/csrange */
  #pragma omp parallel for schedule(static,4096) default(none) \
     shared(nvtxs, xadj) \
     reduction(max: maxdegree)
  for (vi=0; vi<nvtxs; vi++) 
    maxdegree = max(maxdegree, (INT_t)(xadj[vi+1]-xadj[vi]));

  csrange = maxdegree+1; 
  csrange = 16*((csrange+15)/16); /* get the per thread arrays to be alligned 
                                     at the start of the cache line */

  gcounts = (INT_t *)malloc((nthreads*csrange) * sizeof(INT_t));
  assert_malloc(gcounts);
#if 0
  psums   = (ssize_t *)malloc(nthreads * sizeof(ssize_t));
#else
  psums   = (INT_t *)malloc(nthreads * sizeof(INT_t));
#endif
  assert_malloc(psums);

  #pragma omp parallel default(none) \
    shared(nvtxs, nthreads, maxdegree, csrange, xadj, adjncy, nxadj, nadjncy, perm, iperm, gcounts, psums, chunkptr) 
  {
    INT_t vi, vistart, viend, vj, nedges, nchunks;
    INT_t ti, di, ci, dstart, dend;
    INT_t *counts, *buffer;
#if 0
    ssize_t ej, ejend, psum, chunksize;
#else
    INT_t ej, ejend, psum, chunksize;
#endif
    int mytid = omp_get_thread_num();

    vistart = mytid*((nvtxs+nthreads-1)/nthreads);
    viend   = min(nvtxs, (mytid+1)*((nvtxs+nthreads-1)/nthreads));

    dstart = mytid*((csrange+nthreads-1)/nthreads);
    dend   = min(csrange, (mytid+1)*((csrange+nthreads-1)/nthreads));

    /* Compute the local counts */
    counts = gcounts + mytid*csrange;
    for (int i = 0 ; i < csrange ; i++)
      counts[i] = 0;

    for (vi=vistart; vi<viend; vi++) 
      counts[xadj[vi+1]-xadj[vi]]++;
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
    for (di=dend-1; di>=dstart; di--) { 
      counts = gcounts + (nthreads-1)*csrange;
      for (ti=nthreads-1; ti>=0; ti--) {
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
    for (vi=vistart; vi<viend; vi++) {
      perm[vi] = counts[xadj[vi+1]-xadj[vi]]++;
      nxadj[perm[vi]] = xadj[vi+1]-xadj[vi]+1; /* the +1 is for the diagonal */
      iperm[perm[vi]] = vi;
    }
    #pragma omp barrier

    #pragma omp barrier
    /* compute the local sums and their prefix sums */
    for (psum=0, vi=vistart; vi<viend; vi++)
      psum += nxadj[vi];
    psums[mytid] = psum;
    #pragma omp barrier

    #pragma omp single
    for (ti=1; ti<nthreads; ti++)
      psums[ti] += psums[ti-1];
    #pragma omp barrier

    /* Compute the actual prefix sums of the nxadj[] array assigned to each thread.
       This is done from right to left to get it into the desired exclusive
       prefix sum stage. */
    psum = psums[mytid];
    if (mytid == nthreads-1)
      nxadj[nvtxs] = psum;
    for (vi=viend-1; vi>=vistart; vi--) { 
      psum -= nxadj[vi];
      nxadj[vi] = psum;
    }
    #pragma omp barrier
      
    /* Compute the chunk-based partitioning of the work for the reordered/sorted graph */
    chunksize = 1+psums[nthreads-1]/(100*nthreads);
    for (nchunks=0, psum=0, vi=vistart; vi<viend; vi++) {
      if ((psum += nxadj[vi+1]-nxadj[vi]) >= chunksize) {
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
    chunkptr = (INT_t *)malloc((psums[nthreads-1]+1) * sizeof(INT_t));
    assert_malloc(chunkptr);
    #pragma omp barrier

    nchunks = psums[mytid];
    chunkptr[nchunks] = viend;
    for (psum=0, vi=viend-1; vi>=vistart; vi--) {
      if ((psum += nxadj[vi+1]-nxadj[vi]) >= chunksize) {
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
    #pragma omp for schedule(dynamic, 1) nowait
    for (ci=nchunks-1; ci>=0; ci--) {
      for (vi=chunkptr[ci]; vi<chunkptr[ci+1]; vi++) {
        vj = iperm[vi];
        buffer = nadjncy+nxadj[vi];
        for (nedges=0, ej=xadj[vj], ejend=xadj[vj+1]; ej<ejend; ej++, nedges++)  {
          buffer[nedges] = perm[adjncy[ej]];
	  
	}
        buffer[nedges++] = vi; /* put the diagonal */

        if (nedges > 1)
	  qsort(buffer, nedges, sizeof(INT_t), intCompare);
      }
    }

  }

  free(perm);
  free(iperm);
  free(gcounts);
  free(psums);
  free(chunkptr);

  return graph;
}


/*************************************************************************/
/*! The hash-map-based triangle-counting routine that uses the JIK
    triangle enumeration scheme.

    This version implements the following:
     - It does not store location information in L
     - Adds diagonal entries in U to make loops tighter
     - Reverts the order within U's adjancency lists to allow ++ traversal
*/
/*************************************************************************/
UINT_t tc_MapJIK_P(const GRAPH_TYPE *originalGraph)
{
  INT_t vi, vj, nvtxs, startv;
#if 0
  ssize_t ei, ej;
#else
  INT_t ei, ej;
#endif
  UINT_t ntriangles=0;
#if 0
  ssize_t *xadj, *uxadj;
#else
  INT_t *xadj, *uxadj;
#endif
  INT_t *adjncy;
  INT_t l2, maxhmsize=0;
  GRAPH_TYPE *graph;

  graph = ptc_Preprocess(originalGraph);

  nvtxs  = (INT_t)graph->numVertices;
#if 0
  xadj   = (ssize_t *)graph->rowPtr;
#else
  xadj   = (INT_t *)graph->rowPtr;
#endif
  adjncy = (INT_t *)graph->colInd;

#if 0
  uxadj = (ssize_t *)malloc(nvtxs * sizeof(ssize_t)); /* the locations of the upper triangular part */
#else
  uxadj = (INT_t *)malloc(nvtxs * sizeof(INT_t)); /* the locations of the upper triangular part */
#endif
  assert_malloc(uxadj);

  /* populate uxadj[] and determine the size of the hash-map */
  startv = nvtxs;
  #pragma omp parallel for schedule(dynamic,1024) \
     default(none) \
     shared(nvtxs, xadj, adjncy, uxadj) \
     private(vj, ei, ej) \
     reduction(max: maxhmsize) \
     reduction(min: startv)
  for (vi=nvtxs-1; vi>=0; vi--) {
    for (ei=xadj[vi+1]-1; adjncy[ei]>vi; ei--);
    uxadj[vi] = ei;
    maxhmsize = max(maxhmsize, (INT_t)(xadj[vi+1]-uxadj[vi]));
    // startv = (uxadj[vi] != xadj[vi] ? vi : startv);
    // Original code is WRONG. You should not read a reduction variable arbitrarily?
    if (uxadj[vi] != xadj[vi] && vi < startv)
      startv = vi;
    /* flip the order of Adj(vi)'s upper triangular adjacency list */
    for (ej=xadj[vi+1]-1; ei<ej; ei++, ej--) {
      vj = adjncy[ei];
      adjncy[ei] = adjncy[ej];
      adjncy[ej] = vj;
    }
  }

  /* convert the hash-map is converted into a format that is compatible with a 
     bitwise AND operation */
  for (l2=1; maxhmsize>(1<<l2); l2++);
  maxhmsize = (1<<(l2+4))-1;

#if 0
  printf("& compatible maxhmsize: %"PRId32", startv: %d\n", maxhmsize, startv);
#endif

#if 1
  if (nvtxs < maxhmsize) {
    fprintf(stderr,"ERROR: MapJIK does not work when nvtxs < maxhmsize\n");
    return 0;
  }
#endif

  #pragma omp parallel default(none) \
    shared(nvtxs, xadj, adjncy, uxadj, maxhmsize, startv) \
    reduction(+: ntriangles)
  {
    INT_t vi, vj, vk, vl, nlocal;
#if 0
    ssize_t ei, eiend, eistart, ej, ejend, ejstart;
#else
    INT_t ei, eiend, eistart, ej, ejend, ejstart;
#endif
    INT_t l, nc;
    INT_t l2=1, hmsize=(1<<(l2+4))-1, *hmap;
    int mytid = omp_get_thread_num();

    hmap = (INT_t *)calloc(maxhmsize+1, sizeof(INT_t));
    assert_malloc(hmap);

    /* Phase 1: Count triangles for vj < nvtxs-maxhmsize */
    #pragma omp for schedule(dynamic,GK_SBSIZE) nowait
    for (vj=startv; vj<nvtxs-maxhmsize; vj++) {
      if (xadj[vj+1]-uxadj[vj] == 1 || xadj[vj] == uxadj[vj])
        continue;
  
      /* adjust hmsize if needed */
      if (xadj[vj+1]-uxadj[vj] > (1<<l2)) {
        for (++l2; (xadj[vj+1]-uxadj[vj])>(1<<l2); l2++);
        hmsize = (1<<(l2+4))-1;
      }

      /* hash Adj(vj) */
      for (nc=0, ej=uxadj[vj], ejend=xadj[vj+1]-1; ej<ejend; ej++) {
        vk = adjncy[ej];
        for (l=(vk&hmsize); hmap[l]!=0; l=((l+1)&hmsize), nc++);
        hmap[l] = vk;
      }
  
      nlocal = 0;

      /* find intersections */
      if (nc > 0) { /* we had collisions */
        for (ej=xadj[vj], ejend=uxadj[vj]; ej<ejend; ej++) {
          vi = adjncy[ej];
          for (ei=uxadj[vi]; adjncy[ei]>vj; ei++) {
            vk = adjncy[ei];
            for (l=vk&hmsize; hmap[l]!=0 && hmap[l]!=vk; l=((l+1)&hmsize));
            if (hmap[l] == vk) 
              nlocal++;
          }
        }
  
        /* reset hash */
        for (ej=uxadj[vj], ejend=xadj[vj+1]-1; ej<ejend; ej++) {
          vk = adjncy[ej];
          for (l=(vk&hmsize); hmap[l]!=vk; l=((l+1)&hmsize));
          hmap[l] = 0;
        }
      }
      else { /* there were no collisons */
        for (ej=xadj[vj], ejend=uxadj[vj]; ej<ejend; ej++) {
          vi = adjncy[ej];
#ifdef TC_VECOPT 
          for (eiend=uxadj[vi]; adjncy[eiend]>vj; eiend++);
          for (ei=uxadj[vi]; ei<eiend; ei++) 
#else
          for (ei=uxadj[vi]; adjncy[ei]>vj; ei++) 
#endif
          {
            vk = adjncy[ei];
            nlocal += (hmap[vk&hmsize] == vk);
          }
        }
  
        /* reset hash */
        for (ej=uxadj[vj], ejend=xadj[vj+1]-1; ej<ejend; ej++) 
          hmap[adjncy[ej]&hmsize] = 0;
      }
      
      if (nlocal > 0)
        ntriangles += nlocal;
    }


    /* Phase 2: Count triangles for the last hmsize vertices, which can be done
                faster by using hmap as a direct map array. */
    hmap -= (nvtxs - maxhmsize);
    #pragma omp for schedule(dynamic,GK_DBSIZE) nowait
    for (vj=nvtxs-1; vj>=nvtxs-maxhmsize; vj--) {
      if (xadj[vj+1]-uxadj[vj] == 1 || xadj[vj] == uxadj[vj])
        continue;
  
      nlocal = 0;

      if (xadj[vj+1]-uxadj[vj] == nvtxs-vj) { /* complete row */
        /* find intersections */
        for (ej=xadj[vj], ejend=uxadj[vj]; ej<ejend; ej++) {
          vi = adjncy[ej];
          for (ei=uxadj[vi]; adjncy[ei]>vj; ei++);
          nlocal  += ei-uxadj[vi];
        }
      }
      else {
        /* hash Adj(vj) */
        for (ej=uxadj[vj], ejend=xadj[vj+1]-1; ej<ejend; ej++) 
          hmap[adjncy[ej]] = 1;
  
        /* find intersections */
        for (ej=xadj[vj], ejend=uxadj[vj]; ej<ejend; ej++) {
          vi = adjncy[ej];
#ifdef TC_VECOPT 
          for (eiend=uxadj[vi]; adjncy[eiend]>vj; eiend++);
          for (ei=uxadj[vi]; ei<eiend; ei++) 
#else
          for (ei=uxadj[vi]; adjncy[ei]>vj; ei++) 
#endif
            nlocal += hmap[adjncy[ei]];
        }
  
        /* reset hash */
        for (ej=uxadj[vj], ejend=xadj[vj+1]-1; ej<ejend; ej++) 
          hmap[adjncy[ej]] = 0;
      }

      if (nlocal > 0)
        ntriangles += nlocal;
    }
    hmap += (nvtxs - maxhmsize);

    free(hmap);
  }

  free_graph(graph);
  free(uxadj);

  return (UINT_t)ntriangles;
}