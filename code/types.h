#ifndef _TYPES_H
#define _TYPES_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <time.h>
#include <stdatomic.h>

#ifndef UINT_t
#define UINT_t uint64_t
#endif
#ifndef INT_t
#define INT_t int64_t
#endif

typedef uint64_t bool_t;
#define BOLT_CACHE_LEN (TC_CACHE_LINE_SIZE / sizeof(bool_t))
#ifdef _WIN32
#define BOLT_COUNTER(x) (__popcnt64(x))
#else
#define BOLT_COUNTER(x) (__builtin_popcountll(x))
#endif
#ifdef _WIN32
#define BOLT_EXTRACTOR(x,y) unsigned long PASTE(extract,__LINE__); \
    _BitScanForward64(&PASTE(extract,__LINE__),x); \
    y = PASTE(extract,__LINE__);
#else
#define BOLT_EXTRACTOR(x,y) y = __builtin_ctzll(x);
#endif
#define SET_BIT(i, array, type) const UINT_t PASTE(array_idx,__LINE__) = (i) / (BYTE_SZ(type)); \
    const UINT_t PASTE(byte_idx,__LINE__) = (i) % (BYTE_SZ(type)); \
    array[PASTE(array_idx,__LINE__)] = array[PASTE(array_idx,__LINE__)] | (((type)1) << PASTE(byte_idx,__LINE__));
#define UNSET_BIT(i, array, type) const UINT_t PASTE(array_idx,__LINE__) = (i) / (BYTE_SZ(type)); \
    const UINT_t PASTE(byte_idx,__LINE__) = (i) % (BYTE_SZ(type)); \
    array[PASTE(array_idx,__LINE__)] = array[PASTE(array_idx,__LINE__)] & (((type)0) << PASTE(byte_idx,__LINE__));
#define CHECK_BIT(result, i, array, type) const UINT_t PASTE(array_idx,__LINE__) = (i) / (BYTE_SZ(type)); \
    const UINT_t PASTE(byte_idx,__LINE__) = (i) % (BYTE_SZ(type)); \
    result = (array[PASTE(array_idx,__LINE__)] >> PASTE(byte_idx,__LINE__)) & ((type)1);
#define CLEAR_ALL_BIT(i, array, type) const UINT_t PASTE(array_idx,__LINE__) = i / (BYTE_SZ(type)); \
    array[PASTE(array_idx,__LINE__)] = ZERO;
#define BYTE_SZ(x) (sizeof(x) * 8)
#define MAX_BITS(x, size) (x / BYTE_SZ(size) + ((x) % BYTE_SZ(size) > 0 ? 1 : 0))

#ifndef HERO
#define HERO (UINT_t) -1
#endif

#ifndef ZERO
#define ZERO (UINT_t) 0
#endif

#define NAME_LEN 32

#define PARENS ()

#define EXPAND(...) EXPAND4(EXPAND4(EXPAND4(EXPAND4(__VA_ARGS__))))
#define EXPAND4(...) EXPAND3(EXPAND3(EXPAND3(EXPAND3(__VA_ARGS__))))
#define EXPAND3(...) EXPAND2(EXPAND2(EXPAND2(EXPAND2(__VA_ARGS__))))
#define EXPAND2(...) EXPAND1(EXPAND1(EXPAND1(EXPAND1(__VA_ARGS__))))
#define EXPAND1(...) __VA_ARGS__

#define FOR_EACH(macro, ...)                                    \
  __VA_OPT__(EXPAND(FOR_EACH_HELPER(macro, __VA_ARGS__)))
#define FOR_EACH_HELPER(macro, a1, ...)                         \
  macro(a1)                                                     \
  __VA_OPT__(FOR_EACH_AGAIN PARENS (macro, __VA_ARGS__))
#define FOR_EACH_AGAIN() FOR_EACH_HELPER

#define ACTUAL_PASTE(x,y) x ## y
#define PASTE(x,y) ACTUAL_PASTE(x,y)
#define DEFER(x) x
#define DO(...) __VA_ARGS__

typedef struct {
    UINT_t numVertices;
    UINT_t numEdges;
    UINT_t* rowPtr;
    UINT_t* colInd;
} GRAPH_TYPE;

typedef struct {
  UINT_t src;
  UINT_t dst;
} edge_t;

typedef struct {
  UINT_t numVertices;
  UINT_t numEdges;
  edge_t * edgeList;
} GRAPH_eTYPE;

typedef int (*comparator_cls)(const void*, const void*);
typedef void (*merger_cls)(void*, const void*);

enum {NS_P_S = 1000000000};

#ifdef _WIN32
#include <windows.h>
#define set_point(in1) QueryPerformanceCounter(&in1);
static double find_difference(LARGE_INTEGER in1, LARGE_INTEGER in2)
{
  LARGE_INTEGER PCFreq;
  QueryPerformanceFrequency(&PCFreq);
  return (double)((in2.QuadPart - in1.QuadPart)) / PCFreq.QuadPart;
}
#else
#define set_point(in1) clock_gettime(CLOCK_MONOTONIC, &in1);
static double find_difference(struct timespec in1, struct timespec in2)
{
  struct timespec td;                               
  td.tv_nsec = in2.tv_nsec - in1.tv_nsec;
  td.tv_sec = in2.tv_sec - in1.tv_sec;
  if(td.tv_sec > 0 && td.tv_nsec < 0)
  {                                                                
    td.tv_nsec += NS_P_S;                                          
    td.tv_sec--;                                                   
  }                                                                
  else if(td.tv_sec < 0 && td.tv_nsec > 0)
  {
    td.tv_nsec -= NS_P_S;
    td.tv_sec++;
  }
  return ((double)td.tv_sec) + (((double) td.tv_nsec) / NS_P_S);
}
#endif


#define TC_SENTINEL (UINT_t) (-1)
#define TC_CACHE_LINE_SIZE 64

#define SIZE_CACHE_LEN ((TC_CACHE_LINE_SIZE) / sizeof(size_t))

#define ODD(n) (((n)&1)==1)
#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

static void assert_malloc(const void *ptr) {
  if (ptr==NULL) {
    fprintf(stderr,"ERROR: Null pointer\n");
    exit(1);
  }
}

static void signal_free(void ** const ptr) {
  free(*ptr);
  *ptr = NULL;
}

static inline void* alloc_aligned(size_t alignment, size_t size) {
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#else
  return aligned_alloc(alignment, size);
#endif
}

static inline void free_aligned(void* ptr) {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

extern bool QUIET;
extern bool PRINT;

#endif