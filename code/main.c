#include "tc_dynamic_parallel.h"
#include "types.h"
#include "graph.h"
#include "merge.h"
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#endif
#include "tc_dynamic_state.h"
#include "benchmarking.h"
#include "tc_parallel.h"
#include <omp.h>

#ifndef LOOP_CNT
#define LOOP_CNT       5
#endif
#ifndef LOOP_TRIAL
#define LOOP_TRIAL      1
#endif

bool QUIET  = false;

static FILE *outfile = NULL;
static char *INFILENAME = NULL;

void usage() {

  printf("Triangle Counting\n\n");
  printf("Usage:\n\n");
  printf(" -f <filename>   [Input Graph in Matrix Market format]\n");
  printf("Optional arguments:\n");
  printf(" -o <filename>   [Output File]\n");
  printf(" -Q              [Turn on Quiet mode]\n");
  printf(" -p #            [Parallel: Use # threads/cores]\n");
  printf(" -P              [Parallel: Use maximum number of cores]\n");
  printf(" -c #            [Run dynamic algorithms with constant inputs on given number of batches]\n");
  printf(" -s #            [Run dynamic algorithms with scaling inputs on given number of batches]\n");
  printf(" -i <#|.>        [Run dynamic algorithms with given percentage or number of insertions]\n");
  printf(" -w <#|.>        [Run dynamic algorithms with given percentage or number of withdrawals]\n");
  printf("# - Integer      . - Float\n");
  return;
}

void extended_usage() {
  usage();
  printf(" -l <filename>   [Run multiple parameters on algorithms. Place each set of commands on its own line]\n");
}

void select_usage_error(const bool read_direct) {
  if(read_direct) {
    extended_usage();
  }
  else {
    usage();
  }
  exit(8);
}

void manage_benchmark_memory(BenchmarkingParameters** bencher, int * bencher_len,
  int* bencher_allocated, const int new_len) {
  if(*bencher == NULL) {
    *bencher_len = 0;
    *bencher_allocated = new_len;
    *bencher = (BenchmarkingParameters*)malloc(new_len * 
      sizeof(BenchmarkingParameters));
    assert_malloc(*bencher);
  }
  else {
    const UINT_t new_bencher_len = *bencher_allocated + new_len;
    *bencher = (BenchmarkingParameters*)realloc(*bencher, 
      sizeof(BenchmarkingParameters) * new_bencher_len);
    assert_malloc(*bencher);
    *bencher_allocated = new_bencher_len;
  }
}

void manage_string_memory(char** in, const size_t new_len) {
  if(*in == NULL) {
    *in = (char*)malloc(new_len * sizeof(char));
    assert_malloc(*in);
  }
  else {
    *in = (char*)realloc(*in, new_len * sizeof(char));
    assert_malloc(*in);
  }
}

bool isint(char * in) {
  int temp;
  for(temp = 0; in[temp] != '\0' && in[temp] != '.'; ++temp);
  return in[temp] == '\0';
}

char* dupstr(const char* const sin) {
  size_t len = strlen(sin) + 1;
  char* toreturn = (char*)malloc(sizeof(char) * len);
  assert_malloc(toreturn);
  memcpy(toreturn, sin, len * sizeof(char));
  return toreturn;
}

int parseFlags(int argc, char **argv, const bool read_direct, 
  BenchmarkingParameters** out, int* out_len, int* allocated_len) {

  if (argc < 1) {
    select_usage_error(read_direct);
  }
  int exit_code = 0;
  
  int argcc;
  int *actual_argc;
  char *** actual_argv;
  
  if(read_direct) {
    argcc = 1;
    actual_argc = &argc;
    actual_argv = &argv;
    manage_benchmark_memory(out, out_len, 
      allocated_len, 1);
  }
  else {
    const int BUFFER_SIZE = 4;
    char buffer[BUFFER_SIZE];
    (*out_len)++;
    
    FILE* incmd = NULL;
    argcc = 0;
    incmd = fopen(argv[0], "r");
    if(incmd == NULL) {
      perror("Could not open file for reading parameters\n");
      exit_code |= 2;
      return exit_code;
    }
    for(;;) {
      size_t nchars = fread(buffer, sizeof(char), 
        BUFFER_SIZE, incmd);
      if(ferror(incmd)) {
        perror("Error while reading parameter file.\n");
        exit_code |= 1;
        fclose(incmd);
        return exit_code;
      }
      for(size_t i = 0; i < nchars; ++i) {
        if(buffer[i] == '\n') {
          argcc++;
        }
      }
      if(feof(incmd)) {
        break;
      }
    }
    actual_argc = (int*)malloc(argcc * sizeof(int));
    assert_malloc(actual_argc);
    actual_argv = (char***)malloc(argcc * sizeof(char**));
    assert_malloc(actual_argv);
    for(int i = 0; i < argcc; ++i) {
      actual_argc[i] = 0;
      actual_argv[i] = NULL;
    }
    manage_benchmark_memory(out, out_len, 
      allocated_len, argcc);
    
    rewind(incmd);
    size_t i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, p = 0, q = 0;
    for(;;) {
      size_t nchars = fread(buffer, sizeof(char), 
        BUFFER_SIZE, incmd);
      if(ferror(incmd)) {
        perror("Error while reading parameter file.\n");
        exit_code |= 1;
        fclose(incmd);
        return exit_code;
      }
      for(i = 0; i < nchars; i = j) { 
        for(l = 0; i < nchars && buffer[i] == ' '; ++i, ++l); 
        if((l > 0 && actual_argv[k][n] != NULL) || (i < nchars && buffer[i] == '\n')) { 
          n++; 
          p = 0; q = 0; 
        } 
        for(j = i; j < nchars && buffer[j] != '\n'; ++j);
        for(l = i, p = 0; l < j; l = m) {
          for(m = l; m < j && buffer[m] != ' '; ++m);
          if(m < j || (buffer[m - 1] != ' ')) {
            p++;
          } for(; m < j && buffer[m] == ' '; ++m);
        } actual_argc[k] += (p - q);
        if(actual_argv[k] == NULL) {
          actual_argv[k] = (char**)malloc(actual_argc[k] 
            * sizeof(char*));
          assert_malloc(actual_argv[k]);
        }
        else {
          actual_argv[k] = (char**)realloc(actual_argv[k], 
            actual_argc[k] * sizeof(char*));
          assert_malloc(actual_argv[k]);
        }
        for(m = n + q; m < actual_argc[k]; ++m) {
          actual_argv[k][m] = NULL;
        }
        for(l = i; l < j; l = m) {
          for(m = l; m < j && buffer[m] != ' '; ++m);
          const size_t old_size = (actual_argv[k][n] == NULL) ? 0 : 
            strlen(actual_argv[k][n]);
          const size_t new_size = 1 + old_size + m - l;
          manage_string_memory(actual_argv[k] + n, new_size);
          snprintf(actual_argv[k][n] + old_size, new_size, 
            "%.*s", (int)(m - l), buffer + l);
          if(m < j || (buffer[m - 1] == ' ')) {
            n++;
          }
          for(; m < j && buffer[m] == ' '; ++m);
        }
        if(j < nchars) {
          k++; j++;
          n = 0; p = 0; q = 0;
        }
        else {
          if(buffer[j - 1] != ' ') { 
            q = 1;
          } 
          else { 
            q = 0; 
          }
        }
      }
      if(feof(incmd)) {
        break;
      }
    }
    fclose(incmd);
  }
  for(int i = 0, update_in_switch = 0; i < argcc; ++i) {
    FILE* infile = NULL;
    int temp_argc = actual_argc[i];
    char** temp_argv = actual_argv[i];
    BenchmarkingParameters* temp_bench = *out + *out_len;
    reset_benchmarking_parameters(temp_bench);
    bool input_selected = false;
    while ((temp_argc > 1) && (temp_argv[1][0] == '-')) {

      switch (temp_argv[1][1]) {

        case 'f':
          if (temp_argc < 3 || !read_direct || input_selected)
            select_usage_error(read_direct);
          if (!QUIET)
            printf("Input Graph: %s\n", temp_argv[2]);
          infile = fopen(temp_argv[2], "r");
          if (infile == NULL) {
            select_usage_error(read_direct);
          }
          INFILENAME = temp_argv[2];
          input_selected = true;
          fclose(infile);
          temp_argv+=2;
          temp_argc-=2;
          break;

        case 'o':
          if (temp_argc < 3)
            select_usage_error(read_direct);
          if (!QUIET)
            printf("Output file: %s\n", temp_argv[2]);
          temp_bench->output_is_stdout = false;
          temp_bench->output_file = dupstr(temp_argv[2]);
          infile = fopen(temp_argv[2], "a");
          if (infile == NULL)
            select_usage_error(read_direct);
          fclose(infile);
          temp_argv+=2;
          temp_argc-=2;
          break;

        case 'Q':
          if(!read_direct)
            select_usage_error(read_direct);
          QUIET = true;
          temp_argv++;
          temp_argc--;
          break;
        
        case 'l':
          {
            const int saved_location = *out_len;
            exit_code |= parseFlags(actual_argc[i], temp_argv + 2,  
              false, out, out_len, allocated_len);
            temp_bench = *out + saved_location;
            temp_bench->algos = META;
            update_in_switch = 1;
            temp_argv+=2;
            temp_argc-=2;
            break;
          }
        
        case 'P':
          temp_bench->parallel_count = omp_get_max_threads();
          temp_argv++;
          temp_argc--;
          break;
          
        case 'p':
          if (argc < 3)
            select_usage_error(read_direct);
          temp_bench->parallel_count = atoi(temp_argv[2]);
          temp_argv+=2;
          temp_argc-=2;
          break;
        
        case 'c':
          if(argc < 3)
            select_usage_error(read_direct);
          temp_bench->algos = DYNAMIC_CONSTANT_CPU;
          temp_bench->batch_count = atoi(temp_argv[2]);
          temp_argv+=2;
          temp_argc-=2;
          break;
        
        case 's':
          if(argc < 3)
            select_usage_error(read_direct);
          temp_bench->algos = DYNAMIC_REALTIME_CPU;
          temp_bench->batch_count = atoi(temp_argv[2]);
          temp_argv+=2;
          temp_argc-=2;
          break;
        
        case 'i':
          if(argc < 3)
            select_usage_error(read_direct);
          temp_bench->insert_edges = true;
          if(isint(temp_argv[2])) {
            temp_bench->add_count = atoi(temp_argv[2]);
            temp_bench->add_is_float = false;
          }
          else {
            temp_bench->add_ratio = atof(temp_argv[2]);
            temp_bench->add_is_float = true;
          }
          temp_argv+=2;
          temp_argc-=2;
          break;
        
        case 'w':
          if(argc < 3)
            select_usage_error(read_direct);
          temp_bench->withdraw_edges = true;
          if(isint(temp_argv[2])) {
            temp_bench->delete_count = atoi(temp_argv[2]);
            temp_bench->delete_is_float = false;
          }
          else {
            temp_bench->delete_ratio = atof(temp_argv[2]);
            temp_bench->delete_is_float = true;
          }
          temp_argv+=2;
          temp_argc-=2;
          break;
      
        default:
          fprintf(stderr,"Wrong Argument: %s\n", argv[1]);
          select_usage_error(read_direct);
          break;
      }

    }
    if (read_direct && !input_selected) {
      select_usage_error(read_direct);
    }
    if ((temp_bench->algos == DYNAMIC_CONSTANT_CPU ||
      temp_bench->algos == DYNAMIC_REALTIME_CPU) && 
      (!(temp_bench->insert_edges || temp_bench->withdraw_edges))) {
      select_usage_error(read_direct);
    }
    if(update_in_switch == 1) {
      update_in_switch = 0;
    }
    else {
      ++(*out_len);
    }
  }
  if(!read_direct) {
    for(size_t i = 0; i < argcc; ++i) {
      for(size_t j = 0; j < actual_argc[i]; ++j) {
        free(actual_argv[i][j]);
      }
      free(actual_argv[i]);
    }
    free(actual_argv);
    free(actual_argc);
  }
  
  return exit_code;
}

static UINT_t correctTriangleCount;
/* Check the correctness of the triangle count.
   Return 1 if worked, 0 if failed */
bool check_triangleCount(const GRAPH_TYPE *graph, const UINT_t numTriangles) {
  return (numTriangles==correctTriangleCount);
}

typedef UINT_t (*alt_iow)(ALT_DYNAMIC_STATE*, GRAPH_eTYPE*);

typedef UINT_t (*dbtl_iow)(DBTL_GRAPH*, GRAPH_eTYPE*);

typedef union static_dtc {
  UINT_t (* iow)(GRAPH_TYPE*, GRAPH_eTYPE*);
} static_dtc;

typedef union alt_dtc {
  alt_iow * iow;
} alt_dtc;

typedef union dbtl_dtc {
  dbtl_iow* iow;
} dbtl_dtc;

typedef union any_batch{
  GRAPH_eTYPE* e;
} any_batch;

typedef union any_dynamicTC {
  static_dtc* STTIC;
  dbtl_dtc* DBTL;
  alt_dtc* ALT;
} any_dynamicTC;

typedef union any_dynamicstate {
  GRAPH_TYPE* STTIC;
  ALT_DYNAMIC_STATE* ALT;
  DBTL_GRAPH* DBTL;
} any_dynamicstate;

typedef enum which_dynamicTC {
  STTIC, ALT, DBTL
} which_dynamicTC;

typedef struct any_dynamicTC_len {
  any_dynamicTC* fn;
  which_dynamicTC fn_type;
  unsigned int fn_len;
  const char (*name)[NAME_LEN];
} any_dynamicTC_len;

void benchmark_TCD_performance_capture_reset_tasks(any_dynamicstate* a_state,
  which_dynamicTC fn_type, const GRAPH_TYPE * const statin,
  const UINT_t hash_len_vtx) {
  switch(fn_type) {
    case STTIC:
      a_state->STTIC->numVertices = statin->numVertices;
      a_state->STTIC->numEdges = statin->numEdges;
      reallocate_graph(a_state->STTIC);
      copy_graph(statin, a_state->STTIC);
      break;
    case ALT:
      reset_alt_state(a_state->ALT, statin);
      break;
    case DBTL:
      reset_DBTL_graph(a_state->DBTL, statin);
      break;
  }
}

double benchmark_TCD_performance_capture_run(any_dynamicTC* a_fn,
  any_dynamicstate* a_state, const which_dynamicTC fn_type, any_batch* a_batch,
  const unsigned int idx, const unsigned int parallel_count, UINT_t * const count)
{
#ifdef _WIN32
    LARGE_INTEGER start;
    LARGE_INTEGER end;
#else
    struct timespec start;
    struct timespec end;
#endif
  omp_set_num_threads(parallel_count > 0 ? parallel_count : 1);
  UINT_t tc;
    switch(fn_type) {
    case STTIC:
    set_point(start);
    tc = a_fn->STTIC->iow(a_state->STTIC, a_batch->e);
    set_point(end);
    break;
    case ALT:
    set_point(start);
    tc = a_fn->ALT->iow[idx](a_state->ALT, a_batch->e);
    set_point(end);
    break;
    case DBTL:
    set_point(start);
    tc = a_fn->DBTL->iow[idx](a_state->DBTL, a_batch->e);
    set_point(end);
    break;
  }
  *count = tc;
  return find_difference(start, end);
}

/**
* Copies batch according to type
*
* @param a Array of batches. One element is to be copied
*
* @param b Batch to be copied to
*
* @param idx Index of the element in the array pointed to by a
*
* @note b must have allocated memory earlier
*/
void copy_correct_batch(const any_batch* a, any_batch* b, size_t idx) {
  copy_graphe(a->e + idx, b->e);
}

void benchmark_TCD_performance_capture(
  any_dynamicTC* a_fn, any_dynamicstate* a_state, which_dynamicTC fn_type, 
  const GRAPH_eTYPE * const originalGraph, any_batch* batc, any_batch* batch,
  const GRAPH_TYPE * const verifying_states, const GRAPH_TYPE * const statin, 
  const unsigned int len, const unsigned int stat_offset, const UINT_t hash_len_vtx,
  FILE * const out, const char (*const name)[NAME_LEN],
  const BenchmarkingParameters* const run_params, double * batch_times,
  double * total_times, UINT_t * batch_counts, const UINT_t start_count) {

  const unsigned int BATCH_COUNT = run_params->batch_count;
  const bool is_detype = run_params->insert_edges && run_params->withdraw_edges;
  UINT_t count, overall_count;
  for(unsigned int j = 0; j < len; ++j) {
    const unsigned int stat_jidx = j + stat_offset;
    // const unsigned int stat_len = len + 1;
    fprintf(out, "Running %s:\n", name[j]);
    for(unsigned int k = LOOP_TRIAL; k > 0; --k) {
      fprintf(out, "TRIAL %d:[", LOOP_TRIAL - k + 1);
      for(unsigned int i = 0; i < BATCH_COUNT; ++i) {
        copy_correct_batch(batc, batch, i);
        benchmark_TCD_performance_capture_run(a_fn, a_state, fn_type, 
          batch, j, run_params->parallel_count, &count);
#ifdef VERIFY_DYNAMIC_STATES
        if(k > 1) {
          switch(fn_type) {
            case STTIC:
              break;
            case ALT:
              check_alt_consistency(verifying_states + i, 
                a_state->ALT);
              break;
            case BASIC:
              check_consistency(verifying_states + i, 
                a_state->BASIC);
              break;
            case DBTL:
              check_DBTL_consistency(verifying_states + i, 
                a_state->DBTL);
              break;
          }
        }
#endif
        switch(run_params->algos)
        {
          case DYNAMIC_CONSTANT_CPU:
            benchmark_TCD_performance_capture_reset_tasks(a_state, 
              fn_type, statin, hash_len_vtx);
            break;
          default:
            break;
        }
        print_trial_progress();
      }
      switch(run_params->algos)
      {
        case DYNAMIC_CONSTANT_CPU:
          break;
        default:
          benchmark_TCD_performance_capture_reset_tasks(a_state, 
            fn_type, statin, hash_len_vtx);
          break;
      }
      fprintf(out, "]\n");
    }
    overall_count = 0;
    for(unsigned int k = 0; k < LOOP_CNT; ++k) {
      fprintf(out, "REAL %d:[", k + 1);
      for(unsigned int i = 0; i < BATCH_COUNT; ++i) {
        if(fn_type == STTIC)
          overall_count = 0;
        copy_correct_batch(batc, batch, i);
        const double run_time = benchmark_TCD_performance_capture_run(a_fn, 
          a_state, fn_type, batch, j, 
          run_params->parallel_count, &count);
        if(run_params->insert_edges || fn_type == STTIC) {
            overall_count += count;
        }
        else {
            overall_count -= count;
        }
        batch_counts[stat_jidx * BATCH_COUNT + i] = start_count + overall_count;
        batch_times[stat_jidx * BATCH_COUNT + i] += run_time;
        total_times[stat_jidx] += run_time;
        check_count_consistency(batch_counts[i], 
          batch_counts[stat_jidx * BATCH_COUNT + i], 
          __FILE__, __LINE__, name[j]);
        switch(run_params->algos)
        {
          case DYNAMIC_CONSTANT_CPU:
            benchmark_TCD_performance_capture_reset_tasks(a_state, 
              fn_type, statin, hash_len_vtx);
            overall_count = 0;
            break;
          default:
            break;
        }
        print_reals_progress();
      }
      switch(run_params->algos)
      {
        case DYNAMIC_CONSTANT_CPU:
          break;
        default:
          benchmark_TCD_performance_capture_reset_tasks(a_state, 
            fn_type, statin, hash_len_vtx);
          overall_count = 0;
          break;
      }
      fprintf(out, "]\n");
    }
  }
}

void print_batch_number_of_edges(GRAPH_eTYPE* batch) {
  printf("\tNumber of edges:\t%zu\n", batch->numEdges);
}

/**
* Benchmark tests the sensitivity of dynamic algorithm to different inputs (batches).
*
* @note Use option -i or -w. Not both. 
*
* @param f A function that takes a graph and list of edges. Static algorithm.
* 
* @param g An array of dynamic algorithms that take ALT_DYNAMIC_STATE and list of edges.
*
* @param h An array of dynamic algorithms that take BASIC_DYNAMIC_STATE and list of edges.
*
* @param originalGraph Original edge graph, not modified during operation
*
* @param statin Original graph, not modified during operation
*
* @param dbtl_len Number of algorithms taking DBTL_GRAPH
*
* @param alt_len Number of algorithms taking ALT_DYNAMIC_STATE
*
* @param basic_len Number of algorithms taking BASIC_DYNAMIC_STATE
*
* @param name Constant array having names corresponding to algorithms in @param g and @param h
*
* @param num_threads Number of threads to use in case of parallel algorithms
*/
void benchmark_TCD(any_dynamicTC_len* all_fns, unsigned int all_fns_len,
  unsigned int static_idx, const GRAPH_eTYPE * const originalGraph, 
  const GRAPH_TYPE * const statin, const BenchmarkingParameters* const run_params) {
#ifdef _WIN32
  LARGE_INTEGER start;
  LARGE_INTEGER end;
#else
  struct timespec start;
  struct timespec end;
#endif
  any_batch batch, batc;
  GRAPH_TYPE* static_state;
  GRAPH_TYPE* restrict verifying_states;
  ALT_DYNAMIC_STATE* alt_superstates;
  DBTL_GRAPH* dbtl_superstates;
  GRAPH_eTYPE update_container;
  double * batch_times, * total_times;
  UINT_t * batch_counts, * edge_counts, * edge_counts2;
  double ratio;
  UINT_t random_state = 1;
  unsigned int idx, jdx;
  for(idx = 0, jdx = 0; idx < all_fns_len; ++idx) {
    jdx += all_fns[idx].fn_len;
  }
  const unsigned int len = jdx;
  const unsigned int stat_len = len + 1;
  initialize_TCD_benchmarking(&(batc.e), &(batch.e), statin, 
    originalGraph, &static_state, run_params);
  printf("  Inserting edges? %s\n", run_params->insert_edges ? "true" : "false");
  printf("  Withdrawing edges? %s\n", run_params->withdraw_edges ? "true" : "false");
  printf("  Number of batches: %d\n\n", run_params->batch_count);
  for(unsigned int i = 0; i < run_params->batch_count; ++i) {
      print_batch_number_of_edges(&(batc.e[i]));
  }
  printf("\n");
  
  initialize_TCD_stat_records(&batch_times, &total_times, 
    &batch_counts, &edge_counts, &edge_counts2, 
    run_params, len);
  const UINT_t start_count = correctTriangleCount;
  const unsigned int BATCH_COUNT = run_params->batch_count;
  FILE* const meta_results_out = stdout;
#ifdef PARALLEL
  omp_set_num_threads(omp_get_num_procs() / 2);
#endif
#ifdef VERIFY_DYNAMIC_STATES
  verifying_states = (GRAPH_TYPE*)malloc(BATCH_COUNT
    * sizeof(GRAPH_TYPE));
  assert_malloc(verifying_states);
  fprintf(meta_results_out, "Generating edges and states...\n");
#else
  fprintf(meta_results_out, "Generating edges...\n");
#endif
  set_point(start);
  for(unsigned int i = 0; i < BATCH_COUNT; ++i) {
    if(run_params->insert_edges) {
      random_state = generate_random_edges(static_state, 
      batc.e + i, i == 0, random_state);
    }
    else if(run_params->withdraw_edges) {
      random_state = choose_random_edges(static_state, 
       batc.e + i, i == 0, random_state);
    }
    copy_graphe(batc.e + i, batch.e);
    edge_counts[i] = batc.e[i].numEdges;
    // printf("ID of batch: %d\n", i);
    #define VDGX_CACHE_LEN (TC_CACHE_LINE_SIZE / sizeof(vertexDegree_t))
    #pragma omp parallel default(shared)
    {
      vertexDegree_t* proxy_idx_count, *idx_count;
      UINT_t number_vertices;
      edge_t* edge_list_dst;
      UINT_t* ptr_to_arrays;
      const unsigned int np = omp_get_num_threads();
      const unsigned int rank = omp_get_thread_num();
      find_dlt_base_alt2(batch.e, &edge_list_dst, 
        static_state->numVertices, &proxy_idx_count, 
        &ptr_to_arrays, &number_vertices,
        np, rank);
      idx_count = proxy_idx_count + VDGX_CACHE_LEN;
      #pragma omp barrier
      #pragma omp single
      {
        update_container.edgeList = edge_list_dst;
        update_container.numEdges = 2 * batch.e->numEdges;
        update_container.numVertices = number_vertices;
      }
      if(run_params->insert_edges) {
        merge_GRAPH_TYPE__GRAPH_eTYPE_parallel(static_state, 
            &update_container, idx_count - 1,
            number_vertices, np, rank);
      }
      else {
        demerge_GRAPH_TYPE__GRAPH_eTYPE_parallel(static_state, 
            &update_container, idx_count - 1,
            number_vertices, np, rank);
      }
      #pragma omp barrier
      #pragma omp sections
      {
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
          free(ptr_to_arrays);
        }
      }
    }
    #undef VDGX_CACHE_LEN
    switch(run_params->algos)
    {
      case DYNAMIC_CONSTANT_CPU:
        static_state->numVertices = statin->numVertices;
        static_state->numEdges = statin->numEdges;
        reallocate_graph(static_state);
        copy_graph(statin, static_state);
        break;
      default:
        break;
    }
  }
  set_point(end);
  fprintf(meta_results_out, "Generation completed in %9.6fs\n",
    find_difference(start, end));
  fprintf(meta_results_out, "Running static algorithm:\n");
  reallocate_graph(static_state);
  copy_graph(statin, static_state);
  any_dynamicTC a_fn;
  any_dynamicstate a_state;
  which_dynamicTC a_type;
  
  a_fn.STTIC = all_fns[static_idx].fn->STTIC;
  a_state.STTIC = static_state;
  a_type = STTIC;

  benchmark_TCD_performance_capture(&a_fn, &a_state, a_type, 
    originalGraph, &batc, &batch, verifying_states, statin, 
    1, 0, 0, meta_results_out, 
    all_fns[static_idx].name, run_params, batch_times, total_times, batch_counts, 
    0);
  print_TCD_results(outfile, INFILENAME, statin, total_times, 
    batch_times, edge_counts, edge_counts2, all_fns[static_idx].name, batch_counts, 
    LOOP_CNT, run_params, 0, 1);

#define COMMON_CALL(hash_len_vtx) benchmark_TCD_performance_capture(&a_fn, &a_state, a_type, \
    originalGraph, &batc, &batch, verifying_states, statin, \
    all_fns[i].fn_len, fn_offset, hash_len_vtx, meta_results_out, \
    all_fns[i].name, run_params, batch_times, total_times, batch_counts, start_count); 
  
  for(unsigned int i = 0, fn_offset = 1; i < all_fns_len; ++i) {
    if(i == static_idx)
      continue;
    switch(all_fns[i].fn_type) {
      case ALT:
        {
          const UINT_t alt_hash_len_vtx = initialize_alt_state(&alt_superstates, 
          statin);
          a_fn.ALT = all_fns[i].fn->ALT;
          a_state.ALT = alt_superstates;
          a_type = ALT;
          // benchmark_TCD_performance_capture(&a_fn, &a_state, ALT, 
          //   originalGraph, &batc, &batch, verifying_states, statin, 
          //   all_fns[i].fn_len, fn_offset, alt_hash_len_vtx, 
          //   meta_results_out, all_fns[i].name, run_params, 
          //   batch_times, total_times, batch_counts);
          COMMON_CALL(alt_hash_len_vtx);
          free_alt_state(alt_superstates);
          free(alt_superstates);
          break;
        }
      case DBTL:
        initialize_DBTL_graph(&dbtl_superstates, statin);
        a_fn.DBTL = all_fns[i].fn->DBTL;
        a_state.DBTL = dbtl_superstates;
        a_type = DBTL;
        // benchmark_TCD_performance_capture(&a_fn, &a_state, DBTL, 
        //   originalGraph, &batc, &batch, verifying_states, statin, 
        //   all_fns[i].fn_len, 1, 0, fn_offset, 
        //   meta_results_out, all_fns[i].name, run_params, 
        //   batch_times, total_times, batch_counts);
        COMMON_CALL(0);
        free_DBTL_graph(dbtl_superstates);
        free(dbtl_superstates);
        break;
      
      case STTIC:
      default:
        fprintf(stderr, "Something went wrong... Unknown or static function "
          "passed for benchmarking as a candidate function.\n");
        break;
    }
    print_TCD_results(outfile, INFILENAME, statin, total_times, 
      batch_times, edge_counts, edge_counts2, all_fns[i].name, batch_counts, 
      LOOP_CNT, run_params, fn_offset, all_fns[i].fn_len);
    fn_offset += all_fns[i].fn_len;
  }
#undef COMMON_CALL

  free_TCD_benchmarking(batc.e, batch.e, 
    &static_state, run_params);
  
  free_TCD_stat_records(&batch_times, &total_times, 
    &batch_counts, &edge_counts, 
    &edge_counts2, run_params);
}

static inline UINT_t mapParseInt(char** in_data, char *end_data) {
  UINT_t read_data = 0;

  while(*in_data < end_data && (**in_data < '0' || **in_data > '9')) {
    (*in_data)++;
  }

  while(*in_data < end_data && **in_data >= '0' && **in_data <= '9') {
    read_data *= 10;
    read_data += (**in_data - '0');
    (*in_data)++;
  }
  return read_data;
}

#define EDGE_T_PER_CACHE (TC_CACHE_LINE_SIZE / sizeof(edge_t))

static void readMatrixMarketFileEdges(const char *filename, GRAPH_eTYPE* edgestream, GRAPH_TYPE* graph) {
#ifdef _WIN32
  LARGE_INTEGER start;
  LARGE_INTEGER end;
#else
  struct timespec start;
  struct timespec end;
#endif
  set_point(start);
  FILE *infile = fopen(filename, "r");
  if (infile == NULL) {
    printf("Error opening file %s.\n", filename);
    exit(1);
  }

  char line[256];
  size_t num_rows, num_cols, num_entries;

  // Skip the header lines
  do {
    fgets(line, sizeof(line), infile);
  } while (line[0] == '%');

  sscanf(line, "%zu %zu %zu", &num_rows, &num_cols, &num_entries);

  if (num_rows != num_cols) {
    fprintf(stderr,"ERROR: Matrix Market input file is not square: rows: %zu  cols: %zu  nnz: %zu\n",
	    num_rows, num_cols, num_entries);
    exit(-1);
  }

#if DEBUG
  printf("readMatrixMarketFile: %d %d %d\n",num_rows, num_cols, num_entries);
#endif

  size_t edgeCount = 0;

  edge_t * edgesAWay = (edge_t*) calloc(num_entries, sizeof(edge_t));
  assert_malloc(edgesAWay);

  size_t edgeOneCount, edgeCountNoDup; 
  GRAPH_TYPE* graph1, *graph2;

  size_t file_size;
  fclose(infile);
#ifdef _WIN32
  HANDLE inhandle = CreateFileA(filename, 
    GENERIC_READ, FILE_SHARE_READ, 
    NULL, OPEN_EXISTING, 
    FILE_ATTRIBUTE_NORMAL, NULL);
  if(inhandle == INVALID_HANDLE_VALUE) {
    printf("Error opening Handle for file %s.\n", filename);
    exit(1);
  }

  LARGE_INTEGER handle_size;
  GetFileSizeEx(inhandle, &handle_size);

  file_size = handle_size.QuadPart;

  HANDLE mapping = CreateFileMappingA(inhandle, 
    NULL, PAGE_READONLY, 
    0, 0, NULL);
  
  char * data = (char*)MapViewOfFile(mapping, 
    FILE_MAP_READ, 0, 
    0, 0);
#else
  int inhandle = open(filename, O_RDONLY);

  struct stat inhandle_stat;
  fstat(inhandle, &inhandle_stat);

  file_size = inhandle_stat.st_size;

  char * data = (char*)mmap(
    NULL, file_size,
    PROT_READ,
    MAP_PRIVATE,
    inhandle, 0 
  );
#endif
  char * start_data = data;
  char * finished_data = data + file_size;

  size_t rounded_up_num_entries = (1 + (num_entries) / SIZE_CACHE_LEN) *
    SIZE_CACHE_LEN;
  size_t * line_offsets = (size_t*)alloc_aligned(TC_CACHE_LINE_SIZE,
    (rounded_up_num_entries) * sizeof(size_t));
  assert_malloc(line_offsets);
  line_offsets[0] = 0;

  while(*start_data == '%'){
    while(*start_data != '\n') {
      start_data++;
    }
    start_data++;
  } // Ignore commented %
  while(*start_data != '\n') {
    start_data++;
  } // Ignore line read by fopen
  char *moving_data;
  omp_set_num_threads(omp_get_num_procs() / 2);
  #pragma omp parallel default(shared)
  {
    const unsigned int np = omp_get_num_threads();
    const unsigned int rank = omp_get_thread_num();
    const size_t cache_individual_len_reduced = (2 * num_entries) / np +
      ((2 * num_entries % np) > 0 ? 1 : 0);
    const size_t cache_individual_len = (1 + (
      cache_individual_len_reduced) / EDGE_T_PER_CACHE) *
    EDGE_T_PER_CACHE;

    size_t * thread_offset, *state, *prefix_helper;
    size_t edgeCountNoDup;

    edge_t* edges, *edgesNoDup_p, *edges2;
    
    char *my_cursor, *end_cursor;

    #pragma omp single copyprivate(thread_offset,state,edges,edgesNoDup_p,prefix_helper)
    {
      thread_offset = (size_t*)malloc((np + 1) *
       sizeof(size_t));
      assert_malloc(thread_offset);
      thread_offset[0] = 0;
      prefix_helper = (size_t*)malloc((np + 1) *
        sizeof(size_t));
      assert_malloc(prefix_helper);
      prefix_helper[0] = 0;
      state = (size_t*)alloc_aligned(TC_CACHE_LINE_SIZE,
        np * SIZE_CACHE_LEN * sizeof(size_t));
      assert_malloc(state);
      edges = (edge_t*)alloc_aligned(TC_CACHE_LINE_SIZE, 
        np * cache_individual_len * sizeof(edge_t));
      assert_malloc(edges);
      edgesNoDup_p = (edge_t*)alloc_aligned(TC_CACHE_LINE_SIZE,
        np * cache_individual_len * sizeof(edge_t));
      assert_malloc(edgesNoDup_p);
    }
    
    const size_t candidate_byte_start = ((finished_data - start_data) 
      * rank) / np;
    const size_t candidate_byte_end = ((finished_data - start_data) 
      * (rank + 1)) / np;

    size_t byte_start = candidate_byte_start;
    size_t byte_end = candidate_byte_end;
    if(rank < np - 1) {
      while(start_data[byte_end] != '\n') {
        byte_end++;
      }
    }
    if (rank > 0) {
      while(start_data[byte_start] != '\n') {
        byte_start++;
      }
    }

    my_cursor = start_data;

    size_t * local_offsets = (size_t*)malloc((num_entries) 
    * sizeof(size_t));
    assert_malloc(local_offsets);

    size_t j = 0;
    local_offsets[j] = 0;
    if(byte_end > byte_start) {
      for(size_t i = byte_start; i < byte_end; ++i) {
        if(my_cursor[i] == '\n') {
          local_offsets[j] = i;
          j++;
        }
      }
    }

    const size_t calculated_length = j;

    if(calculated_length > 0) {
      prefix_helper[(rank + 1)] = local_offsets[calculated_length - 1];
    }
    else {
      prefix_helper[rank + 1] = 0;
    }

    thread_offset[rank + 1] = calculated_length;

    #pragma omp barrier
    #pragma omp single
    {
      for(size_t i = 0; i < np; ++i) {
        thread_offset[i + 1] += thread_offset[i];
        prefix_helper[i + 1] += prefix_helper[i];
      }
      if(thread_offset[np] != num_entries + 1) {
        printf("Error in number of entries specified.\n");
        exit(1);
      }
    }

    const size_t calculated_offset = thread_offset[rank];

    const size_t byte_offsets = prefix_helper[rank];

    if(calculated_length >= 1) {
      memcpy(line_offsets + calculated_offset, local_offsets, 
        (calculated_length) * sizeof(size_t));
    }
    
    free(local_offsets);
    state[rank * SIZE_CACHE_LEN] = ((rank + 1) * 2 * num_entries) / np -
      (rank * 2 * num_entries) / np;

    #pragma omp barrier
    #pragma omp single
    {
      free(thread_offset);
    }

    size_t prev_state_sum = 0;
    size_t current_state_limit_ref = 0;

    #pragma omp for schedule(static,EDGE_T_PER_CACHE)
    for(size_t i = 0; i < num_entries; ++i) {
      my_cursor = start_data + line_offsets[i];
      end_cursor = start_data + line_offsets[i + 1];
      const UINT_t a = mapParseInt(&my_cursor, 
        end_cursor);
      my_cursor++;
      const UINT_t b = mapParseInt(&my_cursor, 
        end_cursor);
      const size_t index_in_current_cache = i - prev_state_sum;
      if(index_in_current_cache >= state[current_state_limit_ref]) {
        prev_state_sum += state[current_state_limit_ref];
        current_state_limit_ref++;
      }
      edges[i * 2] = (edge_t) {
        a - 1, b - 1
      };
      edges[i * 2 + 1] = (edge_t) {
        b - 1, a - 1
      };
    }

    const size_t my_cache_line_len = state[rank * SIZE_CACHE_LEN];
    edge_t* my_personal_cache = edgesNoDup_p + rank * cache_individual_len;
    const edge_t* my_personal_cache_input = edges + (rank * 2 * num_entries) / np;
    memcpy(my_personal_cache, my_personal_cache_input, 
      my_cache_line_len * sizeof(edge_t));
    qsort(my_personal_cache, state[rank * SIZE_CACHE_LEN], 
      sizeof(edge_t), compareEdge_t);

    #pragma omp barrier
    #pragma omp single
    {
      free_aligned(line_offsets);
    }

    mergepath_sorted_parallel(cache_individual_len, edges, 
      edgesNoDup_p, sizeof(edge_t), state, 
      np, compareEdge_t, np, rank);
    
    #pragma omp barrier

    memcpy(my_personal_cache, my_personal_cache_input,
      my_cache_line_len * sizeof(edge_t));
    
    state[rank * SIZE_CACHE_LEN] = ((rank + 1) * 2 * num_entries) / np -
      (rank * 2 * num_entries) / np;

    #pragma omp barrier

    pairwise_deduplicate_parallel(cache_individual_len, edgesNoDup_p,
      sizeof(edge_t), state, np, 
      compareEdge_t, keepOneEdge_t, np, rank);
    
    prefix_helper[rank + 1] = state[rank * SIZE_CACHE_LEN];
    
    #pragma omp barrier
    #pragma omp single copyprivate(edgeCountNoDup)
    {
      prefix_helper[0] = 0;
      for(size_t i = 0; i < np; ++i) {
        prefix_helper[i + 1] += prefix_helper[i];
      }
      edgeCountNoDup = prefix_helper[np];
      graph->numVertices = num_rows;
      graph->numEdges = edgeCountNoDup;
      allocate_graph(graph);
    }

    edge_t* reconstruct_edges = edges + prefix_helper[rank];
    memcpy(reconstruct_edges, my_personal_cache, 
      sizeof(edge_t) * state[rank * SIZE_CACHE_LEN]);
    
    #pragma omp barrier    

    if(graph->numVertices < np) {
      #pragma omp single
      {
        convert_edges_to_graph(edges, graph);
      } 
    }
    else {
      convert_edge_to_graph_parallel(edges, graph, np, rank);
    }

    #pragma omp barrier

    const edge_t* personal_one_way_in = edgesNoDup_p + rank * 
      cache_individual_len;
    for(size_t j = 0, i = 0; j < state[rank * SIZE_CACHE_LEN]; ++j) {
      if(personal_one_way_in[j].src < personal_one_way_in[j].dst) {
        i++;
      }
      prefix_helper[rank + 1] = i;
    }
    
    #pragma omp barrier
    #pragma omp single
    {
      for(size_t i = 0; i < np; ++i) {
        prefix_helper[i + 1] += prefix_helper[i];
      }
      edgeOneCount = prefix_helper[np];
      free_aligned(edges);
    }

    edge_t* const my_output_loc = edgesAWay + prefix_helper[rank];
    for(size_t i = 0, k = 0; 
      k < prefix_helper[rank + 1] - prefix_helper[rank]; ++i) {
      if(personal_one_way_in[i].src < personal_one_way_in[i].dst) {
        my_output_loc[k] = personal_one_way_in[i];
        k++;
      }
    }

    #pragma omp barrier
    #pragma omp sections
    {
      #pragma omp section
      {
        free_aligned(state);
      }
      #pragma omp section
      {
        free_aligned(prefix_helper);  
      }
      #pragma omp section
      {
        free_aligned(edgesNoDup_p);
      }
      #pragma omp section
      {
        edgestream->numVertices = num_rows;
        edgestream->numEdges = edgeOneCount;
        edgestream->edgeList = edgesAWay;
      } 
    }
  }
#ifdef _WIN32
  UnmapViewOfFile(data);
  CloseHandle(mapping);
  CloseHandle(inhandle);
#else
  munmap(data, file_size);
  close(inhandle);
#endif
  set_point(end);
  printf("Time taken to read file: %lf\n", 
    find_difference(start, end));

  return;
}

static void readMatrixMarketFile(const char *filename, GRAPH_TYPE* graph) {
  GRAPH_eTYPE update_e;
  readMatrixMarketFileEdges(filename, &update_e, graph);
  free(update_e.edgeList);
}

#undef EDGE_T_PER_CACHE

void parallel_algorithms_announcer(FILE* my_outfile, const BenchmarkingParameters * a_param) {
  omp_set_num_threads(a_param->parallel_count);
  #pragma omp parallel
  {
    if(!QUIET) {
      #pragma omp master
        fprintf(my_outfile,"OpenMP threads: %12d\n",omp_get_num_threads());
    }
  }
}

int main(int argc, char **argv) {
  UINT_t numTriangles = 0;

  outfile = stdout;
  BenchmarkingParameters* all_params = NULL;
  int benchmark_length = 0, allocated_benchmark_length = 0;
  const int exit_code = parseFlags(argc, argv, true, &all_params, 
    &benchmark_length, &allocated_benchmark_length);
  if(exit_code != 0)
    exit(exit_code);
  
  GRAPH_TYPE 
    *originalGraph,
    *graph,
    *rev_originalGraph = NULL,
    *rev_graph = NULL;
  
  GRAPH_eTYPE
    *originalGraph_e,
    *graph_e,
    *rev_originalGraph_e = NULL,
    *rev_graph_e = NULL;

  originalGraph = (GRAPH_TYPE *)malloc(sizeof(GRAPH_TYPE));
  assert_malloc(originalGraph);
  graph = (GRAPH_TYPE *)malloc(sizeof(GRAPH_TYPE));
  assert_malloc(graph);
  originalGraph_e = (GRAPH_eTYPE*)malloc(sizeof(GRAPH_eTYPE));
  assert_malloc(originalGraph_e);
  graph_e = (GRAPH_eTYPE*)malloc(sizeof(GRAPH_eTYPE));
  assert_malloc(graph_e);
  int i;
  
  if (INFILENAME != NULL) {
    readMatrixMarketFileEdges(INFILENAME, originalGraph_e, originalGraph);
      
    // rev_originalGraph_e = reorder_graphe_by_degree(originalGraph_e, REORDER_HIGHEST_DEGREE_FIRST);
    
    // rev_graph_e = originalGraph_e;
    // originalGraph_e = rev_originalGraph_e;
    // rev_originalGraph_e = rev_graph_e;
    // rev_graph_e = NULL;
    
    graph_e->numEdges = originalGraph_e->numEdges;
    graph_e->numVertices = originalGraph_e->numVertices;
    allocate_graphe(graph_e);
    

    // rev_originalGraph = reorder_graph_by_degree(originalGraph, REORDER_HIGHEST_DEGREE_FIRST);

    // rev_graph = originalGraph;
    // originalGraph = rev_originalGraph;
    // rev_originalGraph = rev_graph;
    // rev_graph = NULL;

    graph->numVertices = originalGraph->numVertices;
    graph->numEdges = originalGraph->numEdges;
    allocate_graph(graph);
  }
  else {
    fprintf(stderr,"ERROR: No input graph selected.\n");
    exit(8);
  }
  
  copy_graph(originalGraph, graph);

  omp_set_num_threads(omp_get_max_threads() / 4);
  numTriangles = tc_MapJIK_P(graph);

  correctTriangleCount = numTriangles;

  if (!QUIET)
    fprintf(outfile,"Graph has %zu vertices and %zu undirected edges. Timing loop count %d.\n", 
      originalGraph->numVertices, originalGraph->numEdges/2, LOOP_CNT);
  
  any_dynamicTC_len dynamic_ip[] = {
    {
      .fn = &(any_dynamicTC) {
        &(static_dtc) {
          STATIC_DYNAMIC_PARALLEL_ALGS
        }
      }, 
      .fn_type=STTIC, 
      .fn_len=STATIC_DYNAMIC_PARALLEL_ALGS_LEN, 
      .name=&(char[NAME_LEN]) {
        STATIC_DYNAMIC_PARALLEL_ALGS_STRINGS
      }
    },
    {
      .fn = &(any_dynamicTC) {
        .ALT = &(alt_dtc) {
          .iow = (alt_iow[]) {
            ALT_DYNAMIC_PARALLEL_ALGS
          }
        }
      },
      .fn_type=ALT,
      .fn_len=ALT_DYNAMIC_PARALLEL_ALGS_LEN,
      .name=(char[][NAME_LEN]) {
        ALT_DYNAMIC_PARALLEL_ALGS_STRINGS
      }
    },
    {
      .fn = &(any_dynamicTC) {
        .DBTL = &(dbtl_dtc) {
          .iow = (dbtl_iow[]) {
            DBTL_DYNAMIC_PARALLEL_ALGS
          }
        }
      },
      .fn_type = DBTL,
      .fn_len = DBTL_DYNAMIC_PARALLEL_ALGS_LEN,
      .name = (char[][NAME_LEN]) {
        DBTL_DYNAMIC_PARALLEL_ALGS_STRINGS
      }
    }
  };
  any_dynamicTC_len dynamic_wp[] = {
    {
      .fn = &(any_dynamicTC) {
        &(static_dtc) {
          W_STATIC_DYNAMIC_PARALLEL_ALGS
        }
      }, 
      .fn_type=STTIC, 
      .fn_len=W_STATIC_DYNAMIC_PARALLEL_ALGS_LEN, 
      .name=&(char[NAME_LEN]) {
        W_STATIC_DYNAMIC_PARALLEL_ALGS_STRINGS
      }
    },
    {
      .fn = &(any_dynamicTC) {
        .ALT = &(alt_dtc) {
          .iow = (alt_iow[]) {
            W_ALT_DYNAMIC_PARALLEL_ALGS
          }
        }
      },
      .fn_type=ALT,
      .fn_len=W_ALT_DYNAMIC_PARALLEL_ALGS_LEN,
      .name=(char[][NAME_LEN]) {
        W_ALT_DYNAMIC_PARALLEL_ALGS_STRINGS
      }
    },
    {
      .fn = &(any_dynamicTC) {
        .DBTL = &(dbtl_dtc) {
          .iow = (dbtl_iow[]) {
            W_DBTL_DYNAMIC_PARALLEL_ALGS
          }
        }
      },
      .fn_type = DBTL,
      .fn_len = W_DBTL_DYNAMIC_PARALLEL_ALGS_LEN,
      .name = (char[][NAME_LEN]) {
        W_DBTL_DYNAMIC_PARALLEL_ALGS_STRINGS
      }
    }
  };
  any_dynamicTC_len* ins, *wit;
  unsigned int ins_n, wit_n;

  ins = dynamic_ip;
  ins_n = sizeof(dynamic_ip) / sizeof(dynamic_ip[0]);
  wit = dynamic_wp;
  wit_n = sizeof(dynamic_wp) / sizeof(dynamic_wp[0]);
  for(i = 0; i < benchmark_length; ++i) {
    const BenchmarkingParameters test_bench = all_params[i];
    printf("\n  Running with parameters:\n\n");
    printf("  Number of processes: %d\n", 
      test_bench.parallel_count == 0 ? 1 : test_bench.parallel_count);
    if(test_bench.output_is_stdout) {
      outfile = stdout;
      printf("  Output is : stdout\n");
    }
    else {
      outfile = fopen(test_bench.output_file, "w");
      printf("  Output is : %s\n", test_bench.output_file);
    }
    switch(test_bench.algos) {
      case DYNAMIC_CONSTANT_CPU:
      case DYNAMIC_REALTIME_CPU:
        {
          printf("  Number of batches: %d\n", test_bench.batch_count);
          char combo = (test_bench.insert_edges << 1) | test_bench.withdraw_edges;
          switch(combo) {
            case 0:
              break;
            case 1:
              benchmark_TCD(wit, wit_n, 0, 
                originalGraph_e, originalGraph, 
                &test_bench);
              break;
            case 2:
              benchmark_TCD(ins, ins_n, 0, 
                originalGraph_e, originalGraph, 
                &test_bench);
              break;
            case 3:
              break;
          }
          break;
        }        
      case META:
      default:
        break;
    }
    if(!test_bench.output_is_stdout)
      fclose(outfile);
  }
  
  free_graph(originalGraph);
  free_graph(graph);
  if(rev_originalGraph != NULL)
    free_graph(rev_originalGraph);
  if(rev_graph != NULL)
    free_graph(rev_graph);

  free_graphe(originalGraph_e);
  free_graphe(graph_e);
  if(rev_originalGraph_e != NULL)
    free_graphe(rev_originalGraph_e);
  if(rev_graph_e != NULL)
    free_graphe(rev_graph_e);

  free(all_params);
  fclose(outfile);
  
  return(0);
}