project(
    'dynamic-parallel-cpu-triangle-counting', 
    'c', 
    meson_version: '>=1.1',
    version: '1.0.1',
    default_options: ['c_std=c17']
)

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)

all_deps = [
    m_dep
]

openmp = dependency('openmp', required: true)

all_deps += openmp

sources = files(
    'benchmarking.c',
    'graph.c',
    'main.c',
    'merge.c',
    'tc_dynamic_parallel.c',
    'tc_dynamic_state.c',
    'tc_parallel.c',
)

executable(
    'tc',
    sources,
    dependencies: all_deps
)