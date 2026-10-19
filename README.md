# Exploring Dynamic Parallel Triangle Counting Algorithms on the CPU

## Pseudocode

Pseudocode for the algorithms may be found in the labelled file.

## Build Instructions

To build with meson, run the following

```
meson setup ./build --buildtype release --native-file ./oneapi.ini
```

Remember to initialize your compiler environment!

`--native-file` can be dropped for GCC compiler. `c_std` may have to be adjusted based on compiler.


Next, go to the build directory and run

```
meson compile
```

A makefile is also provided, but it has only been tested with GCC

Run

```
make clean && make -j 5
```

## Run instructions

Go to the directory where `tc` executable is present.

Look at help with

```
./tc -h
```

To give a brief overview of flags, 

`-Q` Turns off intermediate output  
`-f` Specifies input file  
`-o` Directs output to file  

`-c` resets the graph before adding edges for subsequent loops  
`-s` 'preserves' the graph  

`-i` Insertion of edges  
`-w` Deletion of edges  

`-p` Specifies number of processes  
`-P` Uses all of the processes available on the machine  

Allows you to specify a fraction of edges or the number of edges directly. Use decimals to force interpretation as fraction.

### Advanced usage

Chain multiple flags with `-l`. Common `-f` and `-Q` to be specified on command line. `-l` points to a file with a list of commands.
Other flags may be specified in the commands present in the file. They are "combined" with the common command to synthesize runs.

### Examples

For the paper, we added or deleted a constant number of edges. The relevant option is `-c`.

```
./tc -Q -c 5 -i 0.001 -p 5 -f ../../datasets/amazon0601.mtx
```

```
./tc -Q -c 5 -w 0.05 -p 8 -f ../../datasets/amazon0601.mtx
```