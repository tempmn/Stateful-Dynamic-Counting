CFLAGS = -fopenmp -O3

SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)

all: tc

%.o: %.c
	${CC} ${CFLAGS} ${OPTS} -c $< -o $@

tc: $(OBJS)
	${CC} ${CFLAGS} ${OPTS} -o $@ $(OBJS) -lm

clean: 
	rm -f core *~ $(OBJS) tc