CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -pedantic -O2 -Wconversion -Wshadow

CORE_OBJS = niyah_core.o niyah_evidence.o
GRAPH_OBJS = niyah_graph.o

.PHONY: all test clean

all: test_niyah_core test_niyah_graph

test_niyah_core: $(CORE_OBJS) test_niyah_core.o
	$(CC) $(CFLAGS) -o $@ $^

test_niyah_graph: $(GRAPH_OBJS) niyah_graph_test.o
	$(CC) $(CFLAGS) -o $@ $^

test: test_niyah_core test_niyah_graph
	./test_niyah_core
	./test_niyah_graph

clean:
	rm -f *.o test_niyah_core test_niyah_graph

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
