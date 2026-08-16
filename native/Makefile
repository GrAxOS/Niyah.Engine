CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -pedantic -O2 -Wconversion -Wshadow
LDLIBS ?= -lm

CORE_OBJS = niyah_core.o niyah_evidence.o
GRAPH_OBJS = niyah_graph.o
SEARCH_OBJS = niyah_search.o
CRAWL_OBJS = niyah_crawler.o
DOCUMENT_OBJS = niyah_document.o
RUNTIME_OBJS = niyah_runtime.o niyah_core.o niyah_graph.o

TEST_BINS = \
	test_niyah_core \
	test_niyah_graph \
	test_niyah_search \
	test_niyah_crawler \
	test_niyah_document \
	test_niyah_runtime

.PHONY: all test clean

all: $(TEST_BINS)

test_niyah_core: $(CORE_OBJS) test_niyah_core.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_niyah_graph: $(GRAPH_OBJS) niyah_graph_test.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_niyah_search: $(SEARCH_OBJS) niyah_search_test.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_niyah_crawler: $(CRAWL_OBJS) niyah_crawler_test.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_niyah_document: $(DOCUMENT_OBJS) test_niyah_document.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_niyah_runtime: $(RUNTIME_OBJS) niyah_runtime_test.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test: all
	./test_niyah_core
	./test_niyah_graph
	./test_niyah_search
	./test_niyah_crawler
	./test_niyah_document
	./test_niyah_runtime

clean:
	rm -f *.o $(TEST_BINS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
