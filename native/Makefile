CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -pedantic -O2 -Wconversion -Wshadow
CPPFLAGS ?= -D_POSIX_C_SOURCE=200809L
LDLIBS ?= -lm

CORE_OBJS = niyah_core.o niyah_evidence.o niyah_telemetry.o
GRAPH_OBJS = niyah_graph.o
SEARCH_OBJS = niyah_search.o
CRAWL_OBJS = niyah_crawler.o
DOCUMENT_OBJS = niyah_document.o
RUNTIME_OBJS = niyah_runtime.o niyah_core.o niyah_graph.o
TELEMETRY_OBJS = niyah_telemetry.o
STORAGE_OBJS = niyah_storage.o
LLM_OBJS = niyah_llm.o niyah_core.o

TEST_BINS = \
	test_niyah_core \
	test_niyah_graph \
	test_niyah_search \
	test_niyah_crawler \
	test_niyah_document \
	test_niyah_runtime \
	test_niyah_telemetry \
	test_niyah_storage \
	test_niyah_llm

.PHONY: all test clean asan

all: $(TEST_BINS)

test_niyah_core: $(CORE_OBJS) test_niyah_core.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_niyah_graph: $(GRAPH_OBJS) niyah_graph_test.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_niyah_search: $(SEARCH_OBJS) niyah_search_test.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_niyah_crawler: $(CRAWL_OBJS) niyah_crawler_test.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_niyah_document: $(DOCUMENT_OBJS) test_niyah_document.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_niyah_runtime: $(RUNTIME_OBJS) niyah_runtime_test.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_niyah_telemetry: $(TELEMETRY_OBJS) niyah_telemetry_test.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_niyah_storage: $(STORAGE_OBJS) niyah_storage_test.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_niyah_llm: $(LLM_OBJS) niyah_llm_test.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

test: all
	./test_niyah_core
	./test_niyah_graph
	./test_niyah_search
	./test_niyah_crawler
	./test_niyah_document
	./test_niyah_runtime
	./test_niyah_telemetry
	./test_niyah_storage
	./test_niyah_llm

asan: CFLAGS += -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
asan: LDLIBS += -fsanitize=address,undefined
asan: clean test

clean:
	rm -f *.o $(TEST_BINS) niyah-telemetry-test.jsonl niyah-storage-test.tmp

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<
