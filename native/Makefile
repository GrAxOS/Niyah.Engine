CC = gcc
CFLAGS = -std=c11 -O3 -Wall -Wextra
LDFLAGS = -lm

SRCS = niyah_core.c niyah_bridge.c niyah_llm.c niyah_model.c niyah_document.c \
       niyah_graph.c niyah_embedding.c niyah_source.c niyah_storage.c \
       niyah_crawler.c niyah_search.c niyah_attention.c niyah_attention_multihead.c \
       niyah_transformer_layer.c niyah_runtime.c niyah_tokenizer.c niyah_telemetry.c \
       niyah_sampler.c niyah_softmax.c niyah_rmsnorm.c niyah_swiglu.c niyah_matmul.c niyah_rope.c

OBJS = $(SRCS:.c=.o)

TESTS = test_niyah_core niyah_attention_test niyah_attention_multihead_test \
        niyah_bridge_test niyah_embedding_test niyah_graph_test niyah_llm_test \
        niyah_llm_generation_test niyah_matmul_test niyah_rmsnorm_test niyah_rope_test \
        niyah_runtime_test niyah_sampler_test niyah_softmax_test niyah_storage_test \
        niyah_swiglu_test niyah_telemetry_test niyah_tokenizer_test \
        niyah_transformer_layer_test niyah_search_test niyah_crawler_test test_niyah_document

.PHONY: all clean test

all: libniyah.a $(TESTS)

libniyah.a: $(OBJS)
	ar rcs $@ $^

$(TESTS): %: %.c libniyah.a
	$(CC) $(CFLAGS) -o $@ $< -L. -lniyah $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TESTS)
	@echo "Running tests..."
	@for test in $(TESTS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "All tests passed!"

clean:
	rm -f $(OBJS) $(TESTS) libniyah.a
