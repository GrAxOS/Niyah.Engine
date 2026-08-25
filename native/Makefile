# Plain-make build for native/. Mirrors native/CMakeLists.txt.
#
# This Makefile previously listed niyah_attention_multihead.c in SRCS and
# niyah_attention_multihead_test in TESTS. Neither file has ever existed in
# the repository, so `make` failed on the first target.
#
# It then drifted a second time: niyah_sha256.c and niyah_identity.c were
# added to CMakeLists.txt without being added here, and three new tests were
# registered with ctest but not with make. `make test` therefore exercised
# strictly less than `ctest` while appearing to pass. Both lists are now back
# in sync; keep them that way, or delete one build system.

CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wno-unused-parameter -I.
LDFLAGS ?= -lm

# NIYAH_BRIDGE_EXPORTS keeps the evidence/CSP export macros on the
# dllexport side; see the comment in CMakeLists.txt.
CPPFLAGS ?= -DNIYAH_BRIDGE_EXPORTS

SRCS = \
	niyah_core.c \
	niyah_sha256.c \
	niyah_identity.c \
	niyah_matmul.c \
	niyah_softmax.c \
	niyah_rmsnorm.c \
	niyah_swiglu.c \
	niyah_rope.c \
	niyah_attention.c \
	niyah_transformer_layer.c \
	niyah_sampler.c \
	niyah_tokenizer.c \
	niyah_runtime.c \
	niyah_telemetry.c \
	niyah_model.c \
	niyah_llm.c \
	niyah_graph.c \
	niyah_embedding.c \
	niyah_source.c \
	niyah_storage.c \
	niyah_search.c \
	niyah_crawler.c \
	niyah_document.c \
	niyah_bridge.c \
	evidence_envelope.c \
	evidence_graph.c \
	evidence_reasoner.c \
	constraint_solver.c

OBJS = $(SRCS:.c=.o)
LIB  = libniyah.a

TEST_SRCS = \
	test_niyah_core.c \
	test_niyah_document.c \
	niyah_matmul_test.c \
	niyah_softmax_test.c \
	niyah_rmsnorm_test.c \
	niyah_swiglu_test.c \
	niyah_rope_test.c \
	niyah_attention_test.c \
	niyah_transformer_layer_test.c \
	niyah_sampler_test.c \
	niyah_tokenizer_test.c \
	niyah_tokenizer_roundtrip_test.c \
	niyah_config_test.c \
	niyah_identity_test.c \
	niyah_runtime_test.c \
	niyah_telemetry_test.c \
	niyah_graph_test.c \
	niyah_embedding_test.c \
	niyah_search_test.c \
	niyah_storage_test.c \
	niyah_crawler_test.c \
	niyah_bridge_test.c \
	niyah_llm_test.c \
	niyah_llm_generation_test.c \
	niyah_evidence_test.c \
	niyah_csp_test.c

TESTS = $(TEST_SRCS:.c=)

# Measurement instruments. Not tests: they print numbers to read, not
# assertions to pass. `make test` deliberately does not run them.
TOOL_SRCS = \
	niyah_token_tax.c

TOOLS = $(TOOL_SRCS:.c=)

.PHONY: all lib test tools clean

all: lib

lib: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

# Each test links against the static library.
$(TESTS): %: %.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) $(LDFLAGS) -o $@

test: $(TESTS)
	@fail=0; \
	for t in $(TESTS); do \
		printf '%-32s' "$$t"; \
		if ./$$t >/dev/null 2>&1; then \
			echo "ok"; \
		else \
			echo "FAIL"; \
			fail=1; \
		fi; \
	done; \
	exit $$fail

$(TOOLS): %: %.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) $(LDFLAGS) -o $@

tools: $(TOOLS)
	@echo
	@echo "  ./niyah_token_tax --vocab <vocab.txt> \\"
	@echo "                    --parallel testdata/parallel_ar_en.tsv \\"
	@echo "                    --families testdata/root_families_ar.txt"
	@echo

clean:
	rm -f $(OBJS) $(LIB) $(TESTS) $(TOOLS)
