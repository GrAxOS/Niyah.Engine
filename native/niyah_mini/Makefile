# Makefile for NiyahMini

CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wno-unused-parameter -I.
LDFLAGS ?= -lm

# Required for Windows compatibility
CPPFLAGS ?= -DNIYAH_BRIDGE_EXPORTS

# Sources
SRCS = \
	niyah_mini_config.c \
	niyah_mini_vocab.c \
	niyah_mini_model.c \
	niyah_mini_bridge.c

# Headers
HDRS = \
	niyah_mini_config.h \
	niyah_mini_vocab.h \
	niyah_mini_model.h \
	niyah_mini_bridge.h

# Object files
OBJS = $(SRCS:.c=.o)

# Library
LIB = libniyah_mini.a

# Tests
TESTS = \
	test_niyah_mini_config \
	test_niyah_mini_vocab \
	test_niyah_mini_model

.PHONY: all lib test clean

all: lib

lib: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

# Tests
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

clean:
	rm -f $(OBJS) $(LIB) $(TESTS)
