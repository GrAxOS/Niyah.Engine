CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -pedantic -O2

CORE_OBJS = niyah_core.o niyah_evidence.o

.PHONY: all test clean

all: test_niyah_core

test_niyah_core: $(CORE_OBJS) test_niyah_core.o
	$(CC) $(CFLAGS) -o $@ $^

test: test_niyah_core
	./test_niyah_core

clean:
	rm -f *.o test_niyah_core
