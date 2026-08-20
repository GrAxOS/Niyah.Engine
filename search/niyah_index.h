#include "niyah_index.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NIYAH_DOCUMENT_TOKEN_LIMIT 1024u
#define NIYAH_QUERY_TOKEN_LIMIT 128u
#define NIYAH_INITIAL_TERM_CAPACITY 128u
#define NIYAH_INITIAL_DOCUMENT_CAPACITY 64u
#define NIYAH_INITIAL_POSTING_CAPACITY 16u

typedef struct {
    size_t term_count;
    size_t document_count;
    double average_document_length;
    size_t *posting_counts;
    uint32_t *document_frequencies;
    size_t posting_count;
} NiyahIndexCheckpoint;

static int term_compare(const void *a, const void *b) {
    const NiyahSearchHit *ha = (const NiyahSearchHit *)a;
    const NiyahSearchHit *hb = (const NiyahSearchHit *)b;

    if (ha->score < hb->score) return 1;
    if (ha->score > hb->score) return -1;

    if (ha->document_id < hb->document_id) return -1;
    if (ha->document_id > hb->document_id) return 1;

    return 0;
}

static bool token_byte(unsigned char c) {
    return isalnum(c) != 0 ||
           c >= 0x80u ||
           c == '_' ||
           c == '-';
}

static size_t tokenize(
    const char *text,
    char tokens[][NIYAH_TERM_MAX],
    size_t max_tokens
) {
    if (!text || !tokens || max_tokens == 0) {
        return 0;
    }

    size_t count = 0;
    const unsigned char *cursor =
        (const unsigned char *)text;

    while (*cursor && count < max_tokens) {
        while (*cursor && !token_byte(*cursor)) {
            ++cursor;
        }

        if (!*cursor) {
            break;
        }

        size_t length = 0;

        while (*cursor && token_byte(*cursor)) {
            if (length + 1u < NIYAH_TERM_MAX) {
                const unsigned char byte = *cursor;

                tokens[count][length++] =
                    (char)(
                        byte < 0x80u
                            ? tolower(byte)
                            : byte
                    );
            }

            ++cursor;
        }

        tokens[count][length] = '\0';

        if (length > 0) {
            ++count;
        }
    }

    return count;
}

static bool token_seen_before(
    char tokens[][NIYAH_TERM_MAX],
    size_t index,
    const char *token
) {
    if (!tokens || !token) {
        return false;
    }

    for (size_t i = 0; i < index; ++i) {
        if (strcmp(tokens[i], token) == 0) {
            return true;
        }
    }

    return false;
}

static NiyahTermEntry *find_term(
    NiyahInvertedIndex *index,
    const char *term
) {
    if (!index || !term) {
        return NULL;
    }

    for (size_t i = 0; i < index->term_count; ++i) {
        if (strcmp(index->terms[i].term, term) == 0) {
            return &index->terms[i];
        }
    }

    return NULL;
}

static const NiyahTermEntry *find_term_const(
    const NiyahInvertedIndex *index,
    const char *term
) {
    if (!index || !term) {
        return NULL;
    }

    for (size_t i = 0; i < index->term_count; ++i) {
        if (strcmp(index->terms[i].term, term) == 0) {
            return &index->terms[i];
        }
    }

    return NULL;
}

static size_t document_position(
    const NiyahInvertedIndex *index,
    uint64_t document_id
) {
    if (!index || document_id == 0) {
        return SIZE_MAX;
    }

    for (size_t i = 0; i < index->document_count; ++i) {
        if (index->documents[i].document_id == document_id) {
            return i;
        }
    }

    return SIZE_MAX;
}

const NiyahDocument *niyah_index_document(
    const NiyahInvertedIndex *index,
    uint64_t document_id
) {
    if (!index || document_id == 0) {
        return NULL;
    }

    const size_t position =
        document_position(index, document_id);

    if (position == SIZE_MAX) {
        return NULL;
    }

    return &index->documents[position];
}

static bool checked_growth(
    size_t current,
    size_t initial,
    size_t element_size,
    size_t *next_out
) {
    if (!next_out || element_size == 0) {
        return false;
    }

    size_t next =
        current == 0
            ? initial
            : current * 2u;

    if (next < current ||
        next > SIZE_MAX / element_size) {
        return false;
    }

    *next_out = next;
    return true;
}

static bool grow_terms(
    NiyahInvertedIndex *index
) {
    if (!index) {
        return false;
    }

    size_t next = 0;

    if (!checked_growth(
            index->term_capacity,
            NIYAH_INITIAL_TERM_CAPACITY,
            sizeof(*index->terms),
            &next)) {
        return false;
    }

    NiyahTermEntry *terms =
        realloc(index->terms, next * sizeof(*terms));

    if (!terms) {
        return false;
    }

    if (next > index->term_capacity) {
        memset(
            terms + index->term_capacity,
            0,
            (next - index->term_capacity) * sizeof(*terms)
        );
    }

    index->terms = terms;
    index->term_capacity = next;

    return true;
}

static bool grow_documents(
    NiyahInvertedIndex *index
) {
    if (!index) {
        return false;
    }

    size_t next = 0;

    if (!checked_growth(
            index->document_capacity,
            NIYAH_INITIAL_DOCUMENT_CAPACITY,
            sizeof(*index->documents),
            &next)) {
        return false;
    }

    NiyahDocument *documents =
        realloc(index->documents, next * sizeof(*documents));

    if (!documents) {
        return false;
    }

    index->documents = documents;
    index->document_capacity = next;

    return true;
}

static bool grow_postings(
    NiyahTermEntry *entry
) {
    if (!entry) {
        return false;
    }

    size_t next = 0;

    if (!checked_growth(
            entry->posting_capacity,
            NIYAH_INITIAL_POSTING_CAPACITY,
            sizeof(*entry->postings),
            &next)) {
        return false;
    }

    NiyahPosting *postings =
        realloc(entry->postings, next * sizeof(*postings));

    if (!postings) {
        return false;
    }

    entry->postings = postings;
    entry->posting_capacity = next;

    return true;
}

static bool ensure_term_capacity(
    NiyahInvertedIndex *index,
    size_t required
) {
    if (!index) {
        return false;
    }

    while (index->term_capacity < required) {
        if (!grow_terms(index)) {
            return false;
        }
    }

    return true;
}

static bool checkpoint_create(
    const NiyahInvertedIndex *index,
    NiyahIndexCheckpoint *checkpoint
) {
    if (!index || !checkpoint) {
        return false;
    }

    memset(checkpoint, 0, sizeof(*checkpoint));

    checkpoint->term_count = index->term_count;
    checkpoint->document_count = index->document_count;
    checkpoint->average_document_length =
        index->average_document_length;

    if (index->term_count == 0) {
        return true;
    }

    checkpoint->posting_counts =
        calloc(
            index->term_count,
            sizeof(*checkpoint->posting_counts)
        );

    checkpoint->document_frequencies =
        calloc(
            index->term_count,
            sizeof(*checkpoint->document_frequencies)
        );

    if (!checkpoint->posting_counts ||
        !checkpoint->document_frequencies) {
        free(checkpoint->posting_counts);
        free(checkpoint->document_frequencies);
        memset(checkpoint, 0, sizeof(*checkpoint));
        return false;
    }

    for (size_t i = 0; i < index->term_count; ++i) {
        checkpoint->posting_counts[i] =
            index->terms[i].posting_count;

        checkpoint->document_frequencies[i] =
            index->terms[i].document_frequency;

        if (SIZE_MAX - checkpoint->posting_count <
            index->terms[i].posting_count) {
            free(checkpoint->posting_counts);
            free(checkpoint->document_frequencies);
            memset(checkpoint, 0, sizeof(*checkpoint));
            return false;
        }

        checkpoint->posting_count +=
            index->terms[i].posting_count;
    }

    return true;
}

static void checkpoint_free(
    NiyahIndexCheckpoint *checkpoint
) {
    if (!checkpoint) {
        return;
    }

    free(checkpoint->posting_counts);
    free(checkpoint->document_frequencies);
    memset(checkpoint, 0, sizeof(*checkpoint));
}

static void rollback_to_checkpoint(
    NiyahInvertedIndex *index,
    const NiyahIndexCheckpoint *checkpoint
) {
    if (!index || !checkpoint) {
        return;
    }

    while (index->term_count > checkpoint->term_count) {
        NiyahTermEntry *entry =
            &index->terms[index->term_count - 1u];

        free(entry->postings);
        memset(entry, 0, sizeof(*entry));
        --index->term_count;
    }

    for (size_t i = 0;
         i < checkpoint->term_count &&
         i < index->term_count;
         ++i) {
        index->terms[i].posting_count =
            checkpoint->posting_counts[i];

        index->terms[i].document_frequency =
            checkpoint->document_frequencies[i];
    }

    index->document_count =
        checkpoint->document_count;

    index->average_document_length =
        checkpoint->average_document_length;
}

void niyah_index_init(
    NiyahInvertedIndex *index,
    double k1,
    double b
) {
    if (!index) {
        return;
    }

    memset(index, 0, sizeof(*index));

    index->k1 =
        isfinite(k1) && k1 > 0.0
            ? k1
            : 1.2;

    index->b =
        isfinite(b) && b >= 0.0 && b <= 1.0
            ? b
            : 0.75;
}

void niyah_index_free(
    NiyahInvertedIndex *index
) {
    if (!index) {
        return;
    }

    for (size_t i = 0; i < index->term_count; ++i) {
        free(index->terms[i].postings);
        index->terms[i].postings = NULL;
        index->terms[i].posting_count = 0;
        index->terms[i].posting_capacity = 0;
        index->terms[i].document_frequency = 0;
    }

    free(index->terms);
    free(index->documents);

    memset(index, 0, sizeof(*index));
}

bool niyah_index_add_document(
    NiyahInvertedIndex *index,
    const NiyahDocument *document
) {
    if (!index ||
        !document ||
        document->document_id == 0) {
        return false;
    }

    if (niyah_index_document(
            index,
            document->document_id)) {
        return false;
    }

    char tokens[
        NIYAH_DOCUMENT_TOKEN_LIMIT
    ][NIYAH_TERM_MAX];

    const size_t token_count =
        tokenize(
            document->text,
            tokens,
            NIYAH_DOCUMENT_TOKEN_LIMIT
        );

    if (token_count > UINT32_MAX) {
        return false;
    }

    NiyahIndexCheckpoint checkpoint;

    if (!checkpoint_create(index, &checkpoint)) {
        return false;
    }

    size_t unique_token_count = 0;

    for (size_t i = 0; i < token_count; ++i) {
        if (!token_seen_before(tokens, i, tokens[i])) {
            ++unique_token_count;
        }
    }

    if (unique_token_count >
        SIZE_MAX - index->term_count) {
        checkpoint_free(&checkpoint);
        return false;
    }

    size_t new_term_count =