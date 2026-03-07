CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c11 -D_POSIX_C_SOURCE=200809L
LDLIBS  = -lm -lzstd -llz4
TESTDATA = ../ucodec/testdata

.PHONY: all test clean

all: libshard_v2.a test_golden test_writer test_metadata test_mmap test_streaming test_compression test_safety

libshard_v2.a: shard_v2.o
	ar rcs $@ $^

shard_v2.o: shard_v2.c shard_v2.h
	$(CC) $(CFLAGS) -c $< -o $@

test_golden: test_golden.c shard_v2.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_writer: test_writer.c shard_v2.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_metadata: test_metadata.c shard_v2.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_mmap: test_mmap.c shard_v2.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_streaming: test_streaming.c shard_v2.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_compression: test_compression.c shard_v2.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_safety: test_safety.c shard_v2.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) -lpthread

test: test_golden test_writer test_metadata test_mmap test_streaming test_compression test_safety
	./test_golden $(TESTDATA)
	./test_writer
	./test_metadata
	./test_mmap $(TESTDATA)
	./test_streaming
	./test_compression
	./test_safety ../ucodec/testdata/safety

clean:
	rm -f *.o *.a test_golden test_writer test_metadata test_mmap test_streaming test_compression test_safety
