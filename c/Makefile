CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c11 -D_POSIX_C_SOURCE=200809L
LDLIBS  = -lm -lzstd -llz4
TESTDATA = ../ucodec/testdata

.PHONY: all test clean

all: libshard.a test_golden test_writer test_metadata test_mmap test_streaming test_compression test_safety

libshard.a: shard.o
	ar rcs $@ $^

shard.o: shard.c shard.h
	$(CC) $(CFLAGS) -c $< -o $@

test_golden: test_golden.c shard.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_writer: test_writer.c shard.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_metadata: test_metadata.c shard.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_mmap: test_mmap.c shard.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_streaming: test_streaming.c shard.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_compression: test_compression.c shard.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_safety: test_safety.c shard.o
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
