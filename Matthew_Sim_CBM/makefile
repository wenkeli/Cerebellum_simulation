CC=g++
NVCC=nvcc
CFLAGS=-O3
RM=rm -f
TARGET=cbm

SRCS = src/main.cpp
OBJS = $(subst .cpp,.o,$(SRCS))

all: $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) $(TARGET)