CC=g++
CPPFLAGS=-O3 -I ../CXX_TOOLS_LIB -I ../CBM_TOOLS_LIB -I ../CBM_STATE_LIB -I ../CBM_CORE_LIB -I ../CBM_VISUAL_LIB -I ../CBM_DATA_LIB
LDFLAGS=-O3
LDLIBS=-lboost_program_options -lcxx_tools -lcbm_tools -lcbm_state -lcbm_core -lcbm_visual -lcbm_data
LDDIRS=-L /opt/apps/boost/1.45.0/lib -L ../libs
RM=rm -f
TARGET=cbm

SRCS = src/main.cpp
OBJS = $(subst .cpp,.o,$(SRCS))

all: $(OBJS)
	$(CC) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LDDIRS) $(LDLIBS)

clean:
	$(RM) $(OBJS)
	$(RM) $(TARGET)

