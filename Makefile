# Files
EXEC_C := test_composer
SRC_C  := composer.cpp test_composer.cpp
OBJ_C  := $(patsubst %.cpp,%.o,$(SRC_C))
# Options
CC	:= g++
CFLAGS	:= -O1
LDFLAGS	:= -L/usr/lib
LDLIBS	:= -lm
# Rules
$(EXEC_C): $(OBJ_C)
	$(CC) -o $@ $^
$(OBJ_C): $(SRC_C)
	$(CC) -c $^
$(SRC_C): composer.h

# Phony targets
.PHONY: clean what clean_composer
clean:
	$(RM) $(EXEC) $(OBJ)

what:
	echo $(OBJ)

clean_composer:
	$(RM) test_composer test_composer.o composer.o 
