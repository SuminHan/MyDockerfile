CC  ?= gcc
CXX ?= g++

CFLAGS = -O0 -g -Wall -pedantic 
CXXFLAGS = -O0 -g -Wall -pedantic

OBJ	= main foo
OBJS  = $(addsuffix .o,$(OBJ))
all:
	make compile

compile:
	make $(OBJS)
	make fooexe

fooexe: $(OBJS) 
	$(CXX) -o fooexe $(OBJS)

main.o: main.c
	$(CC) $(CFLAGS) -c -o main.o main.c

foo.o: foo.cpp
	$(CXX) $(CXXFLAGS) -c -o foo.o foo.cpp

clean:
	rm -rf $(OBJS) fooexe *.dSYM

