OBJS := kmeans_slow.o kmeans_fast.o common.o kmeans_medi.o \
	kmeans_report.o kmeans_lloyd.o kmeans_hirschberg_larmore.o
CXXFLAGS_RELEASE := -Wall -Wextra -O2
CXXFLAGS_DEBUG := -g -Wall -Wextra -fPIE -fsanitize=undefined -DDEBUG #-fsanitize=address
CXXFLAGS := $(CXXFLAGS_DEBUG)
EXEC := run
CXX = g++
TEST := test

all: $(EXEC)

clean:
	$(RM) $(EXEC) $(OBJS)

$(EXEC): $(OBJS) run.c
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

$(TEST) : $(OBJS) test.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^
%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.c
	$(CXX) $(CXXFLAGS) -c -o $@ $<
