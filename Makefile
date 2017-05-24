OBJS := run.o kmeans_slow.o kmeans_fast.o common.o kmeans_medi.o kmeans_report.o kmeans_lloyd.o
#OBJ2 := kmeans_lloyd.o
CXXFLAGS_RELEASE := -Wall -Wextra -O2
CXXFLAGS_DEBUG := -g -Wall -Wextra -fsanitize=address -DDEBUG
CXXFLAGS := $(CXXFLAGS_RELEASE)
EXEC := run
CXX = g++

all: $(EXEC)

clean:
	$(RM) $(EXEC) $(OBJS) $(OBJS:.o=.d)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.d: %.c
	$(SHELL) -ec '$(CXX) -MM $(CXXFLAGS) $< | sed "s/$*\\.o/& $@/g" > $@'

include $(OBJS:.o=.d)
