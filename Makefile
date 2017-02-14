OBJS := run.o kmeans_slow.o kmeans_fast.o common.o kmeans_medi.o
CFLAGS_RELEASE := -Wall -Wextra -O2
CFLAGS_DEBUG := -g -Wall -Wextra -fsanitize=address
CFLAGS := $(CFLAGS_RELEASE)
EXEC := run

all: $(EXEC)

clean:
	$(RM) $(EXEC) $(OBJS) $(OBJS:.o=.d)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

%.d: %.c
	$(SHELL) -ec '$(CC) -MM $(CPPFLAGS) $< | sed "s/$*\\.o/& $@/g" > $@'

include $(OBJS:.o=.d)
