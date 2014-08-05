#!/bin/make
# ref. http://d.hatena.ne.jp/yutakikuchi/20120702/1341185909

TARGET = netapi
SRCS = netapi.cpp
OBJS = $(SRCS:.cpp=.o)

INCDIR = -I include
LIBDIR = -L lib
LIBS = lib/libnupic_core.a

RM:= rm
CXX:= g++
#CXXFLAGS = -Wall -O2
CXXFLAGS =  -O2


$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBDIR) $(LIBS)

$(OBJS): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $(SRCS)

clean:
	$(RM) $(OBJS) $(TARGET)


hello: hello.o
	$(CXX) -o hello hello.o
helloclean:
	$(RM) hello.o hello
