#!/bin/make
# ref. http://d.hatena.ne.jp/yutakikuchi/20120702/1341185909

TARGET = netapi
SRCS = netapi.cpp
OBJS = $(SRCS:.cpp=.o)

INCDIR = -I include
LIBDIR = -L lib
LIBS = -lm

TMP = hello

RM:= rm
CXX:= g++
CXXFLAGS = -Wall -O2

$(TARGET): $(OBJS) hello.o 
	$(CXX) -o $@ $(OBJS) $(LIBDIR) $(LIBS)

$(OBJS): $(SRCS)
	$(CXX) $(INCDIR) -c $(SRCS)

clean:
	$(RM) $(OBJS) $(TARGET)


hello: hello.o
	$(CXX) -o hello hello.o
helc:
	$(RM) hello.o hello
