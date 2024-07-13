TARGET = median_filter
OUTPUT_FOLDER = evaluacion
CXX = g++

CXXFLAGS = -Wall -std=c++11 -O2

SOURCES = median_filter.cpp

OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJECTS) $(TARGET)

run: $(TARGET)
	./$(TARGET) input.pgm output.pgm 3