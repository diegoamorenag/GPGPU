# Nombre del archivo ejecutable
TARGET = median_filter

# Compilador
CXX = g++

# Opciones de compilación
CXXFLAGS = -Wall -std=c++11 -O2

# Bibliotecas
LIBS = -lm

# Archivos fuente
SOURCES = median_filter.cpp

# Archivos objeto (convertir archivos .cpp a .o)
OBJECTS = $(SOURCES:.cpp=.o)

# Regla por defecto
all: $(TARGET)

# Regla para compilar el programa
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Regla para compilar los archivos objeto
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

# Regla para limpiar archivos compilados
clean:
	rm -f $(OBJECTS) $(TARGET)

# Regla para correr el programa (opcional, para comodidad)
run: $(TARGET)
	./$(TARGET) input.png output.png 3