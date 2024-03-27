#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include "../auxFunctions.h"


#define CACHE_LINE_SIZE 64
#define MAX_ELEMENT_IN_LINE_SIZE_ARRAY (CACHE_LINE_SIZE/sizeof(int)) 
#define MAX_ELEMENT_IN_THREE_HALVES_LINE_SIZE_ARRAY (3*CACHE_LINE_SIZE/(2* sizeof(int)))
#define MAX_ELEMENTS_IN_ARRAYS (MAX_ELEMENT_IN_LINE_SIZE_ARRAY * MAX_ELEMENT_IN_THREE_HALVES_LINE_SIZE_ARRAY * 1000)

// Definición del struct con 16 enteros y un contador: total 96 bytes
struct LineSize {
    int array[MAX_ELEMENT_IN_LINE_SIZE_ARRAY]; // Arreglo con capacidad para 16 enteros               // Entero que representa la cantidad de elementos en el arreglo
};

// Definición del struct con 15 enteros y un contador: total 64 bytes
struct ThreeHalvesLineSize {
    int array[MAX_ELEMENT_IN_THREE_HALVES_LINE_SIZE_ARRAY];                
};


// Función para crear un LineSize
LineSize createLineSize() {
    LineSize res;
    for (int i = 0; i < MAX_ELEMENT_IN_LINE_SIZE_ARRAY; ++i) {
        res.array[i] = 1;
    }
    return res;
}

// Función para crear un ThreeHalvesLineSize
ThreeHalvesLineSize createThreeHalvesLineSize() {
    ThreeHalvesLineSize res;
    // Inicializar el arreglo con valores 1
    for (int i = 0; i < MAX_ELEMENT_IN_THREE_HALVES_LINE_SIZE_ARRAY; ++i) {
        res.array[i] = 1;
    }
    return res;
}

LineSize* createLineSizeArray(){
    LineSize* res = new LineSize[MAX_ELEMENTS_IN_ARRAYS / (3/2*CACHE_LINE_SIZE)];
    for(size_t i=0; i<MAX_ELEMENTS_IN_ARRAYS / (3/2*CACHE_LINE_SIZE); i++){
        res[i]=createLineSize();
    } 
    return res;
}

ThreeHalvesLineSize* createThreeHalvesLineSizeArray(){
    ThreeHalvesLineSize* res = new ThreeHalvesLineSize[MAX_ELEMENTS_IN_ARRAYS/CACHE_LINE_SIZE];
    for(size_t i=0; i<MAX_ELEMENTS_IN_ARRAYS/CACHE_LINE_SIZE; i++){
        res[i]=createThreeHalvesLineSize();
    } 
    return res;
}

int AccessLineSize(const LineSize& data){
    int res=0;
    for(int i=0; i<MAX_ELEMENT_IN_LINE_SIZE_ARRAY; i++){
        res=data.array[i];
    }
    return res;
}

int AccessThreeHalvesLineSizeArray(const ThreeHalvesLineSize& data){
    int res=0;
    for(int i=0; i<MAX_ELEMENT_IN_THREE_HALVES_LINE_SIZE_ARRAY; i++){
        res=data.array[i];
    }
    return res;
}

void AccessAlLLineSize(const LineSize* data){
    int res=0;
    
    for(int i=0; i<MAX_ELEMENT_IN_LINE_SIZE_ARRAY; i++){
        res= AccessLineSize(data[i]);
    }
}

void AccessAlLThreeHalvesLinesize(const ThreeHalvesLineSize* data){
    int res=0;
    for(int i=0; i<MAX_ELEMENT_IN_THREE_HALVES_LINE_SIZE_ARRAY; i++){
        res= AccessThreeHalvesLineSizeArray(data[i]);
    }
}


int main() {
    system("mkdir -p Ej1/results");

    // Abrir el archivo en el modo de escritura
    std::ofstream results("Ej1/results/1c");
    LineSize* arrayLineSize = createLineSizeArray();
    ThreeHalvesLineSize* arrayThreeHalvesLineSize = createThreeHalvesLineSizeArray();

    double time64 = Time([&]() { AccessAlLLineSize(arrayLineSize); });
    results << "Tiempo en arreglo tamano una linea: " << time64 << " s" << std::endl;

    double time96 = Time([&]() { AccessAlLThreeHalvesLinesize(arrayThreeHalvesLineSize); });
    results << "Tiempo en arreglo tamano 3/2 lineas: " << time96 << " s" << std::endl;

    results.close();
    delete[] arrayLineSize;
    delete[] arrayThreeHalvesLineSize;
    return 0;
}
