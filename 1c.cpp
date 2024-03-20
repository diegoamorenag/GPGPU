#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

#define MAX_ELEMENTS64 15 
#define MAX_ELEMENTS96 22 
#define MAX_ELEMENTS_IN_ARRAY_64 96*100
#define MAX_ELEMENTS_IN_ARRAY_96 64*100

// Definición del struct con 16 enteros y un contador: total 96 bytes
struct FixedArray_96Bytes {
    int array[MAX_ELEMENTS96]; // Arreglo con capacidad para 16 enteros
    int count;               // Entero que representa la cantidad de elementos en el arreglo
};

// Definición del struct con 15 enteros y un contador: total 64 bytes
struct FixedArray_64Bytes {
    int array[MAX_ELEMENTS64]; 
    int count;               
};


// Función para crear un FixedArray_64Bytes
FixedArray_64Bytes create64() {
    FixedArray_64Bytes res;
    res.count = MAX_ELEMENTS64;
    // Inicializar el arreglo con valores 1
    for (int i = 0; i < MAX_ELEMENTS64; ++i) {
        res.array[i] = 1;
    }
    return res;
}

// Función para crear un FixedArray_96Bytes
FixedArray_96Bytes create96() {
    FixedArray_96Bytes res;
    res.count = MAX_ELEMENTS96;
    // Inicializar el arreglo con valores 1
    for (int i = 0; i < MAX_ELEMENTS96; ++i) {
        res.array[i] = 1;
    }
    return res;
}

FixedArray_64Bytes* create64Array(){
    FixedArray_64Bytes* res = new FixedArray_64Bytes[MAX_ELEMENTS_IN_ARRAY_64];
    for(size_t i=0; i<MAX_ELEMENTS_IN_ARRAY_64; i++){
        res[i]=create64();
    } 
    return res;
}

FixedArray_96Bytes* create96Array(){
    FixedArray_96Bytes* res = new FixedArray_96Bytes[MAX_ELEMENTS_IN_ARRAY_96];
    for(size_t i=0; i<MAX_ELEMENTS_IN_ARRAY_96; i++){
        res[i]=create96();
    } 
    return res;
}

int Sum64(const FixedArray_64Bytes& data){
    int res=0;
    for(int i=0; i<data.count; i++){
        res+=data.array[i];
    }
    return res;
}

int Sum96(const FixedArray_96Bytes& data){
    int res=0;
    for(int i=0; i<data.count; i++){
        res+=data.array[i];
    }
    return res;
}


int main() {
    FixedArray_64Bytes* array64 = create64Array();
    FixedArray_96Bytes* array96 = create96Array();

    int res1 = 0;
    int res2=0;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<MAX_ELEMENTS_IN_ARRAY_64; i++){
        res1+= Sum64(array64[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double>  diff = end - start;
    std::cout << "Tiempo 64: " << diff.count() << " s" << std::endl;
    std::cout << "Res 64: " << res1 << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<MAX_ELEMENTS_IN_ARRAY_96; i++){
        res2+= Sum96(array96[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Tiempo 96: " << diff.count() << " s" << std::endl;
    std::cout << "Res 96: " << res2 << std::endl;



    return 0;
}
