#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BILLION 1000000000L

// Función para medir el tiempo de acceso aleatorio
double medir_acceso_aleatorio(size_t tamano) {
    long* arreglo = malloc(tamano * sizeof(long));
    if (!arreglo) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    // Inicialización del arreglo
    for (size_t i = 0; i < tamano; i++) {
        arreglo[i] = i;
    }

    // Generar índices aleatorios
    size_t* indices = malloc(tamano * sizeof(size_t));
    for (size_t i = 0; i < tamano; i++) {
        indices[i] = i;
    }

    // Barajar los índices
    for (size_t i = 0; i < tamano; i++) {
        size_t j = rand() % tamano;
        size_t t = indices[j];
        indices[j] = indices[i];
        indices[i] = t;
    }

    struct timespec inicio, fin;
    clock_gettime(CLOCK_MONOTONIC, &inicio);

    // Acceder de manera aleatoria
    for (size_t i = 0; i < tamano; i++) {
        arreglo[indices[i]] += 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &fin);

    free(indices);
    free(arreglo);

    return (fin.tv_sec - inicio.tv_sec) + (fin.tv_nsec - inicio.tv_nsec) / (double)BILLION;
}

int main() {
    int muestras = 1000;
    // Tamaños de arreglo que probablemente entren en L1, L2, y L3
    size_t tamanos[] = {1024, 16384, 262144};

    for (int i = 0; i < 3; i++) {
        double tiempo_acumulado = 0.0;
        for (int j = 0; j < muestras; j++) {
            tiempo_acumulado += medir_acceso_aleatorio(tamanos[i]);
        }
        double tiempo_promedio = tiempo_acumulado / muestras;
        printf("Acceso secuencial - Tamaño: %lu, Tiempo promedio: %f microsegundos\n", tamanos[i], tiempo_promedio * 1e6);
    }

    return 0;
}