#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define BILLION 1000000000L

// Función para medir el tiempo de acceso con patrón aleatorio
double medir_acceso_aleatorio(size_t tamano, int iteraciones) {
    size_t bytes = tamano * sizeof(long);
    long* arreglo = (long*)malloc(bytes);
    if (!arreglo) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    // Inicialización del arreglo
    for (size_t i = 0; i < tamano; i++) {
        arreglo[i] = i;
    }

    // Generar índices aleatorios
    size_t* indices = (size_t*)malloc(tamano * sizeof(size_t));
    for (size_t i = 0; i < tamano; i++) {
        indices[i] = i;
    }
    
    // Barajar los índices para obtener un acceso aleatorio
    for (size_t i = 0; i < tamano; i++) {
        size_t j = i + rand() / (RAND_MAX / (tamano - i) + 1);
        size_t t = indices[j];
        indices[j] = indices[i];
        indices[i] = t;
    }

    struct timespec inicio, fin;
    clock_gettime(CLOCK_MONOTONIC, &inicio);

    // Acceder de manera aleatoria
    for (int j = 0; j < iteraciones; j++) {
        for (size_t i = 0; i < tamano; i++) {
            arreglo[indices[i]] += 1;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &fin);

    double tiempo_total = (fin.tv_sec - inicio.tv_sec) + (fin.tv_nsec - inicio.tv_nsec) / (double)BILLION;
    free(indices);
    free(arreglo);
    return tiempo_total / (tamano * iteraciones);
}

int main() {
    int iteraciones = 1000;
    srand(time(NULL)); // Semilla para la generación de números aleatorios

    // Tamaños de prueba para diferentes niveles de caché y memoria principal
    size_t tamanos[] = {4194304*4, 4194304*8, 4194304*16, 4194304*32};
    for (int i = 0; i < sizeof(tamanos) / sizeof(size_t); i++) {
        double tiempo = medir_acceso_aleatorio(tamanos[i], iteraciones);
        printf("Tamaño: %lu, Tiempo por acceso: %f ns\n", tamanos[i], tiempo * 1e9);
    }

    return 0;
}
