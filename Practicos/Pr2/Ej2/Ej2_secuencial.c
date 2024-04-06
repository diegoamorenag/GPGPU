#include <stdio.h>
#include <stdlib.h>

// Asumiendo que tienes funciones como get_text_length y read_file como en tu ejemplo.
int get_text_length(const char * fname)
{
	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags

	size_t pos = ftell(f);    
	fseek(f, 0, SEEK_END);    
	size_t length = ftell(f); 
	fseek(f, pos, SEEK_SET);  

	fclose(f);

	return length;
}

void read_file(const char * fname, int* input)
{
	// printf("leyendo archivo %s\n", fname );

	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find %s file \n", fname);
		exit(1);
	}

	//fread(input, 1, N, f);
	int c; 
	while ((c = getc(f)) != EOF) {
		*(input++) = c;
	}

	fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Debe ingresar el nombre del archivo\n");
        return 1;
    }

    const char *fname = argv[1];
    int length = get_text_length(fname);
    int *text = (int *)malloc(length * sizeof(int));

    if (text == NULL) {
        fprintf(stderr, "Error al reservar memoria\n");
        return 1;
    }

    read_file(fname, text);

    // Arreglo para contar las ocurrencias de cada carácter ASCII.
    int occurrences[256] = {0};

    // Recorrer el texto y contar las ocurrencias de cada carácter.
    for (int i = 0; i < length; i++) {
        occurrences[text[i]]++;
    }

    for (int i = 0; i < 256; i++) {
        if (occurrences[i] > 0) {
            printf("Carácter '%c' (ASCII %d): %d veces\n", i, i, occurrences[i]);
        }
    }

    free(text);

    return 0;
}