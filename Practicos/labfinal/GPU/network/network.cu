#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;

struct PGMImage {
    int width;
    int height;
    int max_val;
    std::vector<unsigned char> data;
};

// Función para leer una imagen PGM
PGMImage readPGM(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se pudo abrir el archivo.");
    }

    PGMImage img;
    std::string line;
    std::getline(file, line);
    if (line != "P5" && line != "P2") {
        throw std::runtime_error("Formato de archivo no soportado. Solo se admite PGM binario (P5) o ASCII (P2).");
    }

    bool isBinary = (line == "P5");

    // Saltar comentarios
    while (std::getline(file, line)) {
        if (line[0] != '#') break;
    }

    std::istringstream iss(line);
    iss >> img.width >> img.height;
    file >> img.max_val;
    file.ignore(); // Saltar el carácter de nueva línea

    img.data.resize(img.width * img.height);
    if (isBinary) {
        file.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
    } else {
        for (int i = 0; i < img.width * img.height; ++i) {
            int pixel;
            file >> pixel;
            img.data[i] = static_cast<unsigned char>(pixel);
        }
    }

    return img;
}

// Función para escribir una imagen PGM
void writePGM(const std::string& filename, const PGMImage& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se pudo crear el archivo: " + filename);
    }

    file << "P5\n" << img.width << " " << img.height << "\n" << img.max_val << "\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
}

typedef unsigned char pixelvalue;
#define PIX_SORT(a,b) { if ((a)>(b)) PIX_SWAP((a),(b)); }
#define PIX_SWAP(a,b) { pixelvalue temp=(a);(a)=(b);(b)=temp; }

__device__ pixelvalue opt_med9(pixelvalue * p) {
    PIX_SORT(p[1], p[2]); PIX_SORT(p[4], p[5]); PIX_SORT(p[7], p[8]);
    PIX_SORT(p[0], p[1]); PIX_SORT(p[3], p[4]); PIX_SORT(p[6], p[7]);
    PIX_SORT(p[1], p[2]); PIX_SORT(p[4], p[5]); PIX_SORT(p[7], p[8]);
    PIX_SORT(p[0], p[3]); PIX_SORT(p[5], p[8]); PIX_SORT(p[4], p[7]);
    PIX_SORT(p[3], p[6]); PIX_SORT(p[1], p[4]); PIX_SORT(p[2], p[5]);
    PIX_SORT(p[4], p[7]); PIX_SORT(p[4], p[2]); PIX_SORT(p[6], p[4]);
    PIX_SORT(p[4], p[2]); 
    return(p[4]);
}

__device__ pixelvalue opt_med25(pixelvalue * p)
{
    PIX_SORT(p[0], p[1]); PIX_SORT(p[3], p[4]); PIX_SORT(p[2], p[4]);
    PIX_SORT(p[2], p[3]); PIX_SORT(p[6], p[7]); PIX_SORT(p[5], p[7]);
    PIX_SORT(p[5], p[6]); PIX_SORT(p[9], p[10]); PIX_SORT(p[8], p[10]);
    PIX_SORT(p[8], p[9]); PIX_SORT(p[12], p[13]); PIX_SORT(p[11], p[13]);
    PIX_SORT(p[11], p[12]); PIX_SORT(p[15], p[16]); PIX_SORT(p[14], p[16]);
    PIX_SORT(p[14], p[15]); PIX_SORT(p[18], p[19]); PIX_SORT(p[17], p[19]);
    PIX_SORT(p[17], p[18]); PIX_SORT(p[21], p[22]); PIX_SORT(p[20], p[22]);
    PIX_SORT(p[20], p[21]); PIX_SORT(p[23], p[24]); PIX_SORT(p[2], p[5]);
    PIX_SORT(p[3], p[6]); PIX_SORT(p[0], p[6]); PIX_SORT(p[0], p[3]);
    PIX_SORT(p[4], p[7]); PIX_SORT(p[1], p[7]); PIX_SORT(p[1], p[4]);
    PIX_SORT(p[11], p[14]); PIX_SORT(p[8], p[14]); PIX_SORT(p[8], p[11]);
    PIX_SORT(p[12], p[15]); PIX_SORT(p[9], p[15]); PIX_SORT(p[9], p[12]);
    PIX_SORT(p[13], p[16]); PIX_SORT(p[10], p[16]); PIX_SORT(p[10], p[13]);
    PIX_SORT(p[20], p[23]); PIX_SORT(p[17], p[23]); PIX_SORT(p[17], p[20]);
    PIX_SORT(p[21], p[24]); PIX_SORT(p[18], p[24]); PIX_SORT(p[18], p[21]);
    PIX_SORT(p[19], p[22]); PIX_SORT(p[8], p[17]); PIX_SORT(p[9], p[18]);
    PIX_SORT(p[0], p[18]); PIX_SORT(p[0], p[9]); PIX_SORT(p[10], p[19]);
    PIX_SORT(p[1], p[19]); PIX_SORT(p[1], p[10]); PIX_SORT(p[11], p[20]);
    PIX_SORT(p[2], p[20]); PIX_SORT(p[2], p[11]); PIX_SORT(p[12], p[21]);
    PIX_SORT(p[3], p[21]); PIX_SORT(p[3], p[12]); PIX_SORT(p[13], p[22]);
    PIX_SORT(p[4], p[22]); PIX_SORT(p[4], p[13]); PIX_SORT(p[14], p[23]);
    PIX_SORT(p[5], p[23]); PIX_SORT(p[5], p[14]); PIX_SORT(p[15], p[24]);
    PIX_SORT(p[6], p[24]); PIX_SORT(p[6], p[15]); PIX_SORT(p[7], p[16]);
    PIX_SORT(p[7], p[19]); PIX_SORT(p[13], p[21]); PIX_SORT(p[15], p[23]);
    PIX_SORT(p[7], p[13]); PIX_SORT(p[7], p[15]); PIX_SORT(p[1], p[9]);
    PIX_SORT(p[3], p[11]); PIX_SORT(p[5], p[17]); PIX_SORT(p[11], p[17]);
    PIX_SORT(p[9], p[17]); PIX_SORT(p[4], p[10]); PIX_SORT(p[6], p[12]);
    PIX_SORT(p[7], p[14]); PIX_SORT(p[4], p[6]); PIX_SORT(p[4], p[7]);
    PIX_SORT(p[12], p[14]); PIX_SORT(p[10], p[14]); PIX_SORT(p[6], p[7]);
    PIX_SORT(p[10], p[12]); PIX_SORT(p[6], p[10]); PIX_SORT(p[6], p[17]);
    PIX_SORT(p[12], p[17]); PIX_SORT(p[7], p[17]); PIX_SORT(p[7], p[10]);
    PIX_SORT(p[12], p[18]); PIX_SORT(p[7], p[12]); PIX_SORT(p[10], p[18]);
    PIX_SORT(p[12], p[20]); PIX_SORT(p[10], p[20]); PIX_SORT(p[10], p[12]);
    return (p[12]);
}

__device__ pixelvalue opt_med49(pixelvalue * p) {
    PIX_SORT(p[0], p[1]); PIX_SORT(p[0], p[2]); PIX_SORT(p[0], p[3]);
    PIX_SORT(p[0], p[4]); PIX_SORT(p[0], p[5]); PIX_SORT(p[0], p[6]);
    PIX_SORT(p[0], p[7]); PIX_SORT(p[0], p[8]); PIX_SORT(p[0], p[9]);
    PIX_SORT(p[0], p[10]); PIX_SORT(p[0], p[11]); PIX_SORT(p[0], p[12]);
    PIX_SORT(p[0], p[13]); PIX_SORT(p[0], p[14]); PIX_SORT(p[0], p[15]);
    PIX_SORT(p[0], p[16]); PIX_SORT(p[0], p[17]); PIX_SORT(p[0], p[18]);
    PIX_SORT(p[0], p[19]); PIX_SORT(p[0], p[20]); PIX_SORT(p[0], p[21]);
    PIX_SORT(p[0], p[22]); PIX_SORT(p[0], p[23]); PIX_SORT(p[0], p[24]);
    PIX_SORT(p[0], p[25]); PIX_SORT(p[0], p[26]); PIX_SORT(p[0], p[27]);
    PIX_SORT(p[0], p[28]); PIX_SORT(p[0], p[29]); PIX_SORT(p[0], p[30]);
    PIX_SORT(p[0], p[31]); PIX_SORT(p[0], p[32]); PIX_SORT(p[0], p[33]);
    PIX_SORT(p[0], p[34]); PIX_SORT(p[0], p[35]); PIX_SORT(p[0], p[36]);
    PIX_SORT(p[0], p[37]); PIX_SORT(p[0], p[38]); PIX_SORT(p[0], p[39]);
    PIX_SORT(p[0], p[40]); PIX_SORT(p[0], p[41]); PIX_SORT(p[0], p[42]);
    PIX_SORT(p[0], p[43]); PIX_SORT(p[0], p[44]); PIX_SORT(p[0], p[45]);
    PIX_SORT(p[0], p[46]); PIX_SORT(p[0], p[47]); PIX_SORT(p[0], p[48]);
    PIX_SORT(p[1], p[2]); PIX_SORT(p[1], p[3]); PIX_SORT(p[1], p[4]);
    PIX_SORT(p[1], p[5]); PIX_SORT(p[1], p[6]); PIX_SORT(p[1], p[7]);
    PIX_SORT(p[1], p[8]); PIX_SORT(p[1], p[9]); PIX_SORT(p[1], p[10]);
    PIX_SORT(p[1], p[11]); PIX_SORT(p[1], p[12]); PIX_SORT(p[1], p[13]);
    PIX_SORT(p[1], p[14]); PIX_SORT(p[1], p[15]); PIX_SORT(p[1], p[16]);
    PIX_SORT(p[1], p[17]); PIX_SORT(p[1], p[18]); PIX_SORT(p[1], p[19]);
    PIX_SORT(p[1], p[20]); PIX_SORT(p[1], p[21]); PIX_SORT(p[1], p[22]);
    PIX_SORT(p[1], p[23]); PIX_SORT(p[1], p[24]); PIX_SORT(p[1], p[25]);
    PIX_SORT(p[1], p[26]); PIX_SORT(p[1], p[27]); PIX_SORT(p[1], p[28]);
    PIX_SORT(p[1], p[29]); PIX_SORT(p[1], p[30]); PIX_SORT(p[1], p[31]);
    PIX_SORT(p[1], p[32]); PIX_SORT(p[1], p[33]); PIX_SORT(p[1], p[34]);
    PIX_SORT(p[1], p[35]); PIX_SORT(p[1], p[36]); PIX_SORT(p[1], p[37]);
    PIX_SORT(p[1], p[38]); PIX_SORT(p[1], p[39]); PIX_SORT(p[1], p[40]);
    PIX_SORT(p[1], p[41]); PIX_SORT(p[1], p[42]); PIX_SORT(p[1], p[43]);
    PIX_SORT(p[1], p[44]); PIX_SORT(p[1], p[45]); PIX_SORT(p[1], p[46]);
    PIX_SORT(p[1], p[47]); PIX_SORT(p[1], p[48]); PIX_SORT(p[2], p[3]);
    PIX_SORT(p[2], p[4]); PIX_SORT(p[2], p[5]); PIX_SORT(p[2], p[6]);
    PIX_SORT(p[2], p[7]); PIX_SORT(p[2], p[8]); PIX_SORT(p[2], p[9]);
    PIX_SORT(p[2], p[10]); PIX_SORT(p[2], p[11]); PIX_SORT(p[2], p[12]);
    PIX_SORT(p[2], p[13]); PIX_SORT(p[2], p[14]); PIX_SORT(p[2], p[15]);
    PIX_SORT(p[2], p[16]); PIX_SORT(p[2], p[17]); PIX_SORT(p[2], p[18]);
    PIX_SORT(p[2], p[19]); PIX_SORT(p[2], p[20]); PIX_SORT(p[2], p[21]);
    PIX_SORT(p[2], p[22]); PIX_SORT(p[2], p[23]); PIX_SORT(p[2], p[24]);
    PIX_SORT(p[2], p[25]); PIX_SORT(p[2], p[26]); PIX_SORT(p[2], p[27]);
    PIX_SORT(p[2], p[28]); PIX_SORT(p[2], p[29]); PIX_SORT(p[2], p[30]);
    PIX_SORT(p[2], p[31]); PIX_SORT(p[2], p[32]); PIX_SORT(p[2], p[33]);
    PIX_SORT(p[2], p[34]); PIX_SORT(p[2], p[35]); PIX_SORT(p[2], p[36]);
    PIX_SORT(p[2], p[37]); PIX_SORT(p[2], p[38]); PIX_SORT(p[2], p[39]);
    PIX_SORT(p[2], p[40]); PIX_SORT(p[2], p[41]); PIX_SORT(p[2], p[42]);
    PIX_SORT(p[2], p[43]); PIX_SORT(p[2], p[44]); PIX_SORT(p[2], p[45]);
    PIX_SORT(p[2], p[46]); PIX_SORT(p[2], p[47]); PIX_SORT(p[2], p[48]);
    PIX_SORT(p[3], p[4]); PIX_SORT(p[3], p[5]); PIX_SORT(p[3], p[6]);
    PIX_SORT(p[3], p[7]); PIX_SORT(p[3], p[8]); PIX_SORT(p[3], p[9]);
    PIX_SORT(p[3], p[10]); PIX_SORT(p[3], p[11]); PIX_SORT(p[3], p[12]);
    PIX_SORT(p[3], p[13]); PIX_SORT(p[3], p[14]); PIX_SORT(p[3], p[15]);
    PIX_SORT(p[3], p[16]); PIX_SORT(p[3], p[17]); PIX_SORT(p[3], p[18]);
    PIX_SORT(p[3], p[19]); PIX_SORT(p[3], p[20]); PIX_SORT(p[3], p[21]);
    PIX_SORT(p[3], p[22]); PIX_SORT(p[3], p[23]); PIX_SORT(p[3], p[24]);
    PIX_SORT(p[3], p[25]); PIX_SORT(p[3], p[26]); PIX_SORT(p[3], p[27]);
    PIX_SORT(p[3], p[28]); PIX_SORT(p[3], p[29]); PIX_SORT(p[3], p[30]);
    PIX_SORT(p[3], p[31]); PIX_SORT(p[3], p[32]); PIX_SORT(p[3], p[33]);
    PIX_SORT(p[3], p[34]); PIX_SORT(p[3], p[35]); PIX_SORT(p[3], p[36]);
    PIX_SORT(p[3], p[37]); PIX_SORT(p[3], p[38]); PIX_SORT(p[3], p[39]);
    PIX_SORT(p[3], p[40]); PIX_SORT(p[3], p[41]); PIX_SORT(p[3], p[42]);
    PIX_SORT(p[3], p[43]); PIX_SORT(p[3], p[44]); PIX_SORT(p[3], p[45]);
    PIX_SORT(p[3], p[46]); PIX_SORT(p[3], p[47]); PIX_SORT(p[3], p[48]);
    PIX_SORT(p[4], p[5]); PIX_SORT(p[4], p[6]); PIX_SORT(p[4], p[7]);
    PIX_SORT(p[4], p[8]); PIX_SORT(p[4], p[9]); PIX_SORT(p[4], p[10]);
    PIX_SORT(p[4], p[11]); PIX_SORT(p[4], p[12]); PIX_SORT(p[4], p[13]);
    PIX_SORT(p[4], p[14]); PIX_SORT(p[4], p[15]); PIX_SORT(p[4], p[16]);
    PIX_SORT(p[4], p[17]); PIX_SORT(p[4], p[18]); PIX_SORT(p[4], p[19]);
    PIX_SORT(p[4], p[20]); PIX_SORT(p[4], p[21]); PIX_SORT(p[4], p[22]);
    PIX_SORT(p[4], p[23]); PIX_SORT(p[4], p[24]); PIX_SORT(p[4], p[25]);
    PIX_SORT(p[4], p[26]); PIX_SORT(p[4], p[27]); PIX_SORT(p[4], p[28]);
    PIX_SORT(p[4], p[29]); PIX_SORT(p[4], p[30]); PIX_SORT(p[4], p[31]);
    PIX_SORT(p[4], p[32]); PIX_SORT(p[4], p[33]); PIX_SORT(p[4], p[34]);
    PIX_SORT(p[4], p[35]); PIX_SORT(p[4], p[36]); PIX_SORT(p[4], p[37]);
    PIX_SORT(p[4], p[38]); PIX_SORT(p[4], p[39]); PIX_SORT(p[4], p[40]);
    PIX_SORT(p[4], p[41]); PIX_SORT(p[4], p[42]); PIX_SORT(p[4], p[43]);
    PIX_SORT(p[4], p[44]); PIX_SORT(p[4], p[45]); PIX_SORT(p[4], p[46]);
    PIX_SORT(p[4], p[47]); PIX_SORT(p[4], p[48]); PIX_SORT(p[5], p[6]);
    PIX_SORT(p[5], p[7]); PIX_SORT(p[5], p[8]); PIX_SORT(p[5], p[9]);
    PIX_SORT(p[5], p[10]); PIX_SORT(p[5], p[11]); PIX_SORT(p[5], p[12]);
    PIX_SORT(p[5], p[13]); PIX_SORT(p[5], p[14]); PIX_SORT(p[5], p[15]);
    PIX_SORT(p[5], p[16]); PIX_SORT(p[5], p[17]); PIX_SORT(p[5], p[18]);
    PIX_SORT(p[5], p[19]); PIX_SORT(p[5], p[20]); PIX_SORT(p[5], p[21]);
    PIX_SORT(p[5], p[22]); PIX_SORT(p[5], p[23]); PIX_SORT(p[5], p[24]);
    PIX_SORT(p[5], p[25]); PIX_SORT(p[5], p[26]); PIX_SORT(p[5], p[27]);
    PIX_SORT(p[5], p[28]); PIX_SORT(p[5], p[29]); PIX_SORT(p[5], p[30]);
    PIX_SORT(p[5], p[31]); PIX_SORT(p[5], p[32]); PIX_SORT(p[5], p[33]);
    PIX_SORT(p[5], p[34]); PIX_SORT(p[5], p[35]); PIX_SORT(p[5], p[36]);
    PIX_SORT(p[5], p[37]); PIX_SORT(p[5], p[38]); PIX_SORT(p[5], p[39]);
    PIX_SORT(p[5], p[40]); PIX_SORT(p[5], p[41]); PIX_SORT(p[5], p[42]);
    PIX_SORT(p[5], p[43]); PIX_SORT(p[5], p[44]); PIX_SORT(p[5], p[45]);
    PIX_SORT(p[5], p[46]); PIX_SORT(p[5], p[47]); PIX_SORT(p[5], p[48]);
    PIX_SORT(p[6], p[7]); PIX_SORT(p[6], p[8]); PIX_SORT(p[6], p[9]);
    PIX_SORT(p[6], p[10]); PIX_SORT(p[6], p[11]); PIX_SORT(p[6], p[12]);
    PIX_SORT(p[6], p[13]); PIX_SORT(p[6], p[14]); PIX_SORT(p[6], p[15]);
    PIX_SORT(p[6], p[16]); PIX_SORT(p[6], p[17]); PIX_SORT(p[6], p[18]);
    PIX_SORT(p[6], p[19]); PIX_SORT(p[6], p[20]); PIX_SORT(p[6], p[21]);
    PIX_SORT(p[6], p[22]); PIX_SORT(p[6], p[23]); PIX_SORT(p[6], p[24]);
    PIX_SORT(p[6], p[25]); PIX_SORT(p[6], p[26]); PIX_SORT(p[6], p[27]);
    PIX_SORT(p[6], p[28]); PIX_SORT(p[6], p[29]); PIX_SORT(p[6], p[30]);
    PIX_SORT(p[6], p[31]); PIX_SORT(p[6], p[32]); PIX_SORT(p[6], p[33]);
    PIX_SORT(p[6], p[34]); PIX_SORT(p[6], p[35]); PIX_SORT(p[6], p[36]);
    PIX_SORT(p[6], p[37]); PIX_SORT(p[6], p[38]); PIX_SORT(p[6], p[39]);
    PIX_SORT(p[6], p[40]); PIX_SORT(p[6], p[41]); PIX_SORT(p[6], p[42]);
    PIX_SORT(p[6], p[43]); PIX_SORT(p[6], p[44]); PIX_SORT(p[6], p[45]);
    PIX_SORT(p[6], p[46]); PIX_SORT(p[6], p[47]); PIX_SORT(p[6], p[48]);
    PIX_SORT(p[7], p[8]); PIX_SORT(p[7], p[9]); PIX_SORT(p[7], p[10]);
    PIX_SORT(p[7], p[11]); PIX_SORT(p[7], p[12]); PIX_SORT(p[7], p[13]);
    PIX_SORT(p[7], p[14]); PIX_SORT(p[7], p[15]); PIX_SORT(p[7], p[16]);
    PIX_SORT(p[7], p[17]); PIX_SORT(p[7], p[18]); PIX_SORT(p[7], p[19]);
    PIX_SORT(p[7], p[20]); PIX_SORT(p[7], p[21]); PIX_SORT(p[7], p[22]);
    PIX_SORT(p[7], p[23]); PIX_SORT(p[7], p[24]); PIX_SORT(p[7], p[25]);
    PIX_SORT(p[7], p[26]); PIX_SORT(p[7], p[27]); PIX_SORT(p[7], p[28]);
    PIX_SORT(p[7], p[29]); PIX_SORT(p[7], p[30]); PIX_SORT(p[7], p[31]);
    PIX_SORT(p[7], p[32]); PIX_SORT(p[7], p[33]); PIX_SORT(p[7], p[34]);
    PIX_SORT(p[7], p[35]); PIX_SORT(p[7], p[36]); PIX_SORT(p[7], p[37]);
    PIX_SORT(p[7], p[38]); PIX_SORT(p[7], p[39]); PIX_SORT(p[7], p[40]);
    PIX_SORT(p[7], p[41]); PIX_SORT(p[7], p[42]); PIX_SORT(p[7], p[43]);
    PIX_SORT(p[7], p[44]); PIX_SORT(p[7], p[45]); PIX_SORT(p[7], p[46]);
    PIX_SORT(p[7], p[47]); PIX_SORT(p[7], p[48]); PIX_SORT(p[8], p[9]);
    PIX_SORT(p[8], p[10]); PIX_SORT(p[8], p[11]); PIX_SORT(p[8], p[12]);
    PIX_SORT(p[8], p[13]); PIX_SORT(p[8], p[14]); PIX_SORT(p[8], p[15]);
    PIX_SORT(p[8], p[16]); PIX_SORT(p[8], p[17]); PIX_SORT(p[8], p[18]);
    PIX_SORT(p[8], p[19]); PIX_SORT(p[8], p[20]); PIX_SORT(p[8], p[21]);
    PIX_SORT(p[8], p[22]); PIX_SORT(p[8], p[23]); PIX_SORT(p[8], p[24]);
    PIX_SORT(p[8], p[25]); PIX_SORT(p[8], p[26]); PIX_SORT(p[8], p[27]);
    PIX_SORT(p[8], p[28]); PIX_SORT(p[8], p[29]); PIX_SORT(p[8], p[30]);
    PIX_SORT(p[8], p[31]); PIX_SORT(p[8], p[32]); PIX_SORT(p[8], p[33]);
    PIX_SORT(p[8], p[34]); PIX_SORT(p[8], p[35]); PIX_SORT(p[8], p[36]);
    PIX_SORT(p[8], p[37]); PIX_SORT(p[8], p[38]); PIX_SORT(p[8], p[39]);
    PIX_SORT(p[8], p[40]); PIX_SORT(p[8], p[41]); PIX_SORT(p[8], p[42]);
    PIX_SORT(p[8], p[43]); PIX_SORT(p[8], p[44]); PIX_SORT(p[8], p[45]);
    PIX_SORT(p[8], p[46]); PIX_SORT(p[8], p[47]); PIX_SORT(p[8], p[48]);
    PIX_SORT(p[9], p[10]); PIX_SORT(p[9], p[11]); PIX_SORT(p[9], p[12]);
    PIX_SORT(p[9], p[13]); PIX_SORT(p[9], p[14]); PIX_SORT(p[9], p[15]);
    PIX_SORT(p[9], p[16]); PIX_SORT(p[9], p[17]); PIX_SORT(p[9], p[18]);
    PIX_SORT(p[9], p[19]); PIX_SORT(p[9], p[20]); PIX_SORT(p[9], p[21]);
    PIX_SORT(p[9], p[22]); PIX_SORT(p[9], p[23]); PIX_SORT(p[9], p[24]);
    PIX_SORT(p[9], p[25]); PIX_SORT(p[9], p[26]); PIX_SORT(p[9], p[27]);
    PIX_SORT(p[9], p[28]); PIX_SORT(p[9], p[29]); PIX_SORT(p[9], p[30]);
    PIX_SORT(p[9], p[31]); PIX_SORT(p[9], p[32]); PIX_SORT(p[9], p[33]);
    PIX_SORT(p[9], p[34]); PIX_SORT(p[9], p[35]); PIX_SORT(p[9], p[36]);
    PIX_SORT(p[9], p[37]); PIX_SORT(p[9], p[38]); PIX_SORT(p[9], p[39]);
    PIX_SORT(p[9], p[40]); PIX_SORT(p[9], p[41]); PIX_SORT(p[9], p[42]);
    PIX_SORT(p[9], p[43]); PIX_SORT(p[9], p[44]); PIX_SORT(p[9], p[45]);
    PIX_SORT(p[9], p[46]); PIX_SORT(p[9], p[47]); PIX_SORT(p[9], p[48]);
    PIX_SORT(p[10], p[11]); PIX_SORT(p[10], p[12]); PIX_SORT(p[10], p[13]);
    PIX_SORT(p[10], p[14]); PIX_SORT(p[10], p[15]); PIX_SORT(p[10], p[16]);
    PIX_SORT(p[10], p[17]); PIX_SORT(p[10], p[18]); PIX_SORT(p[10], p[19]);
    PIX_SORT(p[10], p[20]); PIX_SORT(p[10], p[21]); PIX_SORT(p[10], p[22]);
    PIX_SORT(p[10], p[23]); PIX_SORT(p[10], p[24]); PIX_SORT(p[10], p[25]);
    PIX_SORT(p[10], p[26]); PIX_SORT(p[10], p[27]); PIX_SORT(p[10], p[28]);
    PIX_SORT(p[10], p[29]); PIX_SORT(p[10], p[30]); PIX_SORT(p[10], p[31]);
    PIX_SORT(p[10], p[32]); PIX_SORT(p[10], p[33]); PIX_SORT(p[10], p[34]);
    PIX_SORT(p[10], p[35]); PIX_SORT(p[10], p[36]); PIX_SORT(p[10], p[37]);
    PIX_SORT(p[10], p[38]); PIX_SORT(p[10], p[39]); PIX_SORT(p[10], p[40]);
    PIX_SORT(p[10], p[41]); PIX_SORT(p[10], p[42]); PIX_SORT(p[10], p[43]);
    PIX_SORT(p[10], p[44]); PIX_SORT(p[10], p[45]); PIX_SORT(p[10], p[46]);
    PIX_SORT(p[10], p[47]); PIX_SORT(p[10], p[48]); PIX_SORT(p[11], p[12]);
    PIX_SORT(p[11], p[13]); PIX_SORT(p[11], p[14]); PIX_SORT(p[11], p[15]);
    PIX_SORT(p[11], p[16]); PIX_SORT(p[11], p[17]); PIX_SORT(p[11], p[18]);
    PIX_SORT(p[11], p[19]); PIX_SORT(p[11], p[20]); PIX_SORT(p[11], p[21]);
    PIX_SORT(p[11], p[22]); PIX_SORT(p[11], p[23]); PIX_SORT(p[11], p[24]);
    PIX_SORT(p[11], p[25]); PIX_SORT(p[11], p[26]); PIX_SORT(p[11], p[27]);
    PIX_SORT(p[11], p[28]); PIX_SORT(p[11], p[29]); PIX_SORT(p[11], p[30]);
    PIX_SORT(p[11], p[31]); PIX_SORT(p[11], p[32]); PIX_SORT(p[11], p[33]);
    PIX_SORT(p[11], p[34]); PIX_SORT(p[11], p[35]); PIX_SORT(p[11], p[36]);
    PIX_SORT(p[11], p[37]); PIX_SORT(p[11], p[38]); PIX_SORT(p[11], p[39]);
    PIX_SORT(p[11], p[40]); PIX_SORT(p[11], p[41]); PIX_SORT(p[11], p[42]);
    PIX_SORT(p[11], p[43]); PIX_SORT(p[11], p[44]); PIX_SORT(p[11], p[45]);
    PIX_SORT(p[11], p[46]); PIX_SORT(p[11], p[47]); PIX_SORT(p[11], p[48]);
    PIX_SORT(p[12], p[13]); PIX_SORT(p[12], p[14]); PIX_SORT(p[12], p[15]);
    PIX_SORT(p[12], p[16]); PIX_SORT(p[12], p[17]); PIX_SORT(p[12], p[18]);
    PIX_SORT(p[12], p[19]); PIX_SORT(p[12], p[20]); PIX_SORT(p[12], p[21]);
    PIX_SORT(p[12], p[22]); PIX_SORT(p[12], p[23]); PIX_SORT(p[12], p[24]);
    PIX_SORT(p[12], p[25]); PIX_SORT(p[12], p[26]); PIX_SORT(p[12], p[27]);
    PIX_SORT(p[12], p[28]); PIX_SORT(p[12], p[29]); PIX_SORT(p[12], p[30]);
    PIX_SORT(p[12], p[31]); PIX_SORT(p[12], p[32]); PIX_SORT(p[12], p[33]);
    PIX_SORT(p[12], p[34]); PIX_SORT(p[12], p[35]); PIX_SORT(p[12], p[36]);
    PIX_SORT(p[12], p[37]); PIX_SORT(p[12], p[38]); PIX_SORT(p[12], p[39]);
    PIX_SORT(p[12], p[40]); PIX_SORT(p[12], p[41]); PIX_SORT(p[12], p[42]);
    PIX_SORT(p[12], p[43]); PIX_SORT(p[12], p[44]); PIX_SORT(p[12], p[45]);
    PIX_SORT(p[12], p[46]); PIX_SORT(p[12], p[47]); PIX_SORT(p[12], p[48]);
    PIX_SORT(p[13], p[14]); PIX_SORT(p[13], p[15]); PIX_SORT(p[13], p[16]);
    PIX_SORT(p[13], p[17]); PIX_SORT(p[13], p[18]); PIX_SORT(p[13], p[19]);
    PIX_SORT(p[13], p[20]); PIX_SORT(p[13], p[21]); PIX_SORT(p[13], p[22]);
    PIX_SORT(p[13], p[23]); PIX_SORT(p[13], p[24]); PIX_SORT(p[13], p[25]);
    PIX_SORT(p[13], p[26]); PIX_SORT(p[13], p[27]); PIX_SORT(p[13], p[28]);
    PIX_SORT(p[13], p[29]); PIX_SORT(p[13], p[30]); PIX_SORT(p[13], p[31]);
    PIX_SORT(p[13], p[32]); PIX_SORT(p[13], p[33]); PIX_SORT(p[13], p[34]);
    PIX_SORT(p[13], p[35]); PIX_SORT(p[13], p[36]); PIX_SORT(p[13], p[37]);
    PIX_SORT(p[13], p[38]); PIX_SORT(p[13], p[39]); PIX_SORT(p[13], p[40]);
    PIX_SORT(p[13], p[41]); PIX_SORT(p[13], p[42]); PIX_SORT(p[13], p[43]);
    PIX_SORT(p[13], p[44]); PIX_SORT(p[13], p[45]); PIX_SORT(p[13], p[46]);
    PIX_SORT(p[13], p[47]); PIX_SORT(p[13], p[48]); PIX_SORT(p[14], p[15]);
    PIX_SORT(p[14], p[16]); PIX_SORT(p[14], p[17]); PIX_SORT(p[14], p[18]);
    PIX_SORT(p[14], p[19]); PIX_SORT(p[14], p[20]); PIX_SORT(p[14], p[21]);
    PIX_SORT(p[14], p[22]); PIX_SORT(p[14], p[23]); PIX_SORT(p[14], p[24]);
    PIX_SORT(p[14], p[25]); PIX_SORT(p[14], p[26]); PIX_SORT(p[14], p[27]);
    PIX_SORT(p[14], p[28]); PIX_SORT(p[14], p[29]); PIX_SORT(p[14], p[30]);
    PIX_SORT(p[14], p[31]); PIX_SORT(p[14], p[32]); PIX_SORT(p[14], p[33]);
    PIX_SORT(p[14], p[34]); PIX_SORT(p[14], p[35]); PIX_SORT(p[14], p[36]);
    PIX_SORT(p[14], p[37]); PIX_SORT(p[14], p[38]); PIX_SORT(p[14], p[39]);
    PIX_SORT(p[14], p[40]); PIX_SORT(p[14], p[41]); PIX_SORT(p[14], p[42]);
    PIX_SORT(p[14], p[43]); PIX_SORT(p[14], p[44]); PIX_SORT(p[14], p[45]);
    PIX_SORT(p[14], p[46]); PIX_SORT(p[14], p[47]); PIX_SORT(p[14], p[48]);
    PIX_SORT(p[15], p[16]); PIX_SORT(p[15], p[17]); PIX_SORT(p[15], p[18]);
    PIX_SORT(p[15], p[19]); PIX_SORT(p[15], p[20]); PIX_SORT(p[15], p[21]);
    PIX_SORT(p[15], p[22]); PIX_SORT(p[15], p[23]); PIX_SORT(p[15], p[24]);
    PIX_SORT(p[15], p[25]); PIX_SORT(p[15], p[26]); PIX_SORT(p[15], p[27]);
    PIX_SORT(p[15], p[28]); PIX_SORT(p[15], p[29]); PIX_SORT(p[15], p[30]);
    PIX_SORT(p[15], p[31]); PIX_SORT(p[15], p[32]); PIX_SORT(p[15], p[33]);
    PIX_SORT(p[15], p[34]); PIX_SORT(p[15], p[35]); PIX_SORT(p[15], p[36]);
    PIX_SORT(p[15], p[37]); PIX_SORT(p[15], p[38]); PIX_SORT(p[15], p[39]);
    PIX_SORT(p[15], p[40]); PIX_SORT(p[15], p[41]); PIX_SORT(p[15], p[42]);
    PIX_SORT(p[15], p[43]); PIX_SORT(p[15], p[44]); PIX_SORT(p[15], p[45]);
    PIX_SORT(p[15], p[46]); PIX_SORT(p[15], p[47]); PIX_SORT(p[15], p[48]);
    PIX_SORT(p[16], p[17]); PIX_SORT(p[16], p[18]); PIX_SORT(p[16], p[19]);
    PIX_SORT(p[16], p[20]); PIX_SORT(p[16], p[21]); PIX_SORT(p[16], p[22]);
    PIX_SORT(p[16], p[23]); PIX_SORT(p[16], p[24]); PIX_SORT(p[16], p[25]);
    PIX_SORT(p[16], p[26]); PIX_SORT(p[16], p[27]); PIX_SORT(p[16], p[28]);
    PIX_SORT(p[16], p[29]); PIX_SORT(p[16], p[30]); PIX_SORT(p[16], p[31]);
    PIX_SORT(p[16], p[32]); PIX_SORT(p[16], p[33]); PIX_SORT(p[16], p[34]);
    PIX_SORT(p[16], p[35]); PIX_SORT(p[16], p[36]); PIX_SORT(p[16], p[37]);
    PIX_SORT(p[16], p[38]); PIX_SORT(p[16], p[39]); PIX_SORT(p[16], p[40]);
    PIX_SORT(p[16], p[41]); PIX_SORT(p[16], p[42]); PIX_SORT(p[16], p[43]);
    PIX_SORT(p[16], p[44]); PIX_SORT(p[16], p[45]); PIX_SORT(p[16], p[46]);
    PIX_SORT(p[16], p[47]); PIX_SORT(p[16], p[48]); PIX_SORT(p[17], p[18]);
    PIX_SORT(p[17], p[19]); PIX_SORT(p[17], p[20]); PIX_SORT(p[17], p[21]);
    PIX_SORT(p[17], p[22]); PIX_SORT(p[17], p[23]); PIX_SORT(p[17], p[24]);
    PIX_SORT(p[17], p[25]); PIX_SORT(p[17], p[26]); PIX_SORT(p[17], p[27]);
    PIX_SORT(p[17], p[28]); PIX_SORT(p[17], p[29]); PIX_SORT(p[17], p[30]);
    PIX_SORT(p[17], p[31]); PIX_SORT(p[17], p[32]); PIX_SORT(p[17], p[33]);
    PIX_SORT(p[17], p[34]); PIX_SORT(p[17], p[35]); PIX_SORT(p[17], p[36]);
    PIX_SORT(p[17], p[37]); PIX_SORT(p[17], p[38]); PIX_SORT(p[17], p[39]);
    PIX_SORT(p[17], p[40]); PIX_SORT(p[17], p[41]); PIX_SORT(p[17], p[42]);
    PIX_SORT(p[17], p[43]); PIX_SORT(p[17], p[44]); PIX_SORT(p[17], p[45]);
    PIX_SORT(p[17], p[46]); PIX_SORT(p[17], p[47]); PIX_SORT(p[17], p[48]);
    PIX_SORT(p[18], p[19]); PIX_SORT(p[18], p[20]); PIX_SORT(p[18], p[21]);
    PIX_SORT(p[18], p[22]); PIX_SORT(p[18], p[23]); PIX_SORT(p[18], p[24]);
    PIX_SORT(p[18], p[25]); PIX_SORT(p[18], p[26]); PIX_SORT(p[18], p[27]);
    PIX_SORT(p[18], p[28]); PIX_SORT(p[18], p[29]); PIX_SORT(p[18], p[30]);
    PIX_SORT(p[18], p[31]); PIX_SORT(p[18], p[32]); PIX_SORT(p[18], p[33]);
    PIX_SORT(p[18], p[34]); PIX_SORT(p[18], p[35]); PIX_SORT(p[18], p[36]);
    PIX_SORT(p[18], p[37]); PIX_SORT(p[18], p[38]); PIX_SORT(p[18], p[39]);
    PIX_SORT(p[18], p[40]); PIX_SORT(p[18], p[41]); PIX_SORT(p[18], p[42]);
    PIX_SORT(p[18], p[43]); PIX_SORT(p[18], p[44]); PIX_SORT(p[18], p[45]);
    PIX_SORT(p[18], p[46]); PIX_SORT(p[18], p[47]); PIX_SORT(p[18], p[48]);
    PIX_SORT(p[19], p[20]); PIX_SORT(p[19], p[21]); PIX_SORT(p[19], p[22]);
    PIX_SORT(p[19], p[23]); PIX_SORT(p[19], p[24]); PIX_SORT(p[19], p[25]);
    PIX_SORT(p[19], p[26]); PIX_SORT(p[19], p[27]); PIX_SORT(p[19], p[28]);
    PIX_SORT(p[19], p[29]); PIX_SORT(p[19], p[30]); PIX_SORT(p[19], p[31]);
    PIX_SORT(p[19], p[32]); PIX_SORT(p[19], p[33]); PIX_SORT(p[19], p[34]);
    PIX_SORT(p[19], p[35]); PIX_SORT(p[19], p[36]); PIX_SORT(p[19], p[37]);
    PIX_SORT(p[19], p[38]); PIX_SORT(p[19], p[39]); PIX_SORT(p[19], p[40]);
    PIX_SORT(p[19], p[41]); PIX_SORT(p[19], p[42]); PIX_SORT(p[19], p[43]);
    PIX_SORT(p[19], p[44]); PIX_SORT(p[19], p[45]); PIX_SORT(p[19], p[46]);
    PIX_SORT(p[19], p[47]); PIX_SORT(p[19], p[48]); PIX_SORT(p[20], p[21]);
    PIX_SORT(p[20], p[22]); PIX_SORT(p[20], p[23]); PIX_SORT(p[20], p[24]);
    PIX_SORT(p[20], p[25]); PIX_SORT(p[20], p[26]); PIX_SORT(p[20], p[27]);
    PIX_SORT(p[20], p[28]); PIX_SORT(p[20], p[29]); PIX_SORT(p[20], p[30]);
    PIX_SORT(p[20], p[31]); PIX_SORT(p[20], p[32]); PIX_SORT(p[20], p[33]);
    PIX_SORT(p[20], p[34]); PIX_SORT(p[20], p[35]); PIX_SORT(p[20], p[36]);
    PIX_SORT(p[20], p[37]); PIX_SORT(p[20], p[38]); PIX_SORT(p[20], p[39]);
    PIX_SORT(p[20], p[40]); PIX_SORT(p[20], p[41]); PIX_SORT(p[20], p[42]);
    PIX_SORT(p[20], p[43]); PIX_SORT(p[20], p[44]); PIX_SORT(p[20], p[45]);
    PIX_SORT(p[20], p[46]); PIX_SORT(p[20], p[47]); PIX_SORT(p[20], p[48]);
    PIX_SORT(p[21], p[22]); PIX_SORT(p[21], p[23]); PIX_SORT(p[21], p[24]);
    PIX_SORT(p[21], p[25]); PIX_SORT(p[21], p[26]); PIX_SORT(p[21], p[27]);
    PIX_SORT(p[21], p[28]); PIX_SORT(p[21], p[29]); PIX_SORT(p[21], p[30]);
    PIX_SORT(p[21], p[31]); PIX_SORT(p[21], p[32]); PIX_SORT(p[21], p[33]);
    PIX_SORT(p[21], p[34]); PIX_SORT(p[21], p[35]); PIX_SORT(p[21], p[36]);
    PIX_SORT(p[21], p[37]); PIX_SORT(p[21], p[38]); PIX_SORT(p[21], p[39]);
    PIX_SORT(p[21], p[40]); PIX_SORT(p[21], p[41]); PIX_SORT(p[21], p[42]);
    PIX_SORT(p[21], p[43]); PIX_SORT(p[21], p[44]); PIX_SORT(p[21], p[45]);
    PIX_SORT(p[21], p[46]); PIX_SORT(p[21], p[47]); PIX_SORT(p[21], p[48]);
    PIX_SORT(p[22], p[23]); PIX_SORT(p[22], p[24]); PIX_SORT(p[22], p[25]);
    PIX_SORT(p[22], p[26]); PIX_SORT(p[22], p[27]); PIX_SORT(p[22], p[28]);
    PIX_SORT(p[22], p[29]); PIX_SORT(p[22], p[30]); PIX_SORT(p[22], p[31]);
    PIX_SORT(p[22], p[32]); PIX_SORT(p[22], p[33]); PIX_SORT(p[22], p[34]);
    PIX_SORT(p[22], p[35]); PIX_SORT(p[22], p[36]); PIX_SORT(p[22], p[37]);
    PIX_SORT(p[22], p[38]); PIX_SORT(p[22], p[39]); PIX_SORT(p[22], p[40]);
    PIX_SORT(p[22], p[41]); PIX_SORT(p[22], p[42]); PIX_SORT(p[22], p[43]);
    PIX_SORT(p[22], p[44]); PIX_SORT(p[22], p[45]); PIX_SORT(p[22], p[46]);
    PIX_SORT(p[22], p[47]); PIX_SORT(p[22], p[48]); PIX_SORT(p[23], p[24]);
    PIX_SORT(p[23], p[25]); PIX_SORT(p[23], p[26]); PIX_SORT(p[23], p[27]);
    PIX_SORT(p[23], p[28]); PIX_SORT(p[23], p[29]); PIX_SORT(p[23], p[30]);
    PIX_SORT(p[23], p[31]); PIX_SORT(p[23], p[32]); PIX_SORT(p[23], p[33]);
    PIX_SORT(p[23], p[34]); PIX_SORT(p[23], p[35]); PIX_SORT(p[23], p[36]);
    PIX_SORT(p[23], p[37]); PIX_SORT(p[23], p[38]); PIX_SORT(p[23], p[39]);
    PIX_SORT(p[23], p[40]); PIX_SORT(p[23], p[41]); PIX_SORT(p[23], p[42]);
    PIX_SORT(p[23], p[43]); PIX_SORT(p[23], p[44]); PIX_SORT(p[23], p[45]);
    PIX_SORT(p[23], p[46]); PIX_SORT(p[23], p[47]); PIX_SORT(p[23], p[48]);
    PIX_SORT(p[24], p[25]); PIX_SORT(p[24], p[26]); PIX_SORT(p[24], p[27]);
    PIX_SORT(p[24], p[28]); PIX_SORT(p[24], p[29]); PIX_SORT(p[24], p[30]);
    PIX_SORT(p[24], p[31]); PIX_SORT(p[24], p[32]); PIX_SORT(p[24], p[33]);
    PIX_SORT(p[24], p[34]); PIX_SORT(p[24], p[35]); PIX_SORT(p[24], p[36]);
    PIX_SORT(p[24], p[37]); PIX_SORT(p[24], p[38]); PIX_SORT(p[24], p[39]);
    PIX_SORT(p[24], p[40]); PIX_SORT(p[24], p[41]); PIX_SORT(p[24], p[42]);
    PIX_SORT(p[24], p[43]); PIX_SORT(p[24], p[44]); PIX_SORT(p[24], p[45]);
    PIX_SORT(p[24], p[46]); PIX_SORT(p[24], p[47]); PIX_SORT(p[24], p[48]);
    PIX_SORT(p[25], p[26]); PIX_SORT(p[25], p[27]); PIX_SORT(p[25], p[28]);
    PIX_SORT(p[25], p[29]); PIX_SORT(p[25], p[30]); PIX_SORT(p[25], p[31]);
    PIX_SORT(p[25], p[32]); PIX_SORT(p[25], p[33]); PIX_SORT(p[25], p[34]);
    PIX_SORT(p[25], p[35]); PIX_SORT(p[25], p[36]); PIX_SORT(p[25], p[37]);
    PIX_SORT(p[25], p[38]); PIX_SORT(p[25], p[39]); PIX_SORT(p[25], p[40]);
    PIX_SORT(p[25], p[41]); PIX_SORT(p[25], p[42]); PIX_SORT(p[25], p[43]);
    PIX_SORT(p[25], p[44]); PIX_SORT(p[25], p[45]); PIX_SORT(p[25], p[46]);
    PIX_SORT(p[25], p[47]); PIX_SORT(p[25], p[48]); PIX_SORT(p[26], p[27]);
    PIX_SORT(p[26], p[28]); PIX_SORT(p[26], p[29]); PIX_SORT(p[26], p[30]);
    PIX_SORT(p[26], p[31]); PIX_SORT(p[26], p[32]); PIX_SORT(p[26], p[33]);
    PIX_SORT(p[26], p[34]); PIX_SORT(p[26], p[35]); PIX_SORT(p[26], p[36]);
    PIX_SORT(p[26], p[37]); PIX_SORT(p[26], p[38]); PIX_SORT(p[26], p[39]);
    PIX_SORT(p[26], p[40]); PIX_SORT(p[26], p[41]); PIX_SORT(p[26], p[42]);
    PIX_SORT(p[26], p[43]); PIX_SORT(p[26], p[44]); PIX_SORT(p[26], p[45]);
    PIX_SORT(p[26], p[46]); PIX_SORT(p[26], p[47]); PIX_SORT(p[26], p[48]);
    PIX_SORT(p[27], p[28]); PIX_SORT(p[27], p[29]); PIX_SORT(p[27], p[30]);
    PIX_SORT(p[27], p[31]); PIX_SORT(p[27], p[32]); PIX_SORT(p[27], p[33]);
    PIX_SORT(p[27], p[34]); PIX_SORT(p[27], p[35]); PIX_SORT(p[27], p[36]);
    PIX_SORT(p[27], p[37]); PIX_SORT(p[27], p[38]); PIX_SORT(p[27], p[39]);
    PIX_SORT(p[27], p[40]); PIX_SORT(p[27], p[41]); PIX_SORT(p[27], p[42]);
    PIX_SORT(p[27], p[43]); PIX_SORT(p[27], p[44]); PIX_SORT(p[27], p[45]);
    PIX_SORT(p[27], p[46]); PIX_SORT(p[27], p[47]); PIX_SORT(p[27], p[48]);
    PIX_SORT(p[28], p[29]); PIX_SORT(p[28], p[30]); PIX_SORT(p[28], p[31]);
    PIX_SORT(p[28], p[32]); PIX_SORT(p[28], p[33]); PIX_SORT(p[28], p[34]);
    PIX_SORT(p[28], p[35]); PIX_SORT(p[28], p[36]); PIX_SORT(p[28], p[37]);
    PIX_SORT(p[28], p[38]); PIX_SORT(p[28], p[39]); PIX_SORT(p[28], p[40]);
    PIX_SORT(p[28], p[41]); PIX_SORT(p[28], p[42]); PIX_SORT(p[28], p[43]);
    PIX_SORT(p[28], p[44]); PIX_SORT(p[28], p[45]); PIX_SORT(p[28], p[46]);
    PIX_SORT(p[28], p[47]); PIX_SORT(p[28], p[48]); PIX_SORT(p[29], p[30]);
    PIX_SORT(p[29], p[31]); PIX_SORT(p[29], p[32]); PIX_SORT(p[29], p[33]);
    PIX_SORT(p[29], p[34]); PIX_SORT(p[29], p[35]); PIX_SORT(p[29], p[36]);
    PIX_SORT(p[29], p[37]); PIX_SORT(p[29], p[38]); PIX_SORT(p[29], p[39]);
    PIX_SORT(p[29], p[40]); PIX_SORT(p[29], p[41]); PIX_SORT(p[29], p[42]);
    PIX_SORT(p[29], p[43]); PIX_SORT(p[29], p[44]); PIX_SORT(p[29], p[45]);
    PIX_SORT(p[29], p[46]); PIX_SORT(p[29], p[47]); PIX_SORT(p[29], p[48]);
    PIX_SORT(p[30], p[31]); PIX_SORT(p[30], p[32]); PIX_SORT(p[30], p[33]);
    PIX_SORT(p[30], p[34]); PIX_SORT(p[30], p[35]); PIX_SORT(p[30], p[36]);
    PIX_SORT(p[30], p[37]); PIX_SORT(p[30], p[38]); PIX_SORT(p[30], p[39]);
    PIX_SORT(p[30], p[40]); PIX_SORT(p[30], p[41]); PIX_SORT(p[30], p[42]);
    PIX_SORT(p[30], p[43]); PIX_SORT(p[30], p[44]); PIX_SORT(p[30], p[45]);
    PIX_SORT(p[30], p[46]); PIX_SORT(p[30], p[47]); PIX_SORT(p[30], p[48]);
    PIX_SORT(p[31], p[32]); PIX_SORT(p[31], p[33]); PIX_SORT(p[31], p[34]);
    PIX_SORT(p[31], p[35]); PIX_SORT(p[31], p[36]); PIX_SORT(p[31], p[37]);
    PIX_SORT(p[31], p[38]); PIX_SORT(p[31], p[39]); PIX_SORT(p[31], p[40]);
    PIX_SORT(p[31], p[41]); PIX_SORT(p[31], p[42]); PIX_SORT(p[31], p[43]);
    PIX_SORT(p[31], p[44]); PIX_SORT(p[31], p[45]); PIX_SORT(p[31], p[46]);
    PIX_SORT(p[31], p[47]); PIX_SORT(p[31], p[48]); PIX_SORT(p[32], p[33]);
    PIX_SORT(p[32], p[34]); PIX_SORT(p[32], p[35]); PIX_SORT(p[32], p[36]);
    PIX_SORT(p[32], p[37]); PIX_SORT(p[32], p[38]); PIX_SORT(p[32], p[39]);
    PIX_SORT(p[32], p[40]); PIX_SORT(p[32], p[41]); PIX_SORT(p[32], p[42]);
    PIX_SORT(p[32], p[43]); PIX_SORT(p[32], p[44]); PIX_SORT(p[32], p[45]);
    PIX_SORT(p[32], p[46]); PIX_SORT(p[32], p[47]); PIX_SORT(p[32], p[48]);
    PIX_SORT(p[33], p[34]); PIX_SORT(p[33], p[35]); PIX_SORT(p[33], p[36]);
    PIX_SORT(p[33], p[37]); PIX_SORT(p[33], p[38]); PIX_SORT(p[33], p[39]);
    PIX_SORT(p[33], p[40]); PIX_SORT(p[33], p[41]); PIX_SORT(p[33], p[42]);
    PIX_SORT(p[33], p[43]); PIX_SORT(p[33], p[44]); PIX_SORT(p[33], p[45]);
    PIX_SORT(p[33], p[46]); PIX_SORT(p[33], p[47]); PIX_SORT(p[33], p[48]);
    PIX_SORT(p[34], p[35]); PIX_SORT(p[34], p[36]); PIX_SORT(p[34], p[37]);
    PIX_SORT(p[34], p[38]); PIX_SORT(p[34], p[39]); PIX_SORT(p[34], p[40]);
    PIX_SORT(p[34], p[41]); PIX_SORT(p[34], p[42]); PIX_SORT(p[34], p[43]);
    PIX_SORT(p[34], p[44]); PIX_SORT(p[34], p[45]); PIX_SORT(p[34], p[46]);
    PIX_SORT(p[34], p[47]); PIX_SORT(p[34], p[48]); PIX_SORT(p[35], p[36]);
    PIX_SORT(p[35], p[37]); PIX_SORT(p[35], p[38]); PIX_SORT(p[35], p[39]);
    PIX_SORT(p[35], p[40]); PIX_SORT(p[35], p[41]); PIX_SORT(p[35], p[42]);
    PIX_SORT(p[35], p[43]); PIX_SORT(p[35], p[44]); PIX_SORT(p[35], p[45]);
    PIX_SORT(p[35], p[46]); PIX_SORT(p[35], p[47]); PIX_SORT(p[35], p[48]);
    PIX_SORT(p[36], p[37]); PIX_SORT(p[36], p[38]); PIX_SORT(p[36], p[39]);
    PIX_SORT(p[36], p[40]); PIX_SORT(p[36], p[41]); PIX_SORT(p[36], p[42]);
    PIX_SORT(p[36], p[43]); PIX_SORT(p[36], p[44]); PIX_SORT(p[36], p[45]);
    PIX_SORT(p[36], p[46]); PIX_SORT(p[36], p[47]); PIX_SORT(p[36], p[48]);
    PIX_SORT(p[37], p[38]); PIX_SORT(p[37], p[39]); PIX_SORT(p[37], p[40]);
    PIX_SORT(p[37], p[41]); PIX_SORT(p[37], p[42]); PIX_SORT(p[37], p[43]);
    PIX_SORT(p[37], p[44]); PIX_SORT(p[37], p[45]); PIX_SORT(p[37], p[46]);
    PIX_SORT(p[37], p[47]); PIX_SORT(p[37], p[48]); PIX_SORT(p[38], p[39]);
    PIX_SORT(p[38], p[40]); PIX_SORT(p[38], p[41]); PIX_SORT(p[38], p[42]);
    PIX_SORT(p[38], p[43]); PIX_SORT(p[38], p[44]); PIX_SORT(p[38], p[45]);
    PIX_SORT(p[38], p[46]); PIX_SORT(p[38], p[47]); PIX_SORT(p[38], p[48]);
    PIX_SORT(p[39], p[40]); PIX_SORT(p[39], p[41]); PIX_SORT(p[39], p[42]);
    PIX_SORT(p[39], p[43]); PIX_SORT(p[39], p[44]); PIX_SORT(p[39], p[45]);
    PIX_SORT(p[39], p[46]); PIX_SORT(p[39], p[47]); PIX_SORT(p[39], p[48]);
    PIX_SORT(p[40], p[41]); PIX_SORT(p[40], p[42]); PIX_SORT(p[40], p[43]);
    PIX_SORT(p[40], p[44]); PIX_SORT(p[40], p[45]); PIX_SORT(p[40], p[46]);
    PIX_SORT(p[40], p[47]); PIX_SORT(p[40], p[48]); PIX_SORT(p[41], p[42]);
    PIX_SORT(p[41], p[43]); PIX_SORT(p[41], p[44]); PIX_SORT(p[41], p[45]);
    PIX_SORT(p[41], p[46]); PIX_SORT(p[41], p[47]); PIX_SORT(p[41], p[48]);
    PIX_SORT(p[42], p[43]); PIX_SORT(p[42], p[44]); PIX_SORT(p[42], p[45]);
    PIX_SORT(p[42], p[46]); PIX_SORT(p[42], p[47]); PIX_SORT(p[42], p[48]);
    PIX_SORT(p[43], p[44]); PIX_SORT(p[43], p[45]); PIX_SORT(p[43], p[46]);
    PIX_SORT(p[43], p[47]); PIX_SORT(p[43], p[48]); PIX_SORT(p[44], p[45]);
    PIX_SORT(p[44], p[46]); PIX_SORT(p[44], p[47]); PIX_SORT(p[44], p[48]);
    PIX_SORT(p[45], p[46]); PIX_SORT(p[45], p[47]); PIX_SORT(p[45], p[48]);
    PIX_SORT(p[46], p[47]); PIX_SORT(p[46], p[48]); PIX_SORT(p[47], p[48]);
    return p[24];
}

// Función para ordenar una ventana de píxeles
__device__ void sortWindow(unsigned char* window, int windowSize) {
    for (int i = 0; i < windowSize * windowSize - 1; i++) {
        for (int j = 0; j < windowSize * windowSize - i - 1; j++) {
            if (window[j] > window[j + 1]) {
                unsigned char temp = window[j];
                window[j] = window[j + 1];
                window[j + 1] = temp;
            }
        }
    }
}

// Kernel para aplicar el filtro de mediana usando memoria compartida
template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int WINDOW_SIZE>
__global__ void medianFilterSharedKernel(unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char sharedMem[BLOCK_DIM_Y + WINDOW_SIZE - 1][BLOCK_DIM_X + WINDOW_SIZE - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * BLOCK_DIM_X;
    int by = blockIdx.y * BLOCK_DIM_Y;
    int x = bx + tx;
    int y = by + ty;

    // Cargar datos en memoria compartida
    for (int dy = ty; dy < BLOCK_DIM_Y + WINDOW_SIZE - 1; dy += BLOCK_DIM_Y) {
        for (int dx = tx; dx < BLOCK_DIM_X + WINDOW_SIZE - 1; dx += BLOCK_DIM_X) {
            int globalX = bx + dx - WINDOW_SIZE / 2;
            int globalY = by + dy - WINDOW_SIZE / 2;

            if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
                sharedMem[dy][dx] = input[globalY * width + globalX];
            } else {
                sharedMem[dy][dx] = 0;
            }
        }
    }

    __syncthreads();

    // Aplicar el filtro de mediana
    if (x < width && y < height) {
        unsigned char window[WINDOW_SIZE * WINDOW_SIZE];
        int idx = 0;

        for (int wy = 0; wy < WINDOW_SIZE; wy++) {
            for (int wx = 0; wx < WINDOW_SIZE; wx++) {
                window[idx++] = sharedMem[ty + wy][tx + wx];
            }
        }

        if (WINDOW_SIZE == 3) {
            output[y * width + x] = opt_med9(window);
        } else if (WINDOW_SIZE == 5){
            output[y * width + x] = opt_med25(window);
        } else if (WINDOW_SIZE == 7){
            output[y * width + x] = opt_med49(window);
        } else {
            sortWindow(window, WINDOW_SIZE);
            output[y * width + x] = window[(WINDOW_SIZE * WINDOW_SIZE) / 2];
        }
    }
}

// Función para aplicar el filtro de mediana en la GPU y medir el tiempo
float applyMedianFilterGPU(const PGMImage& input, PGMImage& output, int windowSize) {
    unsigned char *d_input, *d_output;
    size_t size = input.width * input.height * sizeof(unsigned char);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input.data.data(), size, cudaMemcpyHostToDevice);

    const int BLOCK_DIM_X = 16;
    const int BLOCK_DIM_Y = 16;
    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize((input.width + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (input.height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Lanzar el kernel apropiado según el tamaño de la ventana
    switch (windowSize) {
        case 3:
            medianFilterSharedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 3><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        case 5:
            medianFilterSharedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 5><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        case 7:
            medianFilterSharedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 7><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        case 9:
            medianFilterSharedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 9><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        case 11:
            medianFilterSharedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 11><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        default:
            throw std::runtime_error("Tamaño de ventana no soportado");
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(output.data.data(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Uso: " << argv[0] << " <imagen_entrada.pgm> <imagen_salida.pgm> <tamaño_ventana>" << std::endl;
        return 1;
    }

    const char* inputFilename = argv[1];
    const char* outputFilename = argv[2];
    int windowSize = std::atoi(argv[3]);

    if (windowSize % 2 == 0) {
        std::cerr << "El tamaño de la ventana debe ser impar." << std::endl;
        return 1;
    }

    try {
        PGMImage img = readPGM(inputFilename);
        PGMImage filtered = img; // Inicializar con la misma estructura

        const int NUM_ITERATIONS = 10;
        std::vector<float> times(NUM_ITERATIONS);

        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            times[i] = applyMedianFilterGPU(img, filtered, windowSize);
        }

        // Calcular media
        float mean = std::accumulate(times.begin(), times.end(), 0.0f) / NUM_ITERATIONS;

        // Calcular desviación estándar
        float sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0f);
        float stdev = std::sqrt(sq_sum / NUM_ITERATIONS - mean * mean);

        std::cout << "Tiempo promedio: " << mean << " ms" << std::endl;
        std::cout << "Desviación estándar: " << stdev << " ms" << std::endl;

        writePGM(outputFilename, filtered);
        std::cout << "Filtro mediana aplicado exitosamente. Resultado guardado en " << outputFilename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}