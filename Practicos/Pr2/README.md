En la carpeta data se pueden encontrar:
- secreto.txt: contiene el mensaje a descifrar
- texto.txt: contiene el mensaje descifrado
- cantidadesSecuencial: contiene el resultado de la ejecucion de la solucion al ejercicio 2 que ejecuta en cpu de manera secuencial
- cantidadesCUDA: contiene el resultado del ejericio dos usando CUDA y GPU.
- test.txt: se uso para verificar las cantidades que estabamos obteniendo

En la carpeta Ej1:
- ej1.cu: solucion a la parte a del problema 1
- ej1b.cu: solucion a la parte b del problema1
- ej1c.cu: solucion a la parte c del problema
- los demas archivos .exe, .exp y .lib son resultado de la compulacion usando nvcc

En la carpeta Ej2:
- ej2.cu: solucion al problema 2
- Ej2_secuencial.c: solucion al problema 2 pero implementada secuencial para CPU. Se utilizo para comparar los resultados y verificar el correcto funcionamiento del codigo CUDA.
- los demas archivos .exe, .exp y .lib son resultado de la compulacion usando nvcc
