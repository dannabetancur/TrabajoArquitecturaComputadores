//Código para compilar y ejecutar java desde el terminal
// >> javac --add-modules jdk.incubator.vector upv/co/MatrixMultCompare.java
// >> java --add-modules jdk.incubator.vector upv.co.MatrixMultCompare

package upv.co;

import java.util.Random;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.IntVector;

public class MatrixMultCompare {
    
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    public static void main(String[] args) {
        System.err.close();
        
        // Tamaños de matrices (NxN)
        // Nota: matrices grandes consumen MUCHA memoria (N*N*4 bytes * 3 matrices)
        int N = 512; // Puedes probar con 256, 512, 1024, 2048
        
        long memoryNeeded = (long) N * N * 4 * 3; // 3 matrices de floats
        long maxMemory = Runtime.getRuntime().maxMemory();
        
        System.out.println("Memoria heap máxima: " + (maxMemory / (1024*1024)) + " MB");
        System.out.println("Memoria estimada necesaria: " + (memoryNeeded / (1024*1024)) + " MB");
        System.out.println("Multiplicando matrices de " + N + "x" + N + "...\n");
        
        // Crear matrices
        float[][] A = new float[N][N];
        float[][] B = new float[N][N];
        float[][] C_normal = new float[N][N];
        float[][] C_simd = new float[N][N];
        
        // Inicializar con valores aleatorios
        Random r = new Random(42);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = r.nextFloat() * 10; // Valores entre 0 y 10
                B[i][j] = r.nextFloat() * 10;
            }
        }
        
        System.out.println("Tamaño del vector SIMD: " + SPECIES.length() + " elementos (float)");
        System.out.println("Operaciones necesarias: " + (N * N * N) + " multiplicaciones+sumas");
        System.out.println("\n--- WARM-UP (ignorar estos tiempos) ---");
        
        // WARM-UP: 3 iteraciones para que JIT optimice
        for (int iter = 0; iter < 3; iter++) {
            matrixMultNormal(A, B, C_normal, N);
            matrixMultSimd(A, B, C_simd, N);
        }
        
        System.out.println("\n--- MEDICIONES REALES ---");
        
        // Mediciones reales
        int iterations = 5;
        long totalNormal = 0;
        long totalSimd = 0;
        
        for (int iter = 0; iter < iterations; iter++) {
            System.out.println("Iteración " + (iter + 1) + "/" + iterations);
            
            // Multiplicación normal
            long startNormal = System.nanoTime();
            matrixMultNormal(A, B, C_normal, N);
            long endNormal = System.nanoTime();
            totalNormal += (endNormal - startNormal);
            
            // Multiplicación SIMD
            long startSimd = System.nanoTime();
            matrixMultSimd(A, B, C_simd, N);
            long endSimd = System.nanoTime();
            totalSimd += (endSimd - startSimd);
        }
        
        long avgNormal = totalNormal / iterations;
        long avgSimd = totalSimd / iterations;
        
        double avgNormalMillis = avgNormal / 1_000_000.0;
        double avgSimdMillis = avgSimd / 1_000_000.0;
        
        // Mostrar resultados
        double totalNormalMillis = totalNormal / 1_000_000.0;
        double totalSimdMillis = totalSimd / 1_000_000.0;
        
        System.out.println("\n=== TIEMPO TOTAL (" + iterations + " iteraciones) ===");
        System.out.println("Mult normal: " + totalNormal + " ns (" + 
                          String.format("%.2f", totalNormalMillis) + " ms)");
        System.out.println("Mult SIMD:   " + totalSimd + " ns (" + 
                          String.format("%.2f", totalSimdMillis) + " ms)");
        
        System.out.println("\n=== TIEMPO PROMEDIO (por iteración) ===");
        System.out.println("Mult normal: " + avgNormal + " ns (" + 
                          String.format("%.2f", avgNormalMillis) + " ms)");
        System.out.println("Mult SIMD:   " + avgSimd + " ns (" + 
                          String.format("%.2f", avgSimdMillis) + " ms)");
        
        System.out.println("\n=== RENDIMIENTO ===");
        if (avgSimd < avgNormal) {
            double speedup = (double)avgNormal / avgSimd;
            System.out.println("✓ Speedup SIMD: " + String.format("%.2fx más rápido", speedup));
        } else {
            double slowdown = (double)avgSimd / avgNormal;
            System.out.println("✗ SIMD es " + String.format("%.2fx más lento", slowdown));
        }
        
        // Verificación (con tolerancia por errores de punto flotante)
        boolean iguales = true;
        float maxError = 0;
        int errores = 0;
        float tolerance = 0.01f; // Tolerancia para errores de redondeo
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float diff = Math.abs(C_normal[i][j] - C_simd[i][j]);
                if (diff > tolerance) {
                    iguales = false;
                    errores++;
                    if (diff > maxError) {
                        maxError = diff;
                    }
                    if (errores <= 5) { // Mostrar solo los primeros 5 errores
                        System.out.println("\n¡ERROR! Diferencia en [" + i + "][" + j + "]: " + 
                                         C_normal[i][j] + " vs " + C_simd[i][j] + 
                                         " (diff: " + diff + ")");
                    }
                }
            }
        }
        
        if (iguales) {
            System.out.println("\n✓ Resultados correctos (diferencias < " + tolerance + ")");
        } else {
            System.out.println("\n✗ Se encontraron " + errores + " diferencias");
            System.out.println("Error máximo: " + maxError);
        }
        
        // FLOPS (Floating Point Operations Per Second)
        long operations = (long) N * N * (2L * N - 1); // N^2 * (2N-1) ops
        double gflopsNormal = operations / (avgNormal / 1e9) / 1e9;
        double gflopsSimd = operations / (avgSimd / 1e9) / 1e9;
        
        System.out.println("\nRendimiento computacional:");
        System.out.println("Normal: " + String.format("%.2f", gflopsNormal) + " GFLOPS");
        System.out.println("SIMD:   " + String.format("%.2f", gflopsSimd) + " GFLOPS");
        
        // Muestra de resultados
        System.out.println("\nMuestra de C_simd[0][0..4]:");
        for (int j = 0; j < Math.min(5, N); j++) {
            System.out.print(String.format("%.2f ", C_simd[0][j]));
        }
        System.out.println();
    }
    
    /**
     * Multiplicación de matrices clásica: C = A × B
     * Algoritmo: C[i][j] = sum(A[i][k] * B[k][j]) para k=0..N-1
     */
    private static void matrixMultNormal(float[][] A, float[][] B, float[][] C, int N) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < N; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }
    
    /**
     * Multiplicación de matrices con SIMD
     * Vectoriza el bucle interno (k) para procesar múltiples multiplicaciones en paralelo
     */
    private static void matrixMultSimd(float[][] A, float[][] B, float[][] C, int N) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int k = 0;
                int upperBound = SPECIES.loopBound(N);
                
                // Acumulador vectorial para la suma parcial
                FloatVector vsum = FloatVector.zero(SPECIES);
                
                // Procesar bloques completos con SIMD
                for (; k < upperBound; k += SPECIES.length()) {
                    // Cargar vectores de A[i][k..k+vecLen] y B[k..k+vecLen][j]
                    FloatVector va = FloatVector.fromArray(SPECIES, A[i], k);
                    
                    // Para B necesitamos elementos no contiguos: B[k][j], B[k+1][j], ...
                    float[] bColumn = new float[SPECIES.length()];
                    for (int v = 0; v < SPECIES.length(); v++) {
                        bColumn[v] = B[k + v][j];
                    }
                    FloatVector vb = FloatVector.fromArray(SPECIES, bColumn, 0);
                    
                    // Multiplicar y acumular
                    vsum = va.fma(vb, vsum); // fma = fused multiply-add: vsum += va * vb
                }
                
                // Reducir el vector a un escalar (sumar todos los elementos)
                float sum = vsum.reduceLanes(VectorOperators.ADD);
                
                // Procesar elementos restantes (cola escalar)
                for (; k < N; k++) {
                    sum += A[i][k] * B[k][j];
                }
                
                C[i][j] = sum;
            }
        }
    }
}
