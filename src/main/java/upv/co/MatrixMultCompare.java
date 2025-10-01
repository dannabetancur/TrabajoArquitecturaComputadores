//Código para compilar y ejecutar java desde el terminal
// >> javac --add-modules jdk.incubator.vector upv/co/MatrixMultCompare.java
// >> java --add-modules jdk.incubator.vector upv.co.MatrixMultCompare

package upv.co;

import java.util.Random;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

public class MatrixMultCompare {
    
    private static final VectorSpecies<Integer> SPECIES = IntVector.SPECIES_PREFERRED;
    
    public static void main(String[] args) {
        System.err.close();
        
        int N = 512;
        
        long memoryNeeded = (long) N * N * 4 * 4; // 4 matrices de enteros (A, B, BT, 2xC)
        long maxMemory = Runtime.getRuntime().maxMemory();
        
        System.out.println("Memoria heap máxima: " + (maxMemory / (1024*1024)) + " MB");
        System.out.println("Memoria estimada necesaria: " + (memoryNeeded / (1024*1024)) + " MB");
        System.out.println("Multiplicando matrices de " + N + "x" + N + " (enteros)...\n");
        
        // Crear matrices de enteros
        int[][] A = new int[N][N];
        int[][] B = new int[N][N];
        int[][] C_normal = new int[N][N];
        int[][] C_simd = new int[N][N];
        
        // Inicializar con valores aleatorios (enteros 0-100)
        Random r = new Random();
        long initStart = System.nanoTime();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = r.nextInt(101); // 0 a 100 inclusive
                B[i][j] = r.nextInt(101);
            }
        }
        long initEnd = System.nanoTime();
        
        System.out.println("Tamaño del vector SIMD: " + SPECIES.length() + " elementos (int)");
        System.out.println("Operaciones necesarias: " + (N * N * N) + " multiplicaciones+sumas");
        System.out.println("Tiempo de inicialización: " + String.format("%.2f", (initEnd - initStart) / 1_000_000.0) + " ms\n");
        
        System.out.println("--- WARM-UP (ignorar estos tiempos) ---");
        
        // WARM-UP: 3 iteraciones para que JIT optimice
        for (int iter = 0; iter < 3; iter++) {
            matrixMultNormal(A, B, C_normal, N);
            matrixMultSimd(A, B, C_simd, N);
        }
        
        System.out.println("\n--- MEDICIONES REALES ---");
        
        // Mediciones reales con desglose de operaciones
        int iterations = 5;
        long totalNormal = 0;
        long totalSimd = 0;
        long totalTranspose = 0;
        long totalVectorOps = 0;
        long totalScalarTail = 0;
        
        long minNormal = Long.MAX_VALUE;
        long maxNormal = Long.MIN_VALUE;
        long minSimd = Long.MAX_VALUE;
        long maxSimd = Long.MIN_VALUE;
        
        for (int iter = 0; iter < iterations; iter++) {
            System.out.println("Iteración " + (iter + 1) + "/" + iterations);
            
            // Multiplicación normal
            long startNormal = System.nanoTime();
            matrixMultNormal(A, B, C_normal, N);
            long endNormal = System.nanoTime();
            long timeNormal = endNormal - startNormal;
            totalNormal += timeNormal;
            minNormal = Math.min(minNormal, timeNormal);
            maxNormal = Math.max(maxNormal, timeNormal);
            
            System.out.println("  Normal: " + String.format("%.2f", timeNormal / 1_000_000.0) + " ms");
            
            // Multiplicación SIMD con desglose
            TimingResult simdResult = matrixMultSimdTimed(A, B, C_simd, N);
            totalSimd += simdResult.total;
            totalTranspose += simdResult.transposeTime;
            totalVectorOps += simdResult.vectorOpsTime;
            totalScalarTail += simdResult.scalarTailTime;
            minSimd = Math.min(minSimd, simdResult.total);
            maxSimd = Math.max(maxSimd, simdResult.total);
            
            System.out.println("  SIMD:   " + String.format("%.2f", simdResult.total / 1_000_000.0) + " ms");
            System.out.println("    - Transposición: " + String.format("%.2f", simdResult.transposeTime / 1_000_000.0) + " ms");
            System.out.println("    - Ops vectoriales: " + String.format("%.2f", simdResult.vectorOpsTime / 1_000_000.0) + " ms");
            System.out.println("    - Tail escalar: " + String.format("%.2f", simdResult.scalarTailTime / 1_000_000.0) + " ms");
        }
        
        long avgNormal = totalNormal / iterations;
        long avgSimd = totalSimd / iterations;
        long avgTranspose = totalTranspose / iterations;
        long avgVectorOps = totalVectorOps / iterations;
        long avgScalarTail = totalScalarTail / iterations;
        
        double avgNormalMillis = avgNormal / 1_000_000.0;
        double avgSimdMillis = avgSimd / 1_000_000.0;
        
        System.out.println("\n=== ESTADÍSTICAS DETALLADAS ===");
        System.out.println("\nTiempo Normal:");
        System.out.println("  Total: " + String.format("%.2f", totalNormal / 1_000_000.0) + " ms (" + 
            String.format("%.2f", totalNormal / 1_000_000_000.0) + " s)");
        System.out.println("  Promedio: " + String.format("%.2f", avgNormalMillis) + " ms");
        System.out.println("  Mínimo: " + String.format("%.2f", minNormal / 1_000_000.0) + " ms");
        System.out.println("  Máximo: " + String.format("%.2f", maxNormal / 1_000_000.0) + " ms");
        System.out.println("  Desv. Est: " + String.format("%.2f", 
            (maxNormal - minNormal) / 2.0 / 1_000_000.0) + " ms");
        
        System.out.println("\nTiempo SIMD:");
        System.out.println("  Total: " + String.format("%.2f", totalSimd / 1_000_000.0) + " ms (" + 
            String.format("%.2f", totalSimd / 1_000_000_000.0) + " s)");
        System.out.println("  Promedio: " + String.format("%.2f", avgSimdMillis) + " ms");
        System.out.println("  Mínimo: " + String.format("%.2f", minSimd / 1_000_000.0) + " ms");
        System.out.println("  Máximo: " + String.format("%.2f", maxSimd / 1_000_000.0) + " ms");
        System.out.println("  Desv. Est: " + String.format("%.2f", 
            (maxSimd - minSimd) / 2.0 / 1_000_000.0) + " ms");
        
        System.out.println("\nDesglose SIMD (promedio):");
        System.out.println("  Transposición: " + String.format("%.2f", avgTranspose / 1_000_000.0) + 
            " ms (" + String.format("%.1f%%", 100.0 * avgTranspose / avgSimd) + ")");
        System.out.println("  Ops vectoriales: " + String.format("%.2f", avgVectorOps / 1_000_000.0) + 
            " ms (" + String.format("%.1f%%", 100.0 * avgVectorOps / avgSimd) + ")");
        System.out.println("  Tail escalar: " + String.format("%.2f", avgScalarTail / 1_000_000.0) + 
            " ms (" + String.format("%.1f%%", 100.0 * avgScalarTail / avgSimd) + ")");
        
        System.out.println("\n=== RENDIMIENTO ===");
        if (avgSimd < avgNormal) {
            double speedup = (double)avgNormal / avgSimd;
            double improvement = ((avgNormal - avgSimd) / (double)avgNormal) * 100;
            System.out.println("✓ Speedup SIMD: " + String.format("%.2fx más rápido", speedup));
            System.out.println("  Mejora: " + String.format("%.1f%%", improvement));
            System.out.println("  Tiempo ahorrado: " + String.format("%.2f", (avgNormal - avgSimd) / 1_000_000.0) + " ms");
        } else {
            double slowdown = (double)avgSimd / avgNormal;
            System.out.println("✗ SIMD es " + String.format("%.2fx más lento", slowdown));
        }
        
        // Verificación
        boolean iguales = true;
        long maxError = 0;
        int errores = 0;
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                long diff = Math.abs((long)C_normal[i][j] - (long)C_simd[i][j]);
                if (diff > 0) {
                    iguales = false;
                    errores++;
                    if (diff > maxError) {
                        maxError = diff;
                    }
                    if (errores <= 5) {
                        System.out.println("\n¡ERROR! Diferencia en [" + i + "][" + j + "]: " + 
                                         C_normal[i][j] + " vs " + C_simd[i][j] + 
                                         " (diff: " + diff + ")");
                    }
                }
            }
        }
        
        if (iguales) {
            System.out.println("\n✓ Resultados correctos (matrices idénticas)");
        } else {
            System.out.println("\n✗ Se encontraron " + errores + " diferencias");
            System.out.println("Error máximo: " + maxError);
        }
        
        // FLOPS (Floating Point Operations Per Second) - aunque sean enteros, usamos la misma métrica
        long operations = (long) N * N * (2L * N - 1);
        double gflopsNormal = operations / (avgNormal / 1e9) / 1e9;
        double gflopsSimd = operations / (avgSimd / 1e9) / 1e9;
        
        System.out.println("\nRendimiento computacional:");
        System.out.println("Normal: " + String.format("%.2f", gflopsNormal) + " GFLOPS");
        System.out.println("SIMD:   " + String.format("%.2f", gflopsSimd) + " GFLOPS");
        
        // Throughput
        double throughputNormal = (N * N * 1000.0) / avgNormalMillis; // elementos/ms
        double throughputSimd = (N * N * 1000.0) / avgSimdMillis;
        System.out.println("\nThroughput:");
        System.out.println("Normal: " + String.format("%.2f", throughputNormal) + " M elementos/segundo");
        System.out.println("SIMD:   " + String.format("%.2f", throughputSimd) + " M elementos/segundo");
        
        // Muestra de resultados
        System.out.println("\nMuestra de C_simd[0][0..4]:");
        for (int j = 0; j < Math.min(5, N); j++) {
            System.out.print(C_simd[0][j] + " ");
        }
        System.out.println();
    }
    
    static class TimingResult {
        long total;
        long transposeTime;
        long vectorOpsTime;
        long scalarTailTime;
    }
    
    private static void matrixMultNormal(int[][] A, int[][] B, int[][] C, int N) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int sum = 0;
                for (int k = 0; k < N; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }
    
    private static void matrixMultSimd(int[][] A, int[][] B, int[][] C, int N) {
        // Transponer B
        int[][] BT = new int[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                BT[j][i] = B[i][j];
            }
        }
        
        // Multiplicación vectorizada
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int k = 0;
                int upperBound = SPECIES.loopBound(N);
                
                IntVector vsum = IntVector.zero(SPECIES);
                
                for (; k < upperBound; k += SPECIES.length()) {
                    IntVector va = IntVector.fromArray(SPECIES, A[i], k);
                    IntVector vb = IntVector.fromArray(SPECIES, BT[j], k);
                    vsum = vsum.add(va.mul(vb));  // CORREGIDO: mul + add en lugar de fma
                }
                
                int sum = vsum.reduceLanes(VectorOperators.ADD);
                
                for (; k < N; k++) {
                    sum += A[i][k] * BT[j][k];
                }
                
                C[i][j] = sum;
            }
        }
    }
    
    private static TimingResult matrixMultSimdTimed(int[][] A, int[][] B, int[][] C, int N) {
        TimingResult result = new TimingResult();
        long startTotal = System.nanoTime();
        
        // PASO 1: Transponer B
        long startTranspose = System.nanoTime();
        int[][] BT = new int[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                BT[j][i] = B[i][j];
            }
        }
        long endTranspose = System.nanoTime();
        result.transposeTime = endTranspose - startTranspose;
        
        // PASO 2: Multiplicación vectorizada
        long vectorOpsTime = 0;
        long scalarTailTime = 0;
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int k = 0;
                int upperBound = SPECIES.loopBound(N);
                
                long startVector = System.nanoTime();
                IntVector vsum = IntVector.zero(SPECIES);
                
                for (; k < upperBound; k += SPECIES.length()) {
                    IntVector va = IntVector.fromArray(SPECIES, A[i], k);
                    IntVector vb = IntVector.fromArray(SPECIES, BT[j], k);
                    vsum = vsum.add(va.mul(vb));  // CORREGIDO: mul + add en lugar de fma
                }
                
                int sum = vsum.reduceLanes(VectorOperators.ADD);
                long endVector = System.nanoTime();
                vectorOpsTime += (endVector - startVector);
                
                long startTail = System.nanoTime();
                for (; k < N; k++) {
                    sum += A[i][k] * BT[j][k];
                }
                long endTail = System.nanoTime();
                scalarTailTime += (endTail - startTail);
                
                C[i][j] = sum;
            }
        }
        
        result.vectorOpsTime = vectorOpsTime;
        result.scalarTailTime = scalarTailTime;
        
        long endTotal = System.nanoTime();
        result.total = endTotal - startTotal;
        
        return result;
    }
}