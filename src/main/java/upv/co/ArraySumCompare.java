//Código para compilar y ejecutar java desde el terminal
// >> javac --add-modules jdk.incubator.vector upv\co\ArraySumCompare.java
// >> java --add-modules jdk.incubator.vector upv.co.ArraySumCompare

package upv.co;

import java.util.Random;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;

public class ArraySumCompare {
    
    private static final VectorSpecies<Integer> SPECIES = IntVector.SPECIES_PREFERRED;
    
    public static void main(String[] args) {
        System.err.close();
        
        // Calcular tamaño máximo seguro (90% del heap disponible / 4 arrays)
        long maxMemory = Runtime.getRuntime().maxMemory();
        long maxElements = (long) ((maxMemory * 0.9) / (4 * 4)); // 4 arrays, 4 bytes/int
        
        int size = 1_000_000_000; // 100 millones por defecto
        
        // Ajustar si supera la memoria disponible
        if (size > maxElements) {
            size = (int) maxElements;
            System.out.println("⚠️ Tamaño ajustado a " + size + " por límite de memoria");
        }
        
        // Límite máximo de arrays en Java
        if (size > Integer.MAX_VALUE - 8) {
            size = Integer.MAX_VALUE - 8;
            System.out.println("⚠️ Tamaño ajustado a " + size + " (límite de array en Java)");
        }
        
        System.out.println("Memoria heap máxima: " + (maxMemory / (1024*1024)) + " MB");
        System.out.println("Memoria estimada necesaria: " + (size * 4L * 4 / (1024*1024)) + " MB");
        System.out.println("Creando arrays de " + size + " elementos...\n");
        
        int[] a = new int[size];
        int[] b = new int[size];
        int[] cNormal = new int[size];
        int[] cSimd = new int[size];

        Random r = new Random();

        for (int i = 0; i < size; i++) {
            a[i] = r.nextInt(100);
            b[i] = r.nextInt(100);
        }

        System.out.println("Tamaño del vector SIMD: " + SPECIES.length() + " elementos (int)");
        System.out.println("Número de elementos: " + size);
        System.out.println("Iteraciones vectorizadas: " + (size / SPECIES.length()));
        System.out.println("\n--- WARM-UP (ignorar estos tiempos) ---");
        
        // WARM-UP: 5 iteraciones para que JIT optimice
        for (int iter = 0; iter < 5; iter++) {
            sumNormal(a, b, cNormal);
            sumSimd(a, b, cSimd);
        }
        
        System.out.println("\n--- MEDICIONES REALES ---");
        
        // Mediciones reales con múltiples iteraciones
        int iterations = 10;
        long totalNormal = 0;
        long totalSimd = 0;
        
        for (int iter = 0; iter < iterations; iter++) {
            // Suma normal
            long startNormal = System.nanoTime();
            sumNormal(a, b, cNormal);
            long endNormal = System.nanoTime();
            totalNormal += (endNormal - startNormal);
            
            // Suma SIMD
            long startSimd = System.nanoTime();
            sumSimd(a, b, cSimd);
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
        System.out.println("Suma normal: " + totalNormal + " ns (" + 
                          String.format("%.2f", totalNormalMillis) + " ms)");
        System.out.println("Suma SIMD:   " + totalSimd + " ns (" + 
                          String.format("%.2f", totalSimdMillis) + " ms)");
        
        System.out.println("\n=== TIEMPO PROMEDIO (por iteración) ===");
        System.out.println("Suma normal: " + avgNormal + " ns (" + 
                          String.format("%.2f", avgNormalMillis) + " ms)");
        System.out.println("Suma SIMD:   " + avgSimd + " ns (" + 
                          String.format("%.2f", avgSimdMillis) + " ms)");
        
        System.out.println("\n=== RENDIMIENTO ===");
        if (avgSimd < avgNormal) {
            double speedup = (double)avgNormal / avgSimd;
            System.out.println("Speedup SIMD: " + String.format("%.2fx más rápido", speedup));
        } else {
            double slowdown = (double)avgSimd / avgNormal;
            System.out.println("SIMD es " + String.format("%.2fx más lento", slowdown));
            System.out.println("\nPosibles razones:");
            System.out.println("- Operación muy simple (suma), el overhead de SIMD no compensa");
            System.out.println("- Memory-bound: el cuello de botella es leer/escribir memoria");
            System.out.println("- JIT puede auto-vectorizar el código normal");
        }
        
        // Verificación
        boolean iguales = true;
        for (int j = 0; j < size; j++) {
            if (cNormal[j] != cSimd[j]) {
                iguales = false;
                System.out.println("\n¡ERROR! Diferencia en índice " + j + 
                                 ": " + cNormal[j] + " vs " + cSimd[j]);
                break;
            }
        }
        System.out.println("\n¿Resultados iguales? " + iguales);
        
        // Throughput
        long elementsProcessed = (long)size * iterations;
        double throughputNormal = elementsProcessed / (totalNormal / 1e9); // elementos/segundo
        double throughputSimd = elementsProcessed / (totalSimd / 1e9);
        
        System.out.println("\nThroughput:");
        System.out.println("Normal: " + String.format("%.2f", throughputNormal / 1e6) + " M elementos/seg");
        System.out.println("SIMD:   " + String.format("%.2f", throughputSimd / 1e6) + " M elementos/seg");
    }
    
    private static void sumNormal(int[] a, int[] b, int[] c) {
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }
    }
    
    private static void sumSimd(int[] a, int[] b, int[] c) {
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        // Procesar bloques completos con SIMD
        for (; i < upperBound; i += SPECIES.length()) {
            IntVector va = IntVector.fromArray(SPECIES, a, i);
            IntVector vb = IntVector.fromArray(SPECIES, b, i);
            IntVector vc = va.add(vb);
            vc.intoArray(c, i);
        }
        
        // Cola escalar
        for (; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }
    }
}