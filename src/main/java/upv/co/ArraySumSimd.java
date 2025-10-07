//Código para compilar y ejecutar java desde el terminal
// >> javac --add-modules jdk.incubator.vector ArraySumSimd.java
// >> java --add-modules jdk.incubator.vector ArraySumSimd.java


package upv.co;

import java.util.Random;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;

public class ArraySumSimd {
    public static void main(String[] args) {
        int size = 10000;
        int[] a = new int[size];
        int[] b = new int[size];
        int[] c = new int[size];

        Random r = new Random();

        for (int i = 0; i < size; i++) {
            a[i] = r.nextInt(100);
            b[i] = r.nextInt(100);
        }

        VectorSpecies<Integer> SPECIES = IntVector.SPECIES_PREFERRED;

        long startTime = System.nanoTime();

        int i = 0;
        int upperBound = SPECIES.loopBound(size);
        for (; i < upperBound; i += SPECIES.length()) {
            IntVector va = IntVector.fromArray(SPECIES, a, i);
            IntVector vb = IntVector.fromArray(SPECIES, b, i);
            IntVector vc = va.add(vb);
            vc.intoArray(c, i);
        }
        // Suma escalar para los elementos restantes
        for (; i < size; i++) {
            c[i] = a[i] + b[i];
        }

        long endTime = System.nanoTime();

        long durationNano = endTime - startTime;
        double durationMillis = durationNano / 1_000_000.0;

        System.out.println("Suma lista (SIMD):");
        for (int j = 0; j < size; j++) {
            System.out.print(c[j] + (j < size - 1 ? " " : "\n"));
        }

        System.out.println("\nTiempo de ejecución (SIMD):");
        System.out.println(durationNano + " nanosegundos");
        System.out.println(durationMillis + " milisegundos");
    }
}
