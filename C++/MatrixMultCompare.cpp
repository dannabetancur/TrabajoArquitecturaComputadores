#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <immintrin.h>

using namespace std;
using namespace chrono;

// Detectar tamaño del vector SIMD
#ifdef __AVX512F__
    constexpr int SIMD_WIDTH = 16;
    #define SIMD_AVAILABLE "AVX-512"
#elif __AVX2__
    constexpr int SIMD_WIDTH = 8;
    #define SIMD_AVAILABLE "AVX2"
#elif __SSE2__
    constexpr int SIMD_WIDTH = 4;
    #define SIMD_AVAILABLE "SSE2"
#else
    constexpr int SIMD_WIDTH = 1;
    #define SIMD_AVAILABLE "None (scalar)"
#endif

struct TimingResult {
    long long total;
    long long transposeTime;
    long long vectorOpsTime;
    long long scalarTailTime;
};

void matrixMultNormal(const vector<vector<int>>& A, const vector<vector<int>>& B, 
                      vector<vector<int>>& C, int N) {
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

void matrixMultSimd(const vector<vector<int>>& A, const vector<vector<int>>& B, 
                    vector<vector<int>>& C, int N) {
    vector<vector<int>> BT(N, vector<int>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            BT[j][i] = B[i][j];
        }
    }
    
    // Multiplicación vectorizada
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int k = 0;
            int upperBound = (N / SIMD_WIDTH) * SIMD_WIDTH;
            int sum = 0;
            
#ifdef __AVX512F__
            __m512i vsum = _mm512_setzero_si512();
            for (; k < upperBound; k += SIMD_WIDTH) {
                __m512i va = _mm512_loadu_si512((__m512i*)&A[i][k]);
                __m512i vb = _mm512_loadu_si512((__m512i*)&BT[j][k]);
                __m512i vmul = _mm512_mullo_epi32(va, vb);
                vsum = _mm512_add_epi32(vsum, vmul);
            }
            // Reducción horizontal
            sum = _mm512_reduce_add_epi32(vsum);
#elif __AVX2__
            __m256i vsum = _mm256_setzero_si256();
            for (; k < upperBound; k += SIMD_WIDTH) {
                __m256i va = _mm256_loadu_si256((__m256i*)&A[i][k]);
                __m256i vb = _mm256_loadu_si256((__m256i*)&BT[j][k]);
                __m256i vmul = _mm256_mullo_epi32(va, vb);
                vsum = _mm256_add_epi32(vsum, vmul);
            }
            // Reducción horizontal AVX2
            __m128i vsum128 = _mm_add_epi32(_mm256_castsi256_si128(vsum), 
                                            _mm256_extracti128_si256(vsum, 1));
            vsum128 = _mm_hadd_epi32(vsum128, vsum128);
            vsum128 = _mm_hadd_epi32(vsum128, vsum128);
            sum = _mm_cvtsi128_si32(vsum128);
#elif __SSE2__
            __m128i vsum = _mm_setzero_si128();
            for (; k < upperBound; k += SIMD_WIDTH) {
                __m128i va = _mm_loadu_si128((__m128i*)&A[i][k]);
                __m128i vb = _mm_loadu_si128((__m128i*)&BT[j][k]);
                __m128i vmul = _mm_mullo_epi32(va, vb);
                vsum = _mm_add_epi32(vsum, vmul);
            }
            // Reducción horizontal SSE2
            vsum = _mm_hadd_epi32(vsum, vsum);
            vsum = _mm_hadd_epi32(vsum, vsum);
            sum = _mm_cvtsi128_si32(vsum);
#endif
            
            // Procesar elementos restantes (tail escalar)
            for (; k < N; k++) {
                sum += A[i][k] * BT[j][k];
            }
            
            C[i][j] = sum;
        }
    }
}

TimingResult matrixMultSimdTimed(const vector<vector<int>>& A, const vector<vector<int>>& B, 
                                  vector<vector<int>>& C, int N) {
    TimingResult result = {0, 0, 0, 0};
    auto startTotal = high_resolution_clock::now();
    
    // PASO 1: Transponer B
    auto startTranspose = high_resolution_clock::now();
    vector<vector<int>> BT(N, vector<int>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            BT[j][i] = B[i][j];
        }
    }
    auto endTranspose = high_resolution_clock::now();
    result.transposeTime = duration_cast<nanoseconds>(endTranspose - startTranspose).count();
    
    // PASO 2: Multiplicación vectorizada
    long long vectorOpsTime = 0;
    long long scalarTailTime = 0;
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int k = 0;
            int upperBound = (N / SIMD_WIDTH) * SIMD_WIDTH;
            
            auto startVector = high_resolution_clock::now();
            int sum = 0;
            
#ifdef __AVX512F__
            __m512i vsum = _mm512_setzero_si512();
            for (; k < upperBound; k += SIMD_WIDTH) {
                __m512i va = _mm512_loadu_si512((__m512i*)&A[i][k]);
                __m512i vb = _mm512_loadu_si512((__m512i*)&BT[j][k]);
                __m512i vmul = _mm512_mullo_epi32(va, vb);
                vsum = _mm512_add_epi32(vsum, vmul);
            }
            sum = _mm512_reduce_add_epi32(vsum);
#elif __AVX2__
            __m256i vsum = _mm256_setzero_si256();
            for (; k < upperBound; k += SIMD_WIDTH) {
                __m256i va = _mm256_loadu_si256((__m256i*)&A[i][k]);
                __m256i vb = _mm256_loadu_si256((__m256i*)&BT[j][k]);
                __m256i vmul = _mm256_mullo_epi32(va, vb);
                vsum = _mm256_add_epi32(vsum, vmul);
            }
            __m128i vsum128 = _mm_add_epi32(_mm256_castsi256_si128(vsum), 
                                            _mm256_extracti128_si256(vsum, 1));
            vsum128 = _mm_hadd_epi32(vsum128, vsum128);
            vsum128 = _mm_hadd_epi32(vsum128, vsum128);
            sum = _mm_cvtsi128_si32(vsum128);
#elif __SSE2__
            __m128i vsum = _mm_setzero_si128();
            for (; k < upperBound; k += SIMD_WIDTH) {
                __m128i va = _mm_loadu_si128((__m128i*)&A[i][k]);
                __m128i vb = _mm_loadu_si128((__m128i*)&BT[j][k]);
                __m128i vmul = _mm_mullo_epi32(va, vb);
                vsum = _mm_add_epi32(vsum, vmul);
            }
            vsum = _mm_hadd_epi32(vsum, vsum);
            vsum = _mm_hadd_epi32(vsum, vsum);
            sum = _mm_cvtsi128_si32(vsum);
#endif
            
            auto endVector = high_resolution_clock::now();
            vectorOpsTime += duration_cast<nanoseconds>(endVector - startVector).count();
            
            auto startTail = high_resolution_clock::now();
            for (; k < N; k++) {
                sum += A[i][k] * BT[j][k];
            }
            auto endTail = high_resolution_clock::now();
            scalarTailTime += duration_cast<nanoseconds>(endTail - startTail).count();
            
            C[i][j] = sum;
        }
    }
    
    result.vectorOpsTime = vectorOpsTime;
    result.scalarTailTime = scalarTailTime;
    
    auto endTotal = high_resolution_clock::now();
    result.total = duration_cast<nanoseconds>(endTotal - startTotal).count();
    
    return result;
}

int main() {
    int N = 1024;
    
    long long memoryNeeded = (long long)N * N * 4 * 4;
    
    cout << fixed << setprecision(2);
    cout << "Memoria estimada necesaria: " << (memoryNeeded / (1024*1024)) << " MB\n";
    cout << "Multiplicando matrices de " << N << "x" << N << " (enteros)...\n\n";
    
    // Crear matrices
    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    vector<vector<int>> C_normal(N, vector<int>(N));
    vector<vector<int>> C_simd(N, vector<int>(N));
    
    // Inicializar con valores aleatorios
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 100);
    
    auto initStart = high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = dis(gen);
            B[i][j] = dis(gen);
        }
    }
    auto initEnd = high_resolution_clock::now();
    
    cout << "Extensión SIMD disponible: " << SIMD_AVAILABLE << "\n";
    cout << "Tamaño del vector SIMD: " << SIMD_WIDTH << " elementos (int)\n";
    cout << "Operaciones necesarias: " << ((long long)N * N * N) << " multiplicaciones+sumas\n";
    cout << "Tiempo de inicialización: " 
         << duration_cast<milliseconds>(initEnd - initStart).count() << " ms\n\n";
    
    cout << "--- WARM-UP (ignorar estos tiempos) ---\n";
    
    // WARM-UP
    for (int iter = 0; iter < 3; iter++) {
        matrixMultNormal(A, B, C_normal, N);
        matrixMultSimd(A, B, C_simd, N);
    }
    
    cout << "\n--- MEDICIONES REALES ---\n";
    
    // Mediciones reales
    int iterations = 5;
    long long totalNormal = 0;
    long long totalSimd = 0;
    long long totalTranspose = 0;
    long long totalVectorOps = 0;
    long long totalScalarTail = 0;
    
    long long minNormal = LLONG_MAX;
    long long maxNormal = LLONG_MIN;
    long long minSimd = LLONG_MAX;
    long long maxSimd = LLONG_MIN;
    
    for (int iter = 0; iter < iterations; iter++) {
        cout << "Iteración " << (iter + 1) << "/" << iterations << "\n";
        
        // Multiplicación normal
        auto startNormal = high_resolution_clock::now();
        matrixMultNormal(A, B, C_normal, N);
        auto endNormal = high_resolution_clock::now();
        long long timeNormal = duration_cast<nanoseconds>(endNormal - startNormal).count();
        totalNormal += timeNormal;
        minNormal = min(minNormal, timeNormal);
        maxNormal = max(maxNormal, timeNormal);
        
        cout << "  Normal: " << (timeNormal / 1000000.0) << " ms\n";
        
        // Multiplicación SIMD
        TimingResult simdResult = matrixMultSimdTimed(A, B, C_simd, N);
        totalSimd += simdResult.total;
        totalTranspose += simdResult.transposeTime;
        totalVectorOps += simdResult.vectorOpsTime;
        totalScalarTail += simdResult.scalarTailTime;
        minSimd = min(minSimd, simdResult.total);
        maxSimd = max(maxSimd, simdResult.total);
        
        cout << "  SIMD:   " << (simdResult.total / 1000000.0) << " ms\n";
        cout << "    - Transposición: " << (simdResult.transposeTime / 1000000.0) << " ms\n";
        cout << "    - Ops vectoriales: " << (simdResult.vectorOpsTime / 1000000.0) << " ms\n";
        cout << "    - Tail escalar: " << (simdResult.scalarTailTime / 1000000.0) << " ms\n";
    }
    
    long long avgNormal = totalNormal / iterations;
    long long avgSimd = totalSimd / iterations;
    long long avgTranspose = totalTranspose / iterations;
    long long avgVectorOps = totalVectorOps / iterations;
    long long avgScalarTail = totalScalarTail / iterations;
    
    double avgNormalMillis = avgNormal / 1000000.0;
    double avgSimdMillis = avgSimd / 1000000.0;
    
    cout << "\n=== ESTADÍSTICAS DETALLADAS ===\n";
    cout << "\nTiempo Normal:\n";
    cout << "  Total: " << (totalNormal / 1000000.0) << " ms (" 
         << (totalNormal / 1000000000.0) << " s)\n";
    cout << "  Promedio: " << avgNormalMillis << " ms\n";
    cout << "  Mínimo: " << (minNormal / 1000000.0) << " ms\n";
    cout << "  Máximo: " << (maxNormal / 1000000.0) << " ms\n";
    cout << "  Desv. Est: " << ((maxNormal - minNormal) / 2.0 / 1000000.0) << " ms\n";
    
    cout << "\nTiempo SIMD:\n";
    cout << "  Total: " << (totalSimd / 1000000.0) << " ms (" 
         << (totalSimd / 1000000000.0) << " s)\n";
    cout << "  Promedio: " << avgSimdMillis << " ms\n";
    cout << "  Mínimo: " << (minSimd / 1000000.0) << " ms\n";
    cout << "  Máximo: " << (maxSimd / 1000000.0) << " ms\n";
    cout << "  Desv. Est: " << ((maxSimd - minSimd) / 2.0 / 1000000.0) << " ms\n";
    
    cout << "\nDesglose SIMD (promedio):\n";
    cout << "  Transposición: " << (avgTranspose / 1000000.0) 
         << " ms (" << (100.0 * avgTranspose / avgSimd) << "%)\n";
    cout << "  Ops vectoriales: " << (avgVectorOps / 1000000.0) 
         << " ms (" << (100.0 * avgVectorOps / avgSimd) << "%)\n";
    cout << "  Tail escalar: " << (avgScalarTail / 1000000.0) 
         << " ms (" << (100.0 * avgScalarTail / avgSimd) << "%)\n";
    
    cout << "\n=== RENDIMIENTO ===\n";
    if (avgSimd < avgNormal) {
        double speedup = (double)avgNormal / avgSimd;
        double improvement = ((avgNormal - avgSimd) / (double)avgNormal) * 100;
        cout << "✓ Speedup SIMD: " << speedup << "x más rápido\n";
        cout << "  Mejora: " << improvement << "%\n";
        cout << "  Tiempo ahorrado: " << ((avgNormal - avgSimd) / 1000000.0) << " ms\n";
    } else {
        double slowdown = (double)avgSimd / avgNormal;
        cout << "✗ SIMD es " << slowdown << "x más lento\n";
    }
    
    // Verificación
    bool iguales = true;
    long long maxError = 0;
    int errores = 0;
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            long long diff = abs((long long)C_normal[i][j] - (long long)C_simd[i][j]);
            if (diff > 0) {
                iguales = false;
                errores++;
                if (diff > maxError) {
                    maxError = diff;
                }
                if (errores <= 5) {
                    cout << "\n¡ERROR! Diferencia en [" << i << "][" << j << "]: " 
                         << C_normal[i][j] << " vs " << C_simd[i][j] 
                         << " (diff: " << diff << ")\n";
                }
            }
        }
    }
    
    if (iguales) {
        cout << "\n✓ Resultados correctos (matrices idénticas)\n";
    } else {
        cout << "\n✗ Se encontraron " << errores << " diferencias\n";
        cout << "Error máximo: " << maxError << "\n";
    }
    
    // GIOPS
    long long operations = (long long)N * N * (2LL * N - 1);
    double giopsNormal = operations / (avgNormal / 1e9) / 1e9;
    double giopsSimd = operations / (avgSimd / 1e9) / 1e9;
    
    cout << "\nRendimiento computacional:\n";
    cout << "Normal: " << giopsNormal << " GIOPS\n";
    cout << "SIMD:   " << giopsSimd << " GIOPS\n";
    
    // Throughput
    double throughputNormal = (N * N * 1000.0) / avgNormalMillis;
    double throughputSimd = (N * N * 1000.0) / avgSimdMillis;
    cout << "\nThroughput:\n";
    cout << "Normal: " << throughputNormal << " M elementos/segundo\n";
    cout << "SIMD:   " << throughputSimd << " M elementos/segundo\n";
    
    return 0;
}