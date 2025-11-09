// CC3086 - Lab 9: Chat-Box con IA para Smart Agriculture (CUDA)
// Sistema completo de control inteligente de invernadero
// Compilar: nvcc -O3 -std=c++17 main.cu -o chatbox_cuda

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <ctime>

// ======================== UTILIDADES ========================
#define CUDA_OK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line){
    if (code != cudaSuccess){ 
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line); 
        exit(code); 
    }
}

inline int ceilDiv(int a, int b){ return (a + b - 1) / b; }

// ======================== PARÁMETROS ========================
constexpr int D = 8192;
constexpr int K = 10;
constexpr int MAX_QUERY = 512;
constexpr int C = 5;
constexpr int N = 1<<20;
constexpr int W = 2048;

// ======================== INTENCIONES ========================
enum Intent { 
    CONSULTAR_HUMEDAD = 0,
    CONSULTAR_TEMP = 1,
    ACTIVAR_RIEGO = 2,
    DESACTIVAR_RIEGO = 3,
    PROGRAMAR_RIEGO = 4,
    CONSULTAR_ESTADO = 5,
    ACTIVAR_VENTILACION = 6,
    DESACTIVAR_VENTILACION = 7,
    AYUDA = 8,
    DIAGNOSTICO = 9
};

static const char* intentNames[K] = {
    "CONSULTAR_HUMEDAD", "CONSULTAR_TEMP", "ACTIVAR_RIEGO", 
    "DESACTIVAR_RIEGO", "PROGRAMAR_RIEGO", "CONSULTAR_ESTADO",
    "ACTIVAR_VENTILACION", "DESACTIVAR_VENTILACION", "AYUDA", "DIAGNOSTICO"
};

// ======================== FUNCIONES DEVICE STRING ========================
__device__ int d_strlen(const char* str) {
    int len = 0;
    while (str[len] != '\0') len++;
    return len;
}

__device__ void d_strcpy(char* dst, const char* src) {
    int i = 0;
    while (src[i] != '\0') {
        dst[i] = src[i];
        i++;
    }
    dst[i] = '\0';
}

__device__ void d_strcat(char* dst, const char* src) {
    int dst_len = d_strlen(dst);
    int i = 0;
    while (src[i] != '\0') {
        dst[dst_len + i] = src[i];
        i++;
    }
    dst[dst_len + i] = '\0';
}

__device__ void d_ftoa(float value, char* buffer, int decimals) {
    int int_part = (int)value;
    float frac_part = value - int_part;
    if (frac_part < 0) frac_part = -frac_part;
    
    // Convertir parte entera
    int i = 0;
    if (int_part == 0) {
        buffer[i++] = '0';
    } else {
        int temp = int_part;
        int digits = 0;
        while (temp > 0) {
            temp /= 10;
            digits++;
        }
        for (int j = digits - 1; j >= 0; j--) {
            buffer[j] = '0' + (int_part % 10);
            int_part /= 10;
        }
        i = digits;
    }
    
    // Punto decimal
    buffer[i++] = '.';
    
    // Parte fraccionaria
    for (int d = 0; d < decimals; d++) {
        frac_part *= 10;
        int digit = (int)frac_part;
        buffer[i++] = '0' + digit;
        frac_part -= digit;
    }
    
    buffer[i] = '\0';
}

// ======================== HASH 3-GRAMAS ========================
__device__ __forceinline__
uint32_t hash3(uint8_t a, uint8_t b, uint8_t c){
    uint32_t h = 2166136261u;
    h = (h ^ a) * 16777619u;
    h = (h ^ b) * 16777619u;
    h = (h ^ c) * 16777619u;
    return h % D;
}

// ======================== KERNELS NLU ========================
__global__
void tokenize3grams(const char* __restrict__ query, int n, float* __restrict__ vq){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i+2 >= n) return;
    
    uint8_t a = (query[i] >= 'A' && query[i] <= 'Z') ? query[i] + 32 : query[i];
    uint8_t b = (query[i+1] >= 'A' && query[i+1] <= 'Z') ? query[i+1] + 32 : query[i+1];
    uint8_t c = (query[i+2] >= 'A' && query[i+2] <= 'Z') ? query[i+2] + 32 : query[i+2];
    
    uint32_t idx = hash3(a, b, c);
    atomicAdd(&vq[idx], 1.0f);
}

__global__
void l2normalize(float* __restrict__ v, int d){
    __shared__ float ssum[256];
    float acc = 0.f;
    
    for (int j = threadIdx.x; j < d; j += blockDim.x){
        float x = v[j];
        acc += x*x;
    }
    ssum[threadIdx.x] = acc;
    __syncthreads();
    
    for (int offset = blockDim.x>>1; offset > 0; offset >>= 1){
        if (threadIdx.x < offset) 
            ssum[threadIdx.x] += ssum[threadIdx.x + offset];
        __syncthreads();
    }
    
    float norm = sqrtf(ssum[0] + 1e-12f);
    __syncthreads();
    
    for (int j = threadIdx.x; j < d; j += blockDim.x){
        v[j] = v[j] / norm;
    }
}

__global__
void matvecDotCos(const float* __restrict__ M, const float* __restrict__ vq,
                  float* __restrict__ scores, int K, int D){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    
    float acc = 0.f;
    for (int j = 0; j < D; ++j) 
        acc += M[k*D + j] * vq[j];
    scores[k] = acc;
}

// ======================== KERNELS SENSORES ========================
__global__
void compute_sensor_stats(const float* __restrict__ X, int N, int C, int W,
                         float* __restrict__ mean_out, 
                         float* __restrict__ std_out,
                         float* __restrict__ min_out,
                         float* __restrict__ max_out){
    int c = blockIdx.x;
    if (c >= C) return;
    
    __shared__ float ssum[256], ssum2[256];
    __shared__ float smin[256], smax[256];
    
    float sum = 0.f, sum2 = 0.f;
    float local_min = 1e20f, local_max = -1e20f;
    
    int start = max(0, N - W);
    
    for (int i = threadIdx.x; i < W; i += blockDim.x){
        float v = X[(start + i)*C + c];
        sum += v;
        sum2 += v*v;
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }
    
    ssum[threadIdx.x] = sum;
    ssum2[threadIdx.x] = sum2;
    smin[threadIdx.x] = local_min;
    smax[threadIdx.x] = local_max;
    __syncthreads();
    
    for (int off = blockDim.x>>1; off > 0; off >>= 1){
        if (threadIdx.x < off){
            ssum[threadIdx.x] += ssum[threadIdx.x + off];
            ssum2[threadIdx.x] += ssum2[threadIdx.x + off];
            smin[threadIdx.x] = fminf(smin[threadIdx.x], smin[threadIdx.x + off]);
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + off]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0){
        float m = ssum[0] / W;
        float var = fmaxf(ssum2[0]/W - m*m, 0.f);
        mean_out[c] = m;
        std_out[c] = sqrtf(var);
        min_out[c] = smin[0];
        max_out[c] = smax[0];
    }
}

__global__
void detect_anomalies(const float* __restrict__ X, int N, int C,
                     const float* __restrict__ mean, 
                     const float* __restrict__ std,
                     int* __restrict__ anomaly_count,
                     float threshold_sigmas){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    bool is_anomaly = false;
    for (int c = 0; c < C; ++c){
        float val = X[idx * C + c];
        float dev = fabsf(val - mean[c]);
        if (dev > threshold_sigmas * std[c]){
            is_anomaly = true;
            break;
        }
    }
    
    if (is_anomaly){
        atomicAdd(anomaly_count, 1);
    }
}

// ======================== KERNEL FUSIÓN/DECISIÓN ========================
__global__
void fuseDecision(const float* __restrict__ scores, int K,
                 const float* __restrict__ mean_sensors,
                 const float* __restrict__ std_sensors,
                 int* __restrict__ outDecision, 
                 int* __restrict__ outTop,
                 char* __restrict__ outMessage,
                 int max_msg_len){
    __shared__ int topIdx;
    __shared__ float topScore;
    
    if (threadIdx.x == 0){ 
        topIdx = 0; 
        topScore = scores[0]; 
    }
    __syncthreads();
    
    for (int k = threadIdx.x; k < K; k += blockDim.x){
        float s = scores[k];
        if (s > topScore){ 
            atomicMax((int*)&topScore, __float_as_int(s));
            topIdx = k; 
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0){
        *outTop = topIdx;
        int decision = 0;
        
        float hum_suelo = mean_sensors[0];
        float temp = mean_sensors[1];
        float luz = mean_sensors[2];
        float hum_aire = mean_sensors[3];
        float ph = mean_sensors[4];
        
        char buffer[32];
        outMessage[0] = '\0';
        
        switch(topIdx){
            case 0: // CONSULTAR_HUMEDAD
                decision = 1;
                d_strcpy(outMessage, "Humedad suelo: ");
                d_ftoa(hum_suelo * 100, buffer, 1);
                d_strcat(outMessage, buffer);
                d_strcat(outMessage, "%, Hum.aire: ");
                d_ftoa(hum_aire * 100, buffer, 1);
                d_strcat(outMessage, buffer);
                d_strcat(outMessage, "%");
                break;
                
            case 1: // CONSULTAR_TEMP
                decision = 1;
                d_strcpy(outMessage, "Temperatura: ");
                d_ftoa(temp, buffer, 1);
                d_strcat(outMessage, buffer);
                d_strcat(outMessage, "C (rango optimo: 20-28C)");
                break;
                
            case 2: // ACTIVAR_RIEGO
                if (hum_suelo < 0.30f && temp < 35.0f){
                    decision = 1;
                    d_strcpy(outMessage, "Riego activado. Humedad: ");
                    d_ftoa(hum_suelo * 100, buffer, 1);
                    d_strcat(outMessage, buffer);
                    d_strcat(outMessage, "%, Temp: ");
                    d_ftoa(temp, buffer, 1);
                    d_strcat(outMessage, buffer);
                    d_strcat(outMessage, "C");
                } else if (hum_suelo >= 0.30f){
                    decision = 0;
                    d_strcpy(outMessage, "Riego no necesario. Humedad: ");
                    d_ftoa(hum_suelo * 100, buffer, 1);
                    d_strcat(outMessage, buffer);
                    d_strcat(outMessage, "%");
                } else {
                    decision = 2;
                    d_strcpy(outMessage, "Temp muy alta (");
                    d_ftoa(temp, buffer, 1);
                    d_strcat(outMessage, buffer);
                    d_strcat(outMessage, "C). Riego pospuesto.");
                }
                break;
                
            case 3: // DESACTIVAR_RIEGO
                decision = 1;
                d_strcpy(outMessage, "Riego desactivado. Humedad: ");
                d_ftoa(hum_suelo * 100, buffer, 1);
                d_strcat(outMessage, buffer);
                d_strcat(outMessage, "%");
                break;
                
            case 4: // PROGRAMAR_RIEGO
                decision = 1;
                d_strcpy(outMessage, "Programacion de riego configurada");
                break;
                
            case 5: // CONSULTAR_ESTADO
                decision = 1;
                if (hum_suelo < 0.25f) {
                    d_strcpy(outMessage, "Estado: CRITICO | Hum:");
                } else if (hum_suelo < 0.35f) {
                    d_strcpy(outMessage, "Estado: BAJO | Hum:");
                } else if (hum_suelo > 0.60f) {
                    d_strcpy(outMessage, "Estado: ALTO | Hum:");
                } else {
                    d_strcpy(outMessage, "Estado: OPTIMO | Hum:");
                }
                d_ftoa(hum_suelo * 100, buffer, 1);
                d_strcat(outMessage, buffer);
                d_strcat(outMessage, "% Temp:");
                d_ftoa(temp, buffer, 1);
                d_strcat(outMessage, buffer);
                d_strcat(outMessage, "C");
                break;
                
            case 6: // ACTIVAR_VENTILACION
                if (temp > 28.0f || hum_aire > 0.75f){
                    decision = 1;
                    d_strcpy(outMessage, "Ventilacion activada. Temp: ");
                    d_ftoa(temp, buffer, 1);
                    d_strcat(outMessage, buffer);
                    d_strcat(outMessage, "C");
                } else {
                    decision = 0;
                    d_strcpy(outMessage, "Condiciones optimas. Temp: ");
                    d_ftoa(temp, buffer, 1);
                    d_strcat(outMessage, buffer);
                    d_strcat(outMessage, "C");
                }
                break;
                
            case 7: // DESACTIVAR_VENTILACION
                decision = 1;
                d_strcpy(outMessage, "Ventilacion desactivada. Temp: ");
                d_ftoa(temp, buffer, 1);
                d_strcat(outMessage, buffer);
                d_strcat(outMessage, "C");
                break;
                
            case 8: // AYUDA
                decision = 1;
                d_strcpy(outMessage, "Comandos: humedad, temp, riego, estado, diagnostico");
                break;
                
            case 9: // DIAGNOSTICO
                decision = 1;
                if (ph < 5.5f || ph > 7.5f){
                    d_strcpy(outMessage, "pH fuera de rango (");
                    d_ftoa(ph, buffer, 1);
                    d_strcat(outMessage, buffer);
                    d_strcat(outMessage, "). Ajuste necesario.");
                } else if (temp > 32.0f){
                    d_strcpy(outMessage, "Temperatura alta (");
                    d_ftoa(temp, buffer, 1);
                    d_strcat(outMessage, buffer);
                    d_strcat(outMessage, "C). Activar ventilacion.");
                } else if (hum_suelo < 0.25f){
                    d_strcpy(outMessage, "Humedad critica (");
                    d_ftoa(hum_suelo * 100, buffer, 1);
                    d_strcat(outMessage, buffer);
                    d_strcat(outMessage, "%). Riego urgente.");
                } else {
                    d_strcpy(outMessage, "Todos los parametros en rango optimo.");
                }
                break;
                
            default:
                decision = 0;
                d_strcpy(outMessage, "Comando no reconocido. Usa 'ayuda'");
        }
        
        *outDecision = decision;
    }
}

// ======================== HOST HELPERS ========================
void initIntentPrototypes(std::vector<float>& M){
    const char* training_phrases[K][5] = {
        {"cual es la humedad", "humedad del suelo", "nivel de humedad", "humedad actual", "mostrar humedad"},
        {"cual es la temperatura", "temperatura actual", "cuantos grados", "nivel temperatura", "temp del invernadero"},
        {"activa el riego", "enciende riego", "iniciar riego", "regar plantas", "activar agua"},
        {"desactiva riego", "apaga riego", "detener riego", "parar agua", "desactivar agua"},
        {"programar riego", "agenda riego", "riego automatico", "configurar riego", "horario riego"},
        {"estado general", "como esta", "reporte completo", "estado invernadero", "resumen sensores"},
        {"activa ventilacion", "enciende ventiladores", "activar aire", "iniciar ventilacion", "ventilacion on"},
        {"desactiva ventilacion", "apaga ventiladores", "detener aire", "ventilacion off", "parar ventiladores"},
        {"ayuda", "que puedes hacer", "comandos disponibles", "opciones", "help"},
        {"diagnostico", "problemas", "revisar sistema", "analizar", "detectar errores"}
    };
    
    M.resize(K * D);
    
    for (int k = 0; k < K; ++k){
        std::vector<float> avg_vector(D, 0.0f);
        
        for (int p = 0; p < 5; ++p){
            std::vector<float> phrase_vec(D, 0.0f);
            const char* phrase = training_phrases[k][p];
            int len = strlen(phrase);
            
            for (int i = 0; i + 2 < len; ++i){
                uint8_t a = (phrase[i] >= 'A' && phrase[i] <= 'Z') ? phrase[i] + 32 : phrase[i];
                uint8_t b = (phrase[i+1] >= 'A' && phrase[i+1] <= 'Z') ? phrase[i+1] + 32 : phrase[i+1];
                uint8_t c = (phrase[i+2] >= 'A' && phrase[i+2] <= 'Z') ? phrase[i+2] + 32 : phrase[i+2];
                
                uint32_t h = 2166136261u;
                h = (h ^ a) * 16777619u;
                h = (h ^ b) * 16777619u;
                h = (h ^ c) * 16777619u;
                uint32_t idx = h % D;
                
                phrase_vec[idx] += 1.0f;
            }
            
            double norm = 0;
            for (int j = 0; j < D; ++j) norm += phrase_vec[j] * phrase_vec[j];
            norm = std::sqrt(norm) + 1e-12;
            for (int j = 0; j < D; ++j) phrase_vec[j] /= norm;
            
            for (int j = 0; j < D; ++j) avg_vector[j] += phrase_vec[j];
        }
        
        double norm = 0;
        for (int j = 0; j < D; ++j) norm += avg_vector[j] * avg_vector[j];
        norm = std::sqrt(norm) + 1e-12;
        for (int j = 0; j < D; ++j) M[k*D + j] = avg_vector[j] / norm;
    }
}

void synthSensors(std::vector<float>& X){
    X.resize(size_t(N)*C);
    srand(time(NULL));
    
    for (int i = 0; i < N; ++i){
        float t = float(i) / N;
        
        float hum_suelo = 0.45f - 0.20f * t + (rand()%100)/1000.0f;
        hum_suelo = fmaxf(0.15f, fminf(0.70f, hum_suelo));
        
        float temp = 18.0f + 12.0f * sinf(t * 3.14159f) + (rand()%50 - 25)/10.0f;
        temp = fmaxf(15.0f, fminf(40.0f, temp));
        
        float luz = 200.0f + 600.0f * sinf(t * 3.14159f) + (rand()%100);
        luz = fmaxf(50.0f, fminf(1000.0f, luz));
        
        float hum_aire = 0.80f - 0.25f * (temp - 18.0f) / 12.0f + (rand()%50)/1000.0f;
        hum_aire = fmaxf(0.30f, fminf(0.95f, hum_aire));
        
        float ph = 6.5f + (rand()%20 - 10)/20.0f;
        ph = fmaxf(5.0f, fminf(8.0f, ph));
        
        X[i*C + 0] = hum_suelo;
        X[i*C + 1] = temp;
        X[i*C + 2] = luz;
        X[i*C + 3] = hum_aire;
        X[i*C + 4] = ph;
    }
}

// ======================== PROGRAMA PRINCIPAL ========================
int main(int argc, char** argv){
    printf("=== Chat-Box IA Smart Agriculture con CUDA ===\n\n");
    
    cudaStream_t sNLU, sDATA, sFUSE;
    CUDA_OK(cudaStreamCreate(&sNLU));
    CUDA_OK(cudaStreamCreate(&sDATA));
    CUDA_OK(cudaStreamCreate(&sFUSE));
    
    cudaEvent_t evStart, evStop, evNLU, evDATA;
    CUDA_OK(cudaEventCreate(&evStart));
    CUDA_OK(cudaEventCreate(&evStop));
    CUDA_OK(cudaEventCreate(&evNLU));
    CUDA_OK(cudaEventCreate(&evDATA));
    
    printf("Inicializando banco de intenciones...\n");
    std::vector<float> hM; 
    initIntentPrototypes(hM);
    float *dM = nullptr;
    CUDA_OK(cudaMalloc(&dM, K*D*sizeof(float)));
    CUDA_OK(cudaMemcpy(dM, hM.data(), K*D*sizeof(float), cudaMemcpyHostToDevice));
    
    float *hVQ = nullptr, *dVQ = nullptr;
    float *dScores = nullptr, *hScores = nullptr;
    CUDA_OK(cudaHostAlloc(&hVQ, D*sizeof(float), cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc(&hScores, K*sizeof(float), cudaHostAllocDefault));
    CUDA_OK(cudaMalloc(&dVQ, D*sizeof(float)));
    CUDA_OK(cudaMalloc(&dScores, K*sizeof(float)));
    
    printf("Generando datos sinteticos de sensores...\n");
    std::vector<float> hXvec; 
    synthSensors(hXvec);
    float *hX = nullptr;
    CUDA_OK(cudaHostAlloc(&hX, size_t(N)*C*sizeof(float), cudaHostAllocDefault));
    memcpy(hX, hXvec.data(), size_t(N)*C*sizeof(float));
    
    float *dX = nullptr;
    float *dMean = nullptr, *dStd = nullptr, *dMin = nullptr, *dMax = nullptr;
    float hMean[C], hStd[C], hMin[C], hMax[C];
    int *dAnomalyCount = nullptr, hAnomalyCount = 0;
    
    CUDA_OK(cudaMalloc(&dX, size_t(N)*C*sizeof(float)));
    CUDA_OK(cudaMalloc(&dMean, C*sizeof(float)));
    CUDA_OK(cudaMalloc(&dStd, C*sizeof(float)));
    CUDA_OK(cudaMalloc(&dMin, C*sizeof(float)));
    CUDA_OK(cudaMalloc(&dMax, C*sizeof(float)));
    CUDA_OK(cudaMalloc(&dAnomalyCount, sizeof(int)));
    
    int *dDecision = nullptr, *dTop = nullptr;
    int hDecision = 0, hTop = -1;
    char *dMessage = nullptr, *hMessage = nullptr;
    constexpr int MSG_LEN = 512;
    
    CUDA_OK(cudaMalloc(&dDecision, sizeof(int)));
    CUDA_OK(cudaMalloc(&dTop, sizeof(int)));
    CUDA_OK(cudaMalloc(&dMessage, MSG_LEN));
    CUDA_OK(cudaHostAlloc(&hMessage, MSG_LEN, cudaHostAllocDefault));
    
    const char* test_queries[] = {
        "cual es la humedad del suelo en el invernadero",
        "activa el riego porque esta muy seco",
        "cual es la temperatura actual",
        "dame el estado general del sistema",
        "diagnostica si hay algun problema",
        "enciende los ventiladores hace mucho calor",
        "apaga el riego ya esta bien",
        "ayuda"
    };
    int num_queries = 8;
    
    printf("\n=== Procesando %d consultas ===\n\n", num_queries);
    
    float total_latency = 0.0f;
    
    for (int q_idx = 0; q_idx < num_queries; ++q_idx){
        std::string query_str = test_queries[q_idx];
        int qn = std::min<int>(query_str.size(), MAX_QUERY);
        
        char *hQ = nullptr, *dQ = nullptr;
        CUDA_OK(cudaHostAlloc(&hQ, MAX_QUERY, cudaHostAllocDefault));
        CUDA_OK(cudaMalloc(&dQ, MAX_QUERY));
        memset(hQ, 0, MAX_QUERY);
        memcpy(hQ, query_str.data(), qn);
        
        printf("─────────────────────────────────────────────────\n");
        printf("Consulta #%d: \"%s\"\n", q_idx+1, query_str.c_str());
        
        CUDA_OK(cudaEventRecord(evStart, 0));
        
        CUDA_OK(cudaMemsetAsync(dVQ, 0, D*sizeof(float), sNLU));
        CUDA_OK(cudaMemcpyAsync(dQ, hQ, MAX_QUERY, cudaMemcpyHostToDevice, sNLU));
        
        dim3 blkTok(256), grdTok(ceilDiv(qn, (int)blkTok.x));
        tokenize3grams<<<grdTok, blkTok, 0, sNLU>>>(dQ, qn, dVQ);
        
        l2normalize<<<1, 256, 0, sNLU>>>(dVQ, D);
        
        dim3 blkMat(128), grdMat(ceilDiv(K, (int)blkMat.x));
        matvecDotCos<<<grdMat, blkMat, 0, sNLU>>>(dM, dVQ, dScores, K, D);
        
        CUDA_OK(cudaMemcpyAsync(hScores, dScores, K*sizeof(float), cudaMemcpyDeviceToHost, sNLU));
        CUDA_OK(cudaEventRecord(evNLU, sNLU));
        
        CUDA_OK(cudaMemcpyAsync(dX, hX, size_t(N)*C*sizeof(float), cudaMemcpyHostToDevice, sDATA));
        
        compute_sensor_stats<<<C, 256, 0, sDATA>>>(dX, N, C, W, dMean, dStd, dMin, dMax);
        
        CUDA_OK(cudaMemsetAsync(dAnomalyCount, 0, sizeof(int), sDATA));
        dim3 blkAnom(256), grdAnom(ceilDiv(N, (int)blkAnom.x));
        detect_anomalies<<<grdAnom, blkAnom, 0, sDATA>>>(dX, N, C, dMean, dStd, dAnomalyCount, 3.0f);
        
        CUDA_OK(cudaMemcpyAsync(hMean, dMean, C*sizeof(float), cudaMemcpyDeviceToHost, sDATA));
        CUDA_OK(cudaMemcpyAsync(hStd, dStd, C*sizeof(float), cudaMemcpyDeviceToHost, sDATA));
        CUDA_OK(cudaMemcpyAsync(hMin, dMin, C*sizeof(float), cudaMemcpyDeviceToHost, sDATA));
        CUDA_OK(cudaMemcpyAsync(hMax, dMax, C*sizeof(float), cudaMemcpyDeviceToHost, sDATA));
        CUDA_OK(cudaMemcpyAsync(&hAnomalyCount, dAnomalyCount, sizeof(int), cudaMemcpyDeviceToHost, sDATA));
        CUDA_OK(cudaEventRecord(evDATA, sDATA));
        
        CUDA_OK(cudaStreamWaitEvent(sFUSE, evNLU, 0));
        CUDA_OK(cudaStreamWaitEvent(sFUSE, evDATA, 0));
        
        float *dMeanCopy = nullptr;
        CUDA_OK(cudaMalloc(&dMeanCopy, C*sizeof(float)));
        CUDA_OK(cudaMemcpyAsync(dMeanCopy, hMean, C*sizeof(float), cudaMemcpyHostToDevice, sFUSE));
        
        float *dStdCopy = nullptr;
        CUDA_OK(cudaMalloc(&dStdCopy, C*sizeof(float)));
        CUDA_OK(cudaMemcpyAsync(dStdCopy, hStd, C*sizeof(float), cudaMemcpyHostToDevice, sFUSE));
        
        fuseDecision<<<1, 128, 0, sFUSE>>>(dScores, K, dMeanCopy, dStdCopy, dDecision, dTop, dMessage, MSG_LEN);
        
        CUDA_OK(cudaMemcpyAsync(&hDecision, dDecision, sizeof(int), cudaMemcpyDeviceToHost, sFUSE));
        CUDA_OK(cudaMemcpyAsync(&hTop, dTop, sizeof(int), cudaMemcpyDeviceToHost, sFUSE));
        CUDA_OK(cudaMemcpyAsync(hMessage, dMessage, MSG_LEN, cudaMemcpyDeviceToHost, sFUSE));
        
        CUDA_OK(cudaStreamSynchronize(sFUSE));
        CUDA_OK(cudaEventRecord(evStop, 0));
        CUDA_OK(cudaEventSynchronize(evStop));
        
        float ms = 0, msNLU = 0, msDATA = 0;
        CUDA_OK(cudaEventElapsedTime(&ms, evStart, evStop));
        CUDA_OK(cudaEventElapsedTime(&msNLU, evStart, evNLU));
        CUDA_OK(cudaEventElapsedTime(&msDATA, evStart, evDATA));
        
        total_latency += ms;
        
        printf("\n Intención detectada: %s\n", intentNames[hTop]);
        printf(" Scores de intenciones:\n");
        for (int k = 0; k < K; ++k){
            if (hScores[k] > 0.1f){
                printf("   %s: %.3f\n", intentNames[k], hScores[k]);
            }
        }
        
        printf("\n Métricas de sensores (ventana de %d muestras):\n", W);
        const char* sensor_names[] = {"Humedad suelo", "Temperatura", "Luz", "Humedad aire", "pH"};
        const char* units[] = {"%", "°C", "lux", "%", ""};
        for (int c = 0; c < C; ++c){
            float mean_display = (c == 0 || c == 3) ? hMean[c] * 100 : hMean[c];
            float std_display = (c == 0 || c == 3) ? hStd[c] * 100 : hStd[c];
            printf("   %s: %.2f%s (±%.2f) [min: %.2f, max: %.2f]\n", 
                   sensor_names[c], mean_display, units[c], std_display,
                   (c == 0 || c == 3) ? hMin[c]*100 : hMin[c],
                   (c == 0 || c == 3) ? hMax[c]*100 : hMax[c]);
        }
        printf("  Anomalías detectadas: %d (%.2f%%)\n", 
               hAnomalyCount, 100.0f * hAnomalyCount / W);
        
        printf("\n Respuesta del sistema:\n   %s\n", hMessage);
        
        printf("\n  Tiempos de ejecución:\n");
        printf("   - NLU: %.3f ms\n", msNLU);
        printf("   - Sensores: %.3f ms\n", msDATA);
        printf("   - Total (end-to-end): %.3f ms\n", ms);
        printf("\n");
        
        cudaFree(dQ);
        cudaFree(dMeanCopy);
        cudaFree(dStdCopy);
        cudaFreeHost(hQ);
    }
    
    printf("═════════════════════════════════════════════════\n");
    printf(" Procesamiento completado\n");
    printf(" Latencia promedio: %.3f ms\n", total_latency / num_queries);
    printf(" QPS teórico: %.1f queries/segundo\n", 1000.0f / (total_latency / num_queries));
    printf("═════════════════════════════════════════════════\n\n");
    
    printf("=== Benchmark de Escalabilidad ===\n\n");
    int stream_configs[] = {1, 2, 4, 8};
    
    for (int s_idx = 0; s_idx < 4; ++s_idx){
        int num_streams = stream_configs[s_idx];
        int queries_per_stream = 10;
        
        std::vector<cudaStream_t> streams(num_streams);
        for (int i = 0; i < num_streams; ++i){
            CUDA_OK(cudaStreamCreate(&streams[i]));
        }
        
        cudaEvent_t evBenchStart, evBenchStop;
        CUDA_OK(cudaEventCreate(&evBenchStart));
        CUDA_OK(cudaEventCreate(&evBenchStop));
        
        CUDA_OK(cudaEventRecord(evBenchStart, 0));
        
        for (int q = 0; q < queries_per_stream; ++q){
            for (int s = 0; s < num_streams; ++s){
                cudaStream_t stream = streams[s];
                
                char *dQ_bench = nullptr;
                float *dVQ_bench = nullptr, *dScores_bench = nullptr;
                
                CUDA_OK(cudaMalloc(&dQ_bench, MAX_QUERY));
                CUDA_OK(cudaMalloc(&dVQ_bench, D*sizeof(float)));
                CUDA_OK(cudaMalloc(&dScores_bench, K*sizeof(float)));
                
                CUDA_OK(cudaMemsetAsync(dVQ_bench, 0, D*sizeof(float), stream));
                
                dim3 blk(256), grd(ceilDiv(50, 256));
                tokenize3grams<<<grd, blk, 0, stream>>>(dQ_bench, 50, dVQ_bench);
                l2normalize<<<1, 256, 0, stream>>>(dVQ_bench, D);
                
                dim3 blkM(128), grdM(ceilDiv(K, 128));
                matvecDotCos<<<grdM, blkM, 0, stream>>>(dM, dVQ_bench, dScores_bench, K, D);
                
                cudaFree(dQ_bench);
                cudaFree(dVQ_bench);
                cudaFree(dScores_bench);
            }
        }
        
        for (int i = 0; i < num_streams; ++i){
            CUDA_OK(cudaStreamSynchronize(streams[i]));
        }
        
        CUDA_OK(cudaEventRecord(evBenchStop, 0));
        CUDA_OK(cudaEventSynchronize(evBenchStop));
        
        float bench_ms = 0;
        CUDA_OK(cudaEventElapsedTime(&bench_ms, evBenchStart, evBenchStop));
        
        int total_queries = num_streams * queries_per_stream;
        float qps = (total_queries * 1000.0f) / bench_ms;
        
        printf("Streams: %d | Queries: %d | Tiempo: %.2f ms | QPS: %.1f\n",
               num_streams, total_queries, bench_ms, qps);
        
        for (int i = 0; i < num_streams; ++i){
            CUDA_OK(cudaStreamDestroy(streams[i]));
        }
        CUDA_OK(cudaEventDestroy(evBenchStart));
        CUDA_OK(cudaEventDestroy(evBenchStop));
    }
    
    printf("\n=== Recomendaciones de Integración ===\n");
    printf("1. Conectar con dashboard web (Node.js/Flask/FastAPI)\n");
    printf("2. Integrar con MQTT broker para comandos remotos\n");
    printf("3. Almacenar logs en CSV/JSON o BD (SQLite/PostgreSQL)\n");
    printf("4. Conectar GPIO para actuadores físicos (Raspberry Pi)\n");
    printf("5. Agregar WebSocket para respuestas en tiempo real\n\n");
    
    cudaFree(dM); cudaFree(dVQ); cudaFree(dScores);
    cudaFree(dX); cudaFree(dMean); cudaFree(dStd); cudaFree(dMin); cudaFree(dMax);
    cudaFree(dAnomalyCount);
    cudaFree(dDecision); cudaFree(dTop); cudaFree(dMessage);
    
    cudaFreeHost(hVQ); cudaFreeHost(hScores); cudaFreeHost(hX); cudaFreeHost(hMessage);
    
    cudaEventDestroy(evStart); cudaEventDestroy(evStop);
    cudaEventDestroy(evNLU); cudaEventDestroy(evDATA);
    cudaStreamDestroy(sNLU); cudaStreamDestroy(sDATA);
    cudaStreamDestroy(sFUSE);
    
    printf(" Programa finalizado exitosamente\n");
    return 0;
}