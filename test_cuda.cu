#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <filesystem>
#include <algorithm>

// ============================================================
// CUDA error checking macro
// ============================================================

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr,                                                    \
                "CUDA error at %s:%d: %s\n",                               \
                __FILE__, __LINE__,                                        \
                cudaGetErrorString(err));                                  \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

// ============================================================
// Host SoA mesh stored in PINNED memory
//
// Memory layout: [x0][y0][z0][x1][y1][z1][x2][y2][z2]
// All arrays live in ONE contiguous allocation.
// ============================================================

struct HostMeshSoA {
    size_t n = 0;

    float *base = nullptr;

    float *x0 = nullptr;
    float *y0 = nullptr;
    float *z0 = nullptr;

    float *x1 = nullptr;
    float *y1 = nullptr;
    float *z1 = nullptr;

    float *x2 = nullptr;
    float *y2 = nullptr;
    float *z2 = nullptr;

    void allocate(size_t count)
    {
        n = count;

        const size_t totalFloats = 9 * n;
        const size_t bytes = totalFloats * sizeof(float);

        CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void **>(&base),
            bytes,
            cudaHostAllocDefault
        ));

        x0 = base + 0 * n;
        y0 = base + 1 * n;
        z0 = base + 2 * n;

        x1 = base + 3 * n;
        y1 = base + 4 * n;
        z1 = base + 5 * n;

        x2 = base + 6 * n;
        y2 = base + 7 * n;
        z2 = base + 8 * n;
    }

    ~HostMeshSoA()
    {
        if (base)
            cudaFreeHost(base);
    }
};

// ============================================================
// Read little-endian uint32
// ============================================================

static uint32_t read_u32_le(const unsigned char b[4])
{
    return
        static_cast<uint32_t>(b[0])       |
        static_cast<uint32_t>(b[1]) << 8  |
        static_cast<uint32_t>(b[2]) << 16 |
        static_cast<uint32_t>(b[3]) << 24;
}

// ============================================================
// Copy float from STL byte buffer
// ============================================================

static inline float load_float(const unsigned char *p)
{
    float value;
    std::memcpy(&value, p, sizeof(float));
    return value;
}

// ============================================================
// Binary STL loader
// Read STL in large 65,536 triangle chunks instead of tiny fread calls.
// Discard normal vector and attribute byte count.
// ============================================================

bool load_binary_stl(
    const char *filename,
    HostMeshSoA &mesh)
{
    FILE *f = std::fopen(filename, "rb");

    if (!f) {
        std::fprintf(stderr,
                     "Khong mo duoc file: %s\n",
                     filename);
        return false;
    }

    unsigned char header[80];

    if (std::fread(header, 1, 80, f) != 80) {
        std::fprintf(stderr, "STL header khong hop le.\n");
        std::fclose(f);
        return false;
    }

    unsigned char countBytes[4];

    if (std::fread(countBytes, 1, 4, f) != 4) {
        std::fprintf(stderr,
                     "Khong doc duoc triangle count.\n");
        std::fclose(f);
        return false;
    }

    uint32_t triangleCount = read_u32_le(countBytes);

    if (triangleCount == 0) {
        std::fprintf(stderr,
                     "STL khong co triangle.\n");
        std::fclose(f);
        return false;
    }

    mesh.allocate(triangleCount);

    constexpr size_t CHUNK_TRIANGLES = 65536;
    constexpr size_t FACET_BYTES = 50;

    std::vector<unsigned char> buffer(
        CHUNK_TRIANGLES * FACET_BYTES
    );

    size_t done = 0;

    while (done < mesh.n) {

        size_t count =
            std::min(
                CHUNK_TRIANGLES,
                mesh.n - done
            );

        size_t got =
            std::fread(
                buffer.data(),
                FACET_BYTES,
                count,
                f
            );

        if (got != count) {
            std::fprintf(stderr,
                         "STL bi thieu du lieu triangle.\n");
            std::fclose(f);
            return false;
        }

        for (size_t j = 0; j < count; ++j) {

            const unsigned char *p =
                buffer.data() + j * FACET_BYTES;

            size_t i = done + j;

            // bytes 0..11 = normal -> skip

            mesh.x0[i] = load_float(p + 12);
            mesh.y0[i] = load_float(p + 16);
            mesh.z0[i] = load_float(p + 20);

            mesh.x1[i] = load_float(p + 24);
            mesh.y1[i] = load_float(p + 28);
            mesh.z1[i] = load_float(p + 32);

            mesh.x2[i] = load_float(p + 36);
            mesh.y2[i] = load_float(p + 40);
            mesh.z2[i] = load_float(p + 44);

            // bytes 48..49 = attribute -> skip
        }

        done += count;
    }

    std::fclose(f);

    return true;
}

// ============================================================
// CPU reference calculation
// ============================================================

double cpu_surface_area(const HostMeshSoA &m)
{
    double total = 0.0;

    for (size_t i = 0; i < m.n; ++i) {

        double abx =
            static_cast<double>(m.x1[i]) - m.x0[i];

        double aby =
            static_cast<double>(m.y1[i]) - m.y0[i];

        double abz =
            static_cast<double>(m.z1[i]) - m.z0[i];

        double acx =
            static_cast<double>(m.x2[i]) - m.x0[i];

        double acy =
            static_cast<double>(m.y2[i]) - m.y0[i];

        double acz =
            static_cast<double>(m.z2[i]) - m.z0[i];

        double cx =
            aby * acz - abz * acy;

        double cy =
            abz * acx - abx * acz;

        double cz =
            abx * acy - aby * acx;

        total +=
            0.5 *
            std::sqrt(
                cx * cx +
                cy * cy +
                cz * cz
            );
    }

    return total;
}

// ============================================================
// Warp reduction via __shfl_down_sync
// ============================================================

template<typename T>
__inline__ __device__
T warp_reduce_sum(T value)
{
    constexpr unsigned FULL_MASK = 0xffffffffu;

    for (int offset = 16;
         offset > 0;
         offset >>= 1)
    {
        value +=
            __shfl_down_sync(
                FULL_MASK,
                value,
                offset
            );
    }

    return value;
}

// ============================================================
// Block reduction using shared memory for warp sums
// ============================================================

template<int BLOCK_SIZE, typename T>
__inline__ __device__
T block_reduce_sum(T value)
{
    static_assert(
        BLOCK_SIZE % 32 == 0,
        "BLOCK_SIZE must be multiple of warp size"
    );

    constexpr int NUM_WARPS =
        BLOCK_SIZE / 32;

    __shared__ T warpSums[NUM_WARPS];

    int lane =
        threadIdx.x & 31;

    int warp =
        threadIdx.x >> 5;

    value =
        warp_reduce_sum(value);

    if (lane == 0)
        warpSums[warp] = value;

    __syncthreads();

    if (warp == 0) {

        value =
            (lane < NUM_WARPS)
            ? warpSums[lane]
            : static_cast<T>(0);

        value =
            warp_reduce_sum(value);
    }

    return value;
}

// ============================================================
// Main optimized CUDA Kernel
// Optimizations: SoA, __restrict__, grid-stride, register sum,
// no area[N] global array, fmaf, warp shuffle reduction.
// ============================================================

template<int BLOCK_SIZE>
__global__
void triangle_area_partial(
    const float *__restrict__ data,
    size_t n,
    float *__restrict__ partialSums)
{
    const float *x0 = data + 0 * n;
    const float *y0 = data + 1 * n;
    const float *z0 = data + 2 * n;

    const float *x1 = data + 3 * n;
    const float *y1 = data + 4 * n;
    const float *z1 = data + 5 * n;

    const float *x2 = data + 6 * n;
    const float *y2 = data + 7 * n;
    const float *z2 = data + 8 * n;

    size_t index =
        static_cast<size_t>(blockIdx.x) *
        BLOCK_SIZE +
        threadIdx.x;

    size_t stride =
        static_cast<size_t>(gridDim.x) *
        BLOCK_SIZE;

    float localSum = 0.0f;

    for (size_t i = index;
         i < n;
         i += stride)
    {
        float ax = x0[i];
        float ay = y0[i];
        float az = z0[i];

        float abx = x1[i] - ax;
        float aby = y1[i] - ay;
        float abz = z1[i] - az;

        float acx = x2[i] - ax;
        float acy = y2[i] - ay;
        float acz = z2[i] - az;

        float cx =
            fmaf(aby, acz, -abz * acy);

        float cy =
            fmaf(abz, acx, -abx * acz);

        float cz =
            fmaf(abx, acy, -aby * acx);

        float lengthSquared =
            fmaf(
                cx, cx,
                fmaf(
                    cy, cy,
                    cz * cz
                )
            );

        float area =
            0.5f *
            sqrtf(lengthSquared);

        localSum += area;
    }

    float blockSum =
        block_reduce_sum<
            BLOCK_SIZE,
            float
        >(localSum);

    if (threadIdx.x == 0)
        partialSums[blockIdx.x] =
            blockSum;
}

// ============================================================
// Final reduction kernel (FP64 precision)
// ============================================================

template<int BLOCK_SIZE>
__global__
void final_reduce(
    const float *__restrict__ input,
    int count,
    double *__restrict__ result)
{
    double localSum = 0.0;

    for (int i = threadIdx.x;
         i < count;
         i += BLOCK_SIZE)
    {
        localSum +=
            static_cast<double>(
                input[i]
            );
    }

    double sum =
        block_reduce_sum<
            BLOCK_SIZE,
            double
        >(localSum);

    if (threadIdx.x == 0)
        *result = sum;
}

// ============================================================
// main entry point
// ============================================================

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::printf(
            "Loi: Vui long truyen file STL!\nCach dung: %s model.stl\n",
            argv[0]
        );

        return 1;
    }

    // Initialize CUDA early
    CUDA_CHECK(cudaFree(nullptr));

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf("========================================\n");
    std::printf("GPU: %s\n", prop.name);
    std::printf("Compute Capability: %d.%d | SM Count: %d\n", prop.major, prop.minor, prop.multiProcessorCount);
    std::printf("========================================\n");

    // Load STL mesh
    HostMeshSoA mesh;
    if (!load_binary_stl(argv[1], mesh)) {
        return 1;
    }

    std::printf("File STL: %s\n", argv[1]);
    std::printf("So luong tam giac: %zu\n", mesh.n);

    // CPU baseline measurement
    auto cpuStart = std::chrono::high_resolution_clock::now();
    double cpuArea = cpu_surface_area(mesh);
    auto cpuStop = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();

    // Allocate GPU memory
    size_t dataBytes = 9 * mesh.n * sizeof(float);
    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_data), dataBytes));

    constexpr int BLOCK_SIZE = 256;

    int activeBlocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        triangle_area_partial<BLOCK_SIZE>,
        BLOCK_SIZE,
        0
    ));

    size_t blocksNeeded = (mesh.n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t occupancyBlocks = static_cast<size_t>(activeBlocksPerSM) * prop.multiProcessorCount;
    int blocks = static_cast<int>(std::min(blocksNeeded, occupancyBlocks));
    if (blocks < 1) blocks = 1;

    std::printf("Config: BlockSize = %d | GridBlocks = %d\n", BLOCK_SIZE, blocks);

    float *d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_partial), blocks * sizeof(float)));

    double *d_result = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_result), sizeof(double)));

    double *h_result = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&h_result), sizeof(double), cudaHostAllocDefault));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // H2D Transfer Timing
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_data, mesh.base, dataBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2dMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2dMs, start, stop));

    // Warm-up run
    triangle_area_partial<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(d_data, mesh.n, d_partial);
    final_reduce<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(d_partial, blocks, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark runs
    constexpr int RUNS = 20;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int r = 0; r < RUNS; ++r) {
        triangle_area_partial<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(d_data, mesh.n, d_partial);
        final_reduce<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(d_partial, blocks, d_result);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelTotalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTotalMs, start, stop));
    float kernelMs = kernelTotalMs / RUNS;

    // D2H Transfer Timing
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2hMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2hMs, start, stop));

    // Results & Comparison
    double gpuArea = *h_result;
    double absError = std::fabs(cpuArea - gpuArea);
    double relativeError = (cpuArea != 0.0) ? absError / std::fabs(cpuArea) : 0.0;
    double gpuPipelineMs = static_cast<double>(h2dMs) + static_cast<double>(kernelMs) + static_cast<double>(d2hMs);

    std::printf("\n=== KET QUA DIEN TICH BE MAT ===\n");
    std::printf("CPU Area       : %.10f\n", cpuArea);
    std::printf("GPU Area       : %.10f\n", gpuArea);
    std::printf("Absolute Error : %.10e\n", absError);
    std::printf("Relative Error : %.10e\n", relativeError);

    std::printf("\n=== THOI GIAN THUC THI & SPEEDUP ===\n");
    std::printf("CPU Compute    : %.4f ms\n", cpuMs);
    std::printf("GPU H2D        : %.4f ms\n", h2dMs);
    std::printf("GPU Kernel     : %.4f ms\n", kernelMs);
    std::printf("GPU D2H        : %.4f ms\n", d2hMs);
    std::printf("GPU Pipeline   : %.4f ms\n", gpuPipelineMs);

    if (kernelMs > 0.0f) {
        std::printf("Compute Speedup: %.2fx\n", cpuMs / kernelMs);
    }
    if (gpuPipelineMs > 0.0) {
        std::printf("Pipeline Speedup: %.2fx\n", cpuMs / gpuPipelineMs);
    }
    std::printf("========================================\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_result));

    return 0;
}
