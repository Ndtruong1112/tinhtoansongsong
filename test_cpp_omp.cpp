#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <omp.h>

#pragma pack(push, 1)
struct TriangleBinary {
    float normal[3];
    float v0[3];
    float v1[3];
    float v2[3];
    uint16_t attr;
};
#pragma pack(pop)

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file.stl>\n";
        return 1;
    }

    std::ifstream file(argv[1], std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << argv[1] << "\n";
        return 1;
    }

    char header[80];
    file.read(header, 80);

    uint32_t num_triangles = 0;
    file.read(reinterpret_cast<char*>(&num_triangles), sizeof(uint32_t));

    std::vector<TriangleBinary> triangles(num_triangles);
    file.read(reinterpret_cast<char*>(triangles.data()), num_triangles * sizeof(TriangleBinary));
    file.close();

    // 1. Single Thread CPU Baseline
    auto t0 = std::chrono::high_resolution_clock::now();
    double total_seq = 0.0;
    for (uint32_t i = 0; i < num_triangles; ++i) {
        const auto& t = triangles[i];
        double abx = t.v1[0] - t.v0[0], aby = t.v1[1] - t.v0[1], abz = t.v1[2] - t.v0[2];
        double acx = t.v2[0] - t.v0[0], acy = t.v2[1] - t.v0[1], acz = t.v2[2] - t.v0[2];
        double cx = aby * acz - abz * acy;
        double cy = abz * acx - abx * acz;
        double cz = abx * acy - aby * acx;
        total_seq += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_seq_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2. Multi-Thread CPU Max Power (OpenMP 16 Threads + SIMD)
    auto t2 = std::chrono::high_resolution_clock::now();
    double total_omp = 0.0;
    #pragma omp parallel for reduction(+:total_omp) schedule(static)
    for (int i = 0; i < (int)num_triangles; ++i) {
        const auto& t = triangles[i];
        double abx = t.v1[0] - t.v0[0], aby = t.v1[1] - t.v0[1], abz = t.v1[2] - t.v0[2];
        double acx = t.v2[0] - t.v0[0], acy = t.v2[1] - t.v0[1], acz = t.v2[2] - t.v0[2];
        double cx = aby * acz - abz * acy;
        double cy = abz * acx - abx * acz;
        double cz = abx * acy - aby * acx;
        total_omp += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double time_omp_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "TRIANGLES=" << num_triangles
              << " | SEQ_MS=" << time_seq_ms
              << " | OMP_MS=" << time_omp_ms
              << " | AREA=" << total_seq << "\n";

    return 0;
}
