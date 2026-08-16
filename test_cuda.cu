#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Cấu trúc tam giác giữ nguyên
typedef struct {
    float normal[3];
    float v0[3];
    float v1[3];
    float v2[3];
} Triangle;

// Khai báo hàm Kernel __global__ chạy trên GPU
__global__ void calculate_areas(Triangle *d_triangles, double *d_areas, int num_triangles) {
    // Thiết lập công thức tính chỉ số (i) của dữ liệu 1D
    int i = threadIdx.x + blockIdx.x * blockDim.x; 

    if (i < num_triangles) {
        Triangle t = d_triangles[i];
        
        // Tính vector cạnh
        double ab[3] = {t.v1[0]-t.v0[0], t.v1[1]-t.v0[1], t.v1[2]-t.v0[2]};
        double ac[3] = {t.v2[0]-t.v0[0], t.v2[1]-t.v0[1], t.v2[2]-t.v0[2]};

        // Tích có hướng
        double cx = ab[1]*ac[2] - ab[2]*ac[1];
        double cy = ab[2]*ac[0] - ab[0]*ac[2];
        double cz = ab[0]*ac[1] - ab[1]*ac[0];

        // Dùng hàm sqrt có sẵn của CUDA thay vì lặp Newton để tối ưu trên GPU
        double cross_len = cx*cx + cy*cy + cz*cz;
        d_areas[i] = 0.5 * sqrt(cross_len);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Loi: Vui long truyen duong dan file STL!\n");
        printf("Cach dung: %s <duong_dan_file.stl>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        printf("Loi: Khong tim thấy hoac khong mo duoc file '%s'\n", argv[1]);
        return 1;
    }

    char header[81];
    if (fread(header, 1, 80, f) != 80) {
        printf("Loi: File '%s' khong dung dinh dang STL nhiphani (Header loi)!\n", argv[1]);
        fclose(f);
        return 1;
    }
    header[80] = '\0';
    
    uint32_t num_triangles;
    if (fread(&num_triangles, sizeof(uint32_t), 1, f) != 1) {
        printf("Loi: Khong the doc so luong tam giac trong file '%s'!\n", argv[1]);
        fclose(f);
        return 1;
    }

    size_t tri_size = num_triangles * sizeof(Triangle);
    size_t area_size = num_triangles * sizeof(double);

    // 1. Cấp phát bộ nhớ trên Host (CPU) và đọc toàn bộ file vào mảng
    Triangle *h_triangles = (Triangle *)malloc(tri_size);
    double *h_areas = (double *)malloc(area_size);
    if (!h_triangles || !h_areas) {
        printf("Loi: Khong du bộ nho RAM de xu ly %u tam giac!\n", num_triangles);
        fclose(f);
        return 1;
    }
    
    for (uint32_t i = 0; i < num_triangles; i++) {
        fread(h_triangles[i].normal, sizeof(float), 3, f);
        fread(h_triangles[i].v0, sizeof(float), 3, f);
        fread(h_triangles[i].v1, sizeof(float), 3, f);
        fread(h_triangles[i].v2, sizeof(float), 3, f);
        
        uint16_t attr;
        fread(&attr, sizeof(uint16_t), 1, f);
    }
    fclose(f);

    // 2. Gọi cudaMalloc cấp phát bộ nhớ trên GPU
    Triangle *d_triangles;
    double *d_areas;
    cudaMalloc((void **)&d_triangles, tri_size);
    cudaMalloc((void **)&d_areas, area_size);

    // 3. Đẩy dữ liệu từ Host sang Device
    cudaMemcpy(d_triangles, h_triangles, tri_size, cudaMemcpyHostToDevice);

    // 4. Thiết lập số lượng Block/Thread và gọi Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_triangles + threadsPerBlock - 1) / threadsPerBlock;
    calculate_areas<<<blocksPerGrid, threadsPerBlock>>>(d_triangles, d_areas, num_triangles);

    cudaDeviceSynchronize();

    // 5. Trả mảng kết quả diện tích về lại Host
    cudaMemcpy(h_areas, d_areas, area_size, cudaMemcpyDeviceToHost);

    // 6. Tính tổng cuối cùng trên CPU
    double total_area = 0.0;
    for (uint32_t i = 0; i < num_triangles; i++) {
        total_area += h_areas[i];
    }
    printf("Tong dien tich be mat = %.6f\n", total_area);

    // 7. Gọi cudaFree dọn dẹp bộ nhớ
    free(h_triangles);
    free(h_areas);
    cudaFree(d_triangles);
    cudaFree(d_areas);

    return 0;
}
