#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    float normal[3];
    float v0[3];
    float v1[3];
    float v2[3];
} Triangle;

double calculate_triangle_area(Triangle t) {
    double ab[3] = {t.v1[0] - t.v0[0], t.v1[1] - t.v0[1], t.v1[2] - t.v0[2]};
    double ac[3] = {t.v2[0] - t.v0[0], t.v2[1] - t.v0[1], t.v2[2] - t.v0[2]};

    double cx = ab[1] * ac[2] - ab[2] * ac[1];
    double cy = ab[2] * ac[0] - ab[0] * ac[2];
    double cz = ab[0] * ac[1] - ab[1] * ac[0];

    double cross_len = cx * cx + cy * cy + cz * cz;
    return 0.5 * sqrt(cross_len);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Loi: Vui long truyen duong dan file STL!\n");
        printf("Cach dung: %s <duong_dan_file.stl>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        printf("Loi: Khong tim thay hoac khong mo duoc file '%s'\n", argv[1]);
        return 1;
    }

    char header[81];
    if (fread(header, 1, 80, f) != 80) {
        printf("Loi: File '%s' khong dung dinh dang STL nhiphani!\n", argv[1]);
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

    Triangle *h_triangles = (Triangle *)malloc(num_triangles * sizeof(Triangle));
    if (!h_triangles) {
        printf("Loi: Khong du bo nho RAM!\n");
        fclose(f);
        return 1;
    }

    double total_area = 0.0;
    for (uint32_t i = 0; i < num_triangles; i++) {
        fread(h_triangles[i].normal, sizeof(float), 3, f);
        fread(h_triangles[i].v0, sizeof(float), 3, f);
        fread(h_triangles[i].v1, sizeof(float), 3, f);
        fread(h_triangles[i].v2, sizeof(float), 3, f);

        uint16_t attr;
        fread(&attr, sizeof(uint16_t), 1, f);

        total_area += calculate_triangle_area(h_triangles[i]);
    }
    fclose(f);

    printf("Tong dien tich be mat = %.6f\n", total_area);
    free(h_triangles);
    return 0;
}
