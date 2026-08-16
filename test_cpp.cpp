#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdint>

struct Triangle {
    float normal[3];
    float v0[3];
    float v1[3];
    float v2[3];
};

double calculate_triangle_area(const Triangle &t) {
    double ab[3] = {t.v1[0] - t.v0[0], t.v1[1] - t.v0[1], t.v1[2] - t.v0[2]};
    double ac[3] = {t.v2[0] - t.v0[0], t.v2[1] - t.v0[1], t.v2[2] - t.v0[2]};

    double cx = ab[1] * ac[2] - ab[2] * ac[1];
    double cy = ab[2] * ac[0] - ab[0] * ac[2];
    double cz = ab[0] * ac[1] - ab[1] * ac[0];

    double cross_len = cx * cx + cy * cy + cz * cz;
    return 0.5 * std::sqrt(cross_len);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Loi: Vui long truyen duong dan file STL!\n";
        std::cerr << "Cach dung: " << argv[0] << " <duong_dan_file.stl>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Loi: Khong tim thay hoac khong mo duoc file '" << filename << "'\n";
        return 1;
    }

    char header[80];
    if (!file.read(header, 80)) {
        std::cerr << "Loi: File '" << filename << "' khong dung dinh dang STL nhiphani!\n";
        return 1;
    }

    uint32_t num_triangles = 0;
    if (!file.read(reinterpret_cast<char *>(&num_triangles), sizeof(uint32_t))) {
        std::cerr << "Loi: Khong the doc so luong tam giac trong file '" << filename << "'!\n";
        return 1;
    }

    std::vector<Triangle> triangles(num_triangles);
    double total_area = 0.0;

    for (uint32_t i = 0; i < num_triangles; ++i) {
        Triangle t;
        file.read(reinterpret_cast<char *>(t.normal), sizeof(float) * 3);
        file.read(reinterpret_cast<char *>(t.v0), sizeof(float) * 3);
        file.read(reinterpret_cast<char *>(t.v1), sizeof(float) * 3);
        file.read(reinterpret_cast<char *>(t.v2), sizeof(float) * 3);

        uint16_t attr = 0;
        file.read(reinterpret_cast<char *>(&attr), sizeof(uint16_t));

        total_area += calculate_triangle_area(t);
    }

    std::cout << "Tong dien tich be mat = " << total_area << "\n";
    return 0;
}
