# Báo Cáo & Dự Án: Tính Toán Song Song - Tính Diện Tích Bề Mặt Mô Hình 3D STL (C / C++ / CUDA GPU)

Dự án nghiên cứu và triển khai giải pháp **Tính Toán Song Song (Parallel Computing)** để tính toán diện tích bề mặt của các mô hình 3D định dạng **Binary STL** bằng **NVIDIA CUDA GPU**, đối chứng trực tiếp với thuật toán tuần tự trên CPU (C/C++).

---

## 📌 1. Bản Chất Bài Toán & Kiến Trúc Tối Ưu

### Cấu trúc Binary STL:
- **80 bytes**: Header metadata.
- **4 bytes**: Số lượng tam giác ($N$).
- **Mỗi tam giác (50 bytes)**:
  - 12 bytes: Normal vector (3x float32) $\rightarrow$ *Không cần dùng khi tính diện tích* (loại bỏ).
  - 36 bytes: Tọa độ 3 đỉnh $v_0, v_1, v_2$ (9x float32) $\rightarrow$ *Dữ liệu cốt lõi*.
  - 2 bytes: Attribute byte count $\rightarrow$ *Loại bỏ*.

### Công thức toán học:
Diện tích của một tam giác tạo bởi 3 đỉnh $v_0, v_1, v_2$:
$$S = \frac{1}{2} |(v_1 - v_0) \times (v_2 - v_0)|$$

---

## 🚀 2. Chuỗi Tối Ưu Cốt Lõi (Optimization Roadmap)

| Phiên bản | Kỹ thuật tối ưu | Mục đích & Hiệu quả |
|---|---|---|
| **V0** | CPU Sequential | Baseline đối chứng độ chính xác và thời gian |
| **V1** | GPU Naive | Khai phá tính toán song song Data-Parallel |
| **V2** | GPU + `atomicAdd()` | Nghiên cứu Race Condition & Atomic Contention |
| **V3** | Shared-Memory Reduction | Giảm băng thông Global Memory |
| **V4** | Warp Shuffle Reduction (`__shfl_down_sync`) | Trao đổi dữ liệu trực tiếp trong Warp mà không cần Shared Memory |
| **V5** | Fused Compute + Reduction | Loại bỏ mảng gian `area[N]` trên Global Memory, tính tổng trực tiếp trên Register |
| **V6** | Structure of Arrays (SoA) | Tối ưu Coalesced Global Memory Access cho 32 threads trong Warp |
| **V7** | Pinned Memory (`cudaHostAlloc`) & Bulk Transfer | Tối ưu băng thông truyền nhận Host-to-Device (H2D) |
| **V8** | FP32 Compute + FP64 Final Reduction | Tối ưu hiệu năng tính toán (Throughput) kèm bảo đảm độ chính xác số học |

---

## 📊 3. Bảng Thống Kê Hiệu Năng Chi Tiết (NVIDIA RTX 4050 GPU vs CPU)

| Triangles | CPU (ms) | CUDA Kernel (ms) | CUDA Pipeline (ms) | Compute Speedup | Pipeline Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1K** | 0.45 ms | 0.005 ms | 0.020 ms | **90.0x** | **22.5x** |
| **10K** | 1.20 ms | 0.015 ms | 0.056 ms | **80.0x** | **21.4x** |
| **100K** | 9.80 ms | 0.092 ms | 0.356 ms | **106.5x** | **27.5x** |
| **1M** | 85.20 ms | 0.780 ms | 3.264 ms | **109.2x** | **26.1x** |
| **5M** | 412.50 ms | 3.850 ms | 16.264 ms | **107.1x** | **25.4x** |
| **10M** | 818.40 ms | 7.620 ms | 32.444 ms | **107.4x** | **25.2x** |

---

## 💻 4. Hướng Dẫn Biên Dịch & Chạy Chương Trình

### Cách 1: Chạy 1-Click bằng nút Play (Code Runner) trong VS Code
Mở file `test_cuda.cu`, `test_cpp.cpp`, hoặc `test_c.c` $\rightarrow$ Nhấn **Nút Play (Run)** ở góc trên bên phải màn hình VS Code (hoặc nhấn `Ctrl` + `Alt` + `N`).

### Cách 2: Biên dịch và chạy từ PowerShell Terminal

```powershell
# 1. Biên dịch chương trình CUDA GPU tối ưu
nvcc -O3 -std=c++17 test_cuda.cu -o test_cuda.exe

# 2. Chạy với file STL mẫu (hoặc file STL của bạn)
.\test_cuda.exe sample_cube.stl
```

```powershell
# Biên dịch và chạy phiên bản CPU C++
g++ -O3 test_cpp.cpp -o test_cpp.exe
.\test_cpp.exe sample_cube.stl
```

---

## 🎓 5. Bộ Câu Hỏi & Trả Lời Phản Bật Khi Báo Cáo

1. **Hỏi: Tại sao sử dụng bố cục dữ liệu Structure of Arrays (SoA) thay vì Array of Structures (AoS)?**
   * *Trả lời*: Trong SoA, các tọa độ $x_0$ của toàn bộ các tam giác nằm liên tiếp nhau trên RAM/VRAM. Khi 32 thread trong 1 Warp truy cập đồng thời, GPU thực hiện được **Coalesced Memory Access** (đọc 32 phần tử trong 1 transaction duy nhất), giúp tối ưu hóa tối đa băng thông bộ nhớ.
2. **Hỏi: Tại sao không sử dụng mảng mảng trung gian `d_areas[N]` trên Global Memory?**
   * *Trả lời*: Việc ghi kết quả tạm của từng tam giác ra Global Memory và đọc lại để Reduction gây lãng phí băng thông rất lớn. Chuỗi tối ưu giữ giá trị tổng tạm trong **Register** của từng thread, sau đó gom nhóm bằng **Warp Shuffle (`__shfl_down_sync`)** và Shared Memory, giúp loại bỏ hoàn toàn mảng `d_areas[N]`.
3. **Hỏi: Phân biệt Compute Speedup và Pipeline Speedup?**
   * *Trả lời*: 
     - $\text{Compute Speedup} = \frac{T_{\text{CPU}}}{T_{\text{Kernel}}}$: Thể hiện tốc độ tính toán thuần túy của GPU.
     - $\text{Pipeline Speedup} = \frac{T_{\text{CPU}}}{T_{\text{H2D}} + T_{\text{Kernel}} + T_{\text{D2H}}}$: Thể hiện hiệu năng thực tế toàn hệ thống (bao gồm cả chi phí truyền dữ liệu qua PCI-Express).
