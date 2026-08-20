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

## 🛠️ 3. Môi Trường Cần Thiết & Cấu Hình VS Code ("Visual Xanh")

- **Hệ điều hành**: Windows 10/11 x64
- **Card đồ họa**: NVIDIA GPU (Hỗ trợ CUDA Compute Capability 5.0+)
- **Bộ biên dịch C/C++**: WinLibs MinGW-w64 GCC/G++ v16.1.0
- **CUDA Toolkit**: NVIDIA CUDA Toolkit v13.3 (`nvcc`)
- **Host Compiler**: Microsoft Visual C++ Build Tools (`cl.exe`)
- **VS Code Extensions**:
  - `ms-vscode.cpptools` (C/C++ Intellisense & Debugging)
  - `NVIDIA.nsight-vscode-edition` (Cú pháp & Debugging CUDA)
  - `formulahendry.code-runner` (1-Click Run)

---

## 💻 4. Hướng Dẫn Biên Dịch & Chạy Chương Trình

### Cách 1: Chạy 1-Click bằng nút Play (Code Runner) trong VS Code
Mở file `test_cuda.cu`, `test_cpp.cpp`, hoặc `test_c.c` $\rightarrow$ Nhấn **Nút Play (Run)** ở góc trên bên phải màn hình VS Code (hoặc bấm `Ctrl` + `Alt` + `N`).

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

## 📊 5. Kết Quả Chạy Mẫu & Đánh Giá Performance

```text
========================================
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
Compute Capability: 8.9 | SM Count: 20
========================================
File STL: sample_cube.stl
So luong tam giac: 12
Config: BlockSize = 256 | GridBlocks = 20

=== KET QUA DIEN TICH BE MAT ===
CPU Area       : 24.0000000000
GPU Area       : 24.0000000000
Absolute Error : 0.0000000000e+00
Relative Error : 0.0000000000e+00

=== THOI GIAN THUC THI & SPEEDUP ===
CPU Compute    : 0.0034 ms
GPU H2D        : 0.0210 ms
GPU Kernel     : 0.0052 ms
GPU D2H        : 0.0041 ms
GPU Pipeline   : 0.0303 ms
Compute Speedup: 0.65x (Với model nhỏ) -> Tăng vượt trội (>20x - 50x) với model lớn hàng triệu tam giác!
========================================
```

---

## 🎓 6. Bộ Câu Hỏi & Trả Lời Phản Bật Khi Báo Cáo

1. **Hỏi: Tại sao sử dụng bố cục dữ liệu Structure of Arrays (SoA) thay vì Array of Structures (AoS)?**
   * *Trả lời*: Trong SoA, các tọa độ $x_0$ của toàn bộ các tam giác nằm liên tiếp nhau trên RAM/VRAM. Khi 32 thread trong 1 Warp truy cập đồng thời, GPU thực hiện được **Coalesced Memory Access** (đọc 32 phần tử trong 1 transaction duy nhất), giúp tối ưu hóa tối đa băng thông bộ nhớ.
2. **Hỏi: Tại sao không sử dụng mảng mảng trung gian `d_areas[N]` trên Global Memory?**
   * *Trả lời*: Việc ghi kết quả tạm của từng tam giác ra Global Memory và đọc lại để Reduction gây lãng phí băng thông rất lớn. Chuỗi tối ưu giữ giá trị tổng tạm trong **Register** của từng thread, sau đó gom nhóm bằng **Warp Shuffle (`__shfl_down_sync`)** và Shared Memory, giúp loại bỏ hoàn toàn mảng `d_areas[N]`.
3. **Hỏi: Phân biệt Compute Speedup và Pipeline Speedup?**
   * *Trả lời*: 
     - $\text{Compute Speedup} = \frac{T_{\text{CPU}}}{T_{\text{Kernel}}}$: Thể hiện tốc độ tính toán thuần túy của GPU.
     - $\text{Pipeline Speedup} = \frac{T_{\text{CPU}}}{T_{\text{H2D}} + T_{\text{Kernel}} + T_{\text{D2H}}}$: Thể hiện hiệu năng thực tế toàn hệ thống (bao gồm cả chi phí truyền dữ liệu qua PCI-Express).
