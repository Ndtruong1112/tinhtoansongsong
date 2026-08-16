# Tính Toán Song Song - Tính Diện Tích Bề Mặt Mô Hình 3D STL (C / C++ / CUDA GPU)

Dự án tính toán diện tích bề mặt của các mô hình 3D định dạng **STL Binary** ứng dụng công nghệ **Tính Toán Song Song (Parallel Computing)** với NVIDIA CUDA trên GPU, kết hợp với các chương trình đối chứng tuần tự trên CPU bằng C và C++.

---

## 📌 Tính Năng Chính

1. **`test_c.c`**: Thuật toán đọc & tính diện tích bề mặt STL tuần tự trên CPU bằng ngôn ngữ **C** (Biên dịch bằng GCC).
2. **`test_cpp.cpp`**: Thuật toán đọc & tính diện tích bề mặt STL trên CPU sử dụng ngôn ngữ **C++** (Biên dịch bằng G++).
3. **`test_cuda.cu`**: Thuật toán song song hóa trên **NVIDIA GPU** bằng **CUDA Kernel `__global__`**, xử lý hàng triệu tam giác đồng thời trên hàng ngàn luồng (Thread/Block) song song.
4. **`sample_cube.stl`**: Tệp 3D nhị phân mẫu chuẩn (Hình lập phương $2 \times 2 \times 2$ có diện tích bề mặt lý thuyết là $24.0$) để kiểm thử tức thì.

---

## ⚙️ Môi Trường Cần Thiết

- **Trình biên dịch C/C++**: GCC / G++ (MinGW-w64 v16.1.0).
- **Trình biên dịch CUDA**: NVIDIA CUDA Toolkit v13.3 (`nvcc`).
- **Nền tảng biên dịch Host (Windows)**: Microsoft Visual C++ Build Tools (`cl.exe`).
- **Môi trường phát triển (IDE)**: Visual Studio Code với các Extension:
  - `C/C++` & `C/C++ Extension Pack` (`ms-vscode.cpptools`)
  - `NVIDIA Nsight Visual Studio Code Edition` (`NVIDIA.nsight-vscode-edition`)
  - `Code Runner` (`formulahendry.code-runner`)

---

## 🚀 Hướng Dẫn Biên Dịch & Chạy

### 1. Chạy trực tiếp từ Terminal / PowerShell

Mở PowerShell tại thư mục dự án và thực hiện các lệnh sau:

* **Biên dịch và chạy chương trình CUDA (GPU)**:
  ```powershell
  nvcc test_cuda.cu -o test_cuda.exe
  .\test_cuda.exe sample_cube.stl
  ```

* **Biên dịch và chạy chương trình C (CPU)**:
  ```powershell
  gcc test_c.c -o test_c.exe
  .\test_c.exe sample_cube.stl
  ```

* **Biên dịch và chạy chương trình C++ (CPU)**:
  ```powershell
  g++ test_cpp.cpp -o test_cpp.exe
  .\test_cpp.exe sample_cube.stl
  ```

> 💡 *Để tính toán cho file STL của riêng bạn, chỉ cần thay `sample_cube.stl` bằng đường dẫn tới file `.stl` của bạn.*

---

### 2. Chạy 1-Click trong Visual Studio Code

- **Dùng nút Play (Run)**: Mở bất kỳ file nào (`test_c.c`, `test_cpp.cpp`, `test_cuda.cu`) và bấm nút **Play (Run)** ở góc trên bên phải màn hình VS Code (hoặc nhấn `Ctrl` + `Alt` + `N`).
- **Dùng Debug (F5)**: Nhấn `Ctrl` + `Shift` + `D` $\rightarrow$ Chọn cấu hình Debug tương ứng $\rightarrow$ Bấm `F5`.

---

## 📂 Cấu Trúc Thư Mục

```text
├── .vscode/
│   ├── c_cpp_properties.json  # Cấu hình IntelliSense & đường dẫn Include
│   ├── settings.json          # Cấu hình phím tắt Code Runner & Terminal
│   ├── tasks.json             # Cấu hình Build Tasks cho GCC, G++, NVCC
│   └── launch.json            # Cấu hình Debug cho VS Code
├── test_c.c                   # Mã nguồn C (CPU)
├── test_cpp.cpp               # Mã nguồn C++ (CPU)
├── test_cuda.cu               # Mã nguồn CUDA GPU Parallel Kernel
├── sample_cube.stl            # Tệp 3D STL binary mẫu (Cube 2x2x2)
└── README.md                  # Tài liệu hướng dẫn sử dụng
```

---

## 📊 Kết Quả Mẫu (Output)

```text
========================================
[CUDA GPU] Dang tinh toan dien tich tren GPU NVIDIA...
File STL: sample_cube.stl
So luong tam giac: 12
Tong dien tich be mat = 24.000000
========================================
```
