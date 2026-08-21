#!/usr/bin/env python3
"""
3D Mesh Generator for Parallel Computing Benchmarks
Generates high-density Binary STL meshes (3D Wave Surface, UV Sphere, Flat Plane)
with distinct, non-overlapping triangles for CPU and CUDA GPU benchmarking.
"""

import struct
import math
import sys
import os

def generate_wave_mesh(filename: str, target_triangles: int, amplitude: float = 0.3):
    """
    Generates a 3D Sine-Cosine wave surface:
    z = amplitude * sin(pi * x) * cos(pi * y) over x in [-1, 1], y in [-1, 1].
    """
    M = int(math.isqrt(target_triangles // 2))
    if M < 1:
        M = 1
    actual_triangles = M * M * 2
    header = f"3D Wave Surface ({actual_triangles} triangles, A={amplitude})".encode("utf-8").ljust(80, b"\x00")
    
    scale = 2.0
    dx = scale / M
    dy = scale / M

    print(f"[*] Generating '{filename}' with {M}x{M} grid ({actual_triangles:,} triangles)...")
    
    with open(filename, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", actual_triangles))
        
        chunk_size = 100000
        buf = bytearray()
        
        for iy in range(M):
            y0 = -1.0 + iy * dy
            y1 = y0 + dy
            for ix in range(M):
                x0 = -1.0 + ix * dx
                x1 = x0 + dx
                
                z00 = amplitude * math.sin(math.pi * x0) * math.cos(math.pi * y0)
                z10 = amplitude * math.sin(math.pi * x1) * math.cos(math.pi * y0)
                z01 = amplitude * math.sin(math.pi * x0) * math.cos(math.pi * y1)
                z11 = amplitude * math.sin(math.pi * x1) * math.cos(math.pi * y1)
                
                v00 = (x0, y0, z00)
                v10 = (x1, y0, z10)
                v01 = (x0, y1, z01)
                v11 = (x1, y1, z11)
                
                # Triangle 1 (v00, v10, v11)
                buf += struct.pack("<3f3f3f3fH", 0.0, 0.0, 0.0, *v00, *v10, *v11, 0)
                # Triangle 2 (v00, v11, v01)
                buf += struct.pack("<3f3f3f3fH", 0.0, 0.0, 0.0, *v00, *v11, *v01, 0)
                
                if len(buf) >= chunk_size * 50:
                    f.write(buf)
                    buf = bytearray()
        if buf:
            f.write(buf)
            
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"    [+] Finished '{filename}': {actual_triangles:,} triangles, {size_mb:.2f} MB")

def main():
    presets = [
        ("mesh_1k.stl", 1000),
        ("mesh_10k.stl", 10000),
        ("mesh_100k.stl", 100000),
        ("mesh_1m.stl", 1000000),
        ("mesh_5m.stl", 5000000),
        ("mesh_10m.stl", 10000000),
    ]
    
    print("==================================================")
    print("  3D Parametric Mesh Generator (Binary STL Format) ")
    print("==================================================")
    for filename, count in presets:
        generate_wave_mesh(filename, count)
    print("==================================================")
    print("All STL benchmark files generated successfully!")

if __name__ == "__main__":
    main()
