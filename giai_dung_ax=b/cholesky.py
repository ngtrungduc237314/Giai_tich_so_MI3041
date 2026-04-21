import numpy as np
import math
from io import StringIO

# Cấu hình hiển thị Numpy
np.set_printoptions(precision=4, suppress=True, linewidth=100)

def cholesky_ultimate_step_by_step(A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    
    if B.ndim == 1:
        B = B.reshape(-1, 1)
        
    n = A.shape[0]
    m = B.shape[1]

    print("="*70)
    print(" BƯỚC 1: KIỂM TRA & BIẾN ĐỔI HỆ PHƯƠNG TRÌNH")
    print("="*70)
    
    if np.allclose(A, A.T):
        print("=> Ma trận A ĐÃ ĐỐI XỨNG. Giữ nguyên hệ: AX = B")
        M = A
        D = B
    else:
        print("=> Ma trận A KHÔNG ĐỐI XỨNG.")
        print("=> Biến đổi thành hệ phương trình chuẩn: (A^T * A)X = A^T * B")
        M = A.T @ A
        D = A.T @ B
        
        print("\nMa trận hệ số mới M = A^T * A:")
        print(np.round(M, 4))
        print("\nMa trận tự do mới D = A^T * B:")
        print(np.round(D, 4))

    print("\n" + "="*70)
    print(" BƯỚC 2: PHÂN RÃ CHOLESKY (M = L * L^T)")
    print("="*70)
    
    L = np.zeros((n, n))
    for j in range(n):
        print(f"\n>>> Tính các phần tử CỘT {j + 1} của ma trận L:")
        for i in range(j, n):
            # Tính tổng phần bình phương/tích của các phần tử l đã biết
            sum_k = sum(L[i, k] * L[j, k] for k in range(j))
            
            if i == j: # Phần tử trên đường chéo chính (Tính căn)
                val = M[j, j] - sum_k
                if val <= 0:
                    print(f"\n[!] Lỗi: Ma trận không xác định dương tại phần tử [{j+1},{j+1}].")
                    return None
                L[j, j] = math.sqrt(val)
                print(f"  + l_{j+1}{j+1} = sqrt({M[j, j]:.4f} - {sum_k:.4f}) = {L[j, j]:.4f}")
            else: # Phần tử dưới đường chéo chính
                L[i, j] = (M[i, j] - sum_k) / L[j, j]
                print(f"  + l_{i+1}{j+1} = ({M[i, j]:.4f} - {sum_k:.4f}) / {L[j, j]:.4f} = {L[i, j]:.4f}")

    print("\n=> Ma trận tam giác dưới L thu được:")
    print(np.round(L, 4))

    print("\n" + "="*70)
    print(" BƯỚC 3: GIẢI HỆ THUẬN (L * Y = D)")
    print("="*70)
    
    Y = np.zeros((n, m))
    for c in range(m):
        print(f"\n>>> Giải tìm cột Y thứ {c + 1} (Từ trên xuống):")
        for i in range(n):
            sum_y = sum(L[i, k] * Y[k, c] for k in range(i))
            Y[i, c] = (D[i, c] - sum_y) / L[i, i]
            print(f"  + y_{i+1} = ({D[i, c]:.4f} - {sum_y:.4f}) / {L[i, i]:.4f} = {Y[i, c]:.4f}")

    print("\n=> Ma trận trung gian Y thu được:")
    print(np.round(Y, 4))

    print("\n" + "="*70)
    print(" BƯỚC 4: GIẢI HỆ NGHỊCH (L^T * X = Y)")
    print("="*70)
    
    U = L.T
    X = np.zeros((n, m))
    for c in range(m):
        print(f"\n>>> Giải tìm cột X thứ {c + 1} (Từ dưới lên):")
        for i in range(n - 1, -1, -1):
            sum_x = sum(U[i, k] * X[k, c] for k in range(i + 1, n))
            X[i, c] = (Y[i, c] - sum_x) / U[i, i]
            print(f"  + x_{i+1} = ({Y[i, c]:.4f} - {sum_x:.4f}) / {U[i, i]:.4f} = {X[i, c]:.4f}")

    print("\n" + "="*70)
    print(" KẾT QUẢ CUỐI CÙNG: NGHIỆM X")
    print("="*70)
    print(np.round(X, 4))
    return X

# ==========================================
# KHU VỰC NHẬP LIỆU
# ==========================================
input_A = """
 1   2   3
 2   5   1
 1   1   4
"""

input_B = """
 14
 17
 17
"""

try:
    mat_A = np.loadtxt(StringIO(input_A.strip()), dtype=float)
    mat_B = np.loadtxt(StringIO(input_B.strip()), dtype=float)
    
    nghiem_X = cholesky_ultimate_step_by_step(mat_A, mat_B)

except Exception as e:
    print(f"Lỗi: {e}")