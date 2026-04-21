import numpy as np
from io import StringIO

# Cấu hình hiển thị Numpy
np.set_printoptions(linewidth=np.inf, precision=4, suppress=True)

def lu_step_by_step_pro(A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    
    if B.ndim == 1:
        B = B.reshape(-1, 1)

    n_rows, n_cols = A.shape
    epsilon = 1e-12

    print("="*80)
    print(" PHẦN 1: KIỂM TRA ĐIỀU KIỆN & TIỀN XỬ LÝ MA TRẬN")
    print("="*80)

    # ---------------------------------------------------------
    # 1. KIỂM TRA MA TRẬN VUÔNG
    # ---------------------------------------------------------
    if n_rows != n_cols:
        print(f"[*] Ma trận A ({n_rows}x{n_cols}) KHÔNG VUÔNG.")
        print("[*] Đưa về hệ phương trình chuẩn: (A^T * A)X = A^T * B")
        M = A.T @ A
        D = A.T @ B
        print("\nMa trận M = A^T * A mới:")
        print(np.round(M, 4))
        print("\nMa trận tự do D = A^T * B mới:")
        print(np.round(D, 4))
    else:
        print("[*] Ma trận A VUÔNG. Giữ nguyên hệ phương trình M = A, D = B.")
        M = np.copy(A)
        D = np.copy(B)

    n = M.shape[0]

    # ---------------------------------------------------------
    # 2. KIỂM TRA CÁC ĐỊNH THỨC CON CHÍNH
    # ---------------------------------------------------------
    print("\n[*] Kiểm tra các định thức con chính (Leading Principal Minors) của M:")
    for i in range(1, n + 1):
        minor = np.linalg.det(M[:i, :i])
        print(f"  - Định thức con cấp {i} (\u0394_{i}): {minor:.4f}")
        if abs(minor) < epsilon:
            print(f"\n[!] CẢNH BÁO: Định thức con cấp {i} \u2248 0.")
            print("    Không thể áp dụng phân rã LU Doolittle cơ bản (Cần hoán vị dòng).")
            return None
    print("=> Tất cả \u0394_i \u2260 0. Đủ điều kiện phân rã LU.")

    # =========================================================
    # PHẦN 2: CHI TIẾT QUÁ TRÌNH PHÂN RÃ LU (M = L * U)
    # =========================================================
    print("\n" + "="*80)
    print(" PHẦN 2: PHÂN RÃ LU BẰNG PHƯƠNG PHÁP DOOLITTLE (l_ii = 1)")
    print("="*80)

    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        print(f"\n>>> BƯỚC {i+1}: Tính Dòng {i+1} của U và Cột {i+1} của L")
        
        # Tính các phần tử của U (Dòng i, duyệt cột j từ i đến n-1)
        for j in range(i, n):
            sum_lu = sum(L[i, k] * U[k, j] for k in range(i))
            U[i, j] = M[i, j] - sum_lu
            if i == 0:
                print(f"  + u_{i+1}{j+1} = m_{i+1}{j+1} = {U[i, j]:.4f}")
            else:
                print(f"  + u_{i+1}{j+1} = m_{i+1}{j+1} - \u2211(l_{i+1}k * u_k{j+1}) = {M[i, j]:.4f} - {sum_lu:.4f} = {U[i, j]:.4f}")
        
        # Tính các phần tử của L (Cột i, duyệt dòng j từ i+1 đến n-1)
        for j in range(i + 1, n):
            sum_lu = sum(L[j, k] * U[k, i] for k in range(i))
            L[j, i] = (M[j, i] - sum_lu) / U[i, i]
            
            print(f"  + l_{j+1}{i+1} = (m_{j+1}{i+1} - \u2211) / u_{i+1}{i+1} = ({M[j, i]:.4f} - {sum_lu:.4f}) / {U[i, i]:.4f} = {L[j, i]:.4f}")

    print("\n=> Ma trận L (Tam giác dưới):")
    print(np.round(L, 4))
    print("\n=> Ma trận U (Tam giác trên):")
    print(np.round(U, 4))

    # =========================================================
    # PHẦN 3: GIẢI HỆ PHƯƠNG TRÌNH
    # =========================================================
    print("\n" + "="*80)
    print(" PHẦN 3: THẾ TIẾN VÀ THẾ LÙI")
    print("="*80)

    m_b = D.shape[1]
    Y = np.zeros((n, m_b))
    
    print("\n[A] Giải hệ thuận L * Y = D (Từ trên xuống):")
    for c in range(m_b):
        print(f"  --- Với cột tự do thứ {c+1} ---")
        for i in range(n):
            sum_ly = sum(L[i, j] * Y[j, c] for j in range(i))
            Y[i, c] = D[i, c] - sum_ly
            print(f"  y_{i+1} = {D[i, c]:.4f} - ({sum_ly:.4f}) = {Y[i, c]:.4f}")

    print("\n=> Ma trận trung gian Y:")
    print(np.round(Y, 4))

    X = np.zeros((n, m_b))
    print("\n[B] Giải hệ nghịch U * X = Y (Từ dưới lên):")
    for c in range(m_b):
        print(f"  --- Giải tìm cột nghiệm thứ {c+1} ---")
        for i in range(n-1, -1, -1):
            sum_ux = sum(U[i, j] * X[j, c] for j in range(i+1, n))
            X[i, c] = (Y[i, c] - sum_ux) / U[i, i]
            print(f"  x_{i+1} = ({Y[i, c]:.4f} - {sum_ux:.4f}) / {U[i, i]:.4f} = {X[i, c]:.4f}")

    print("\n" + "="*80)
    print(" KẾT QUẢ CUỐI CÙNG: NGHIỆM X")
    print("="*80)
    print(np.round(X, 6))
    return X

# ---------------------------------------------------------
# KHU VỰC CHẠY THỬ (TEST CASE)
# ---------------------------------------------------------
# Thử với hệ phương trình không vuông (3 phương trình, 2 ẩn)
input_A = """
 2  -1
 1   2
 1   1
"""

input_B = """
 2
 1
 4
"""

try:
    A_matrix = np.loadtxt(StringIO(input_A.strip()))
    B_matrix = np.loadtxt(StringIO(input_B.strip()))
    
    nghiem = lu_step_by_step_pro(A_matrix, B_matrix)
    
except Exception as e:
    print(f"Lỗi: {e}")