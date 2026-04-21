import numpy as np
from io import StringIO

np.set_printoptions(linewidth=np.inf, precision=4, suppress=True)

def print_matrix(matrix):
    print(np.round(matrix, 4))
    print("")

def clean_num(val):
    if abs(val) < 1e-10: return "0"
    s = f"{val:.4f}"
    return s.rstrip('0').rstrip('.') if '.' in s else s

def vec_to_str(v):
    return "[" + ", ".join([clean_num(x) for x in v]) + "]^T"

def gauss_jordan_image_algorithm(A, B):
    # ==========================================
    # KHỞI TẠO VÀ CHUẨN BỊ
    # ==========================================
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    if B.ndim == 1: B = B.reshape(-1, 1) # Đảm bảo B là ma trận 2 chiều
    
    rows, num_vars = A.shape
    num_systems = B.shape[1]
    
    # 1. Xét ma trận mở rộng A|B
    aug = np.hstack((A, B))
    epsilon = 1e-10
    
    print("="*70)
    print(" MA TRẬN MỞ RỘNG A|B BAN ĐẦU:")
    print_matrix(aug)
    print("="*70)

    # THUẬT TOÁN BƯỚC 1: Tạo 2 danh sách rỗng
    rowuse = []
    coluse = []
    
    step = 1
    # Vòng lặp khử
    while True:
        # THUẬT TOÁN BƯỚC 4 (Kiểm tra điều kiện dừng)
        if len(rowuse) == rows or len(coluse) == num_vars:
            print("\n>>> Đã duyệt hết hàng hoặc cột. Chuyển sang Bước 5.")
            break
            
        print(f"\n{'-'*20} LẦN LẶP THỨ {step} {'-'*20}")
        
        valid_rows = [i for i in range(rows) if i not in rowuse]
        valid_cols = [j for j in range(num_vars) if j not in coluse]
        
        # THUẬT TOÁN BƯỚC 2: Tìm kiếm phần tử khử a theo ưu tiên
        best_p, best_q = -1, -1
        found_1 = False
        
        # Ưu tiên 1: |a| = 1
        for r in valid_rows:
            for c in valid_cols:
                if np.isclose(abs(aug[r, c]), 1.0):
                    best_p, best_q = r, c
                    found_1 = True
                    break
            if found_1: break
            
        # Ưu tiên 2: |a| = max{|a_ij|}
        if not found_1:
            max_val = -1
            for r in valid_rows:
                for c in valid_cols:
                    if abs(aug[r, c]) > max_val:
                        max_val = abs(aug[r, c])
                        best_p, best_q = r, c
                        
        pivot_val = aug[best_p, best_q]
        
        # Bước 4 (trường hợp có hàng bằng 0 / pivot = 0)
        if abs(pivot_val) < epsilon:
            print(">>> Các phần tử còn lại đều bằng 0 (Pivot = 0). Chuyển sang Bước 5.")
            break
            
        print(f"[Bước 2] Chọn phần tử khử a_{best_p+1},{best_q+1} = {clean_num(pivot_val)}")
        print(f"         (Hàng p={best_p+1}, Cột q={best_q+1} chưa được sử dụng)")
        
        # THUẬT TOÁN BƯỚC 3: Lưu vị trí vào rowuse, coluse
        rowuse.append(best_p)
        coluse.append(best_q)
        print(f"         => rowuse = {[r+1 for r in rowuse]}, coluse = {[c+1 for c in coluse]}")
        
        # KHỬ MA TRẬN theo công thức (*): L_t = L_t - (a_tq / a_pq) * L_p
        print(f"[Bước 3] Khử ma trận (trừ hàng {best_p+1}):")
        has_elim = False
        for t in range(rows):
            if t != best_p: # Cho t chạy từ 1 đến m, t khác p
                factor = aug[t, best_q] / pivot_val
                if abs(factor) > epsilon:
                    aug[t] -= factor * aug[best_p]
                    print(f"  + L_{t+1} = L_{t+1} - ({clean_num(factor)}) * L_{best_p+1}")
                    has_elim = True
                    
        if not has_elim:
            print("  + Không có phần tử nào cần khử ở cột này.")
            
        print("Ma trận sau khi khử ở lần lặp này:")
        print_matrix(aug)
        step += 1

    # THUẬT TOÁN BƯỚC 5: Chuẩn hóa và sắp xếp về dạng bậc thang
    print("="*70)
    print(" BƯỚC 5: CHUẨN HÓA VÀ SẮP XẾP VỀ DẠNG BẬC THANG")
    print("="*70)
    
    # a. Chia từng hàng cho pivot
    for p, q in zip(rowuse, coluse):
        pivot = aug[p, q]
        if not np.isclose(pivot, 1.0):
            aug[p] /= pivot
            print(f" + Chia hàng {p+1} cho {clean_num(pivot)}")
            
    # b. Sắp xếp về dạng bậc thang
    # Cách sắp xếp: Các hàng được sử dụng sẽ được xếp theo thứ tự cột khử (coluse) tăng dần
    # Các hàng không được sử dụng sẽ bị đẩy xuống cuối
    sorted_pairs = sorted(zip(rowuse, coluse), key=lambda x: x[1])
    row_order = [pair[0] for pair in sorted_pairs]
    unused_rows = [i for i in range(rows) if i not in rowuse]
    final_row_order = row_order + unused_rows
    
    aug = aug[final_row_order]
    
    # Sửa lại số 0 tuyệt đối để tránh -0.0000
    aug[np.abs(aug) < epsilon] = 0
    
    print("\nMa trận A|B đã chuẩn hóa và sắp xếp (Dạng RREF):")
    print_matrix(aug)
    
    # THUẬT TOÁN BƯỚC 6: Trả về kết quả
    print("="*70)
    print(" BƯỚC 6: TRẢ VỀ KẾT QUẢ NGHIỆM")
    print("="*70)
    
    rref_A = aug[:, :num_vars]
    rref_B = aug[:, num_vars:]
    rank_A = len(rowuse)
    
    # Xác định biến cơ sở và biến tự do dựa vào coluse
    free_cols = [c for c in range(num_vars) if c not in coluse]
    pivot_cols = sorted(coluse)

    for j in range(num_systems):
        print(f"\n>>> [HỆ PHƯƠNG TRÌNH {j+1} - Tương ứng cột B{j+1}]")
        b = rref_B[:, j]
        
        # Kiểm tra vô nghiệm
        inconsistent = False
        for i in range(rank_A, rows):
            if abs(b[i]) > epsilon:
                inconsistent = True
                break
                
        if inconsistent:
            print(" => KẾT LUẬN: HỆ VÔ NGHIỆM.")
            continue
            
        if rank_A == num_vars:
            X = np.zeros(num_vars)
            for r, c in enumerate(pivot_cols):
                X[c] = b[r]
            print(" => KẾT LUẬN: HỆ CÓ NGHIỆM DUY NHẤT.")
            print(f"    X = {vec_to_str(X)}")
            
        else:
            print(f" => KẾT LUẬN: HỆ CÓ VÔ SỐ NGHIỆM.")
            print("    (Chuyển các cột không có phần tử khử của A ra sau B)")
            
            X_part = np.zeros(num_vars)
            for r, c in enumerate(pivot_cols):
                X_part[c] = b[r]
                
            ans_str = f"    X = {vec_to_str(X_part)}  (Nghiệm riêng X_0)"
            
            for free_c in free_cols:
                V = np.zeros(num_vars)
                V[free_c] = 1.0
                for r, c in enumerate(pivot_cols):
                    V[c] = -rref_A[r, free_c] 
                
                ans_str += f"\n      + t_{free_c+1} * {vec_to_str(V)}"
                
            print(ans_str)

# ==========================================
# KHU VỰC NHẬP MA TRẬN A VÀ B
# ==========================================

# Bạn có thể copy trực tiếp các số cách nhau bởi dấu cách vào đây
input_A = """
3  -2  5  -7  4 
2   9 14 -30  0 
5  -4 18 -26 14 
-4  2  3  -5  2 
1   3  2  -6 -2 
"""

input_B = """
3   5
-5  10
7   11
-2  -4
-2   3
"""

try:
    # Đọc dữ liệu tự động bỏ qua khoảng trắng thừa
    mat_A = np.loadtxt(StringIO(input_A.strip()))
    mat_B = np.loadtxt(StringIO(input_B.strip()))
    
    # Gọi hàm thực thi
    gauss_jordan_image_algorithm(mat_A, mat_B)
except Exception as e:
    print(f"Lỗi nhập liệu: {e}")