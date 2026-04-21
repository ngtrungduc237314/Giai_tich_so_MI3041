import numpy as np
import sympy as sym

# Đặt giới hạn hiển thị dòng không bị cắt
np.set_printoptions(linewidth=np.inf, precision=6, suppress=True)

def gauss_inverse_process_pro(augmented_matrix, b_cols):
    m, total_cols = augmented_matrix.shape
    n = total_cols - b_cols  # Số cột của A
    
    # Làm tròn để khử sai số dấu phẩy động thừa trước khi đưa vào SymPy
    A = np.round(augmented_matrix[:, :n], 10)
    B = np.round(augmented_matrix[:, n:], 10)
    epsilon = 1e-10
    
    print("="*70)
    print(" BẮT ĐẦU QUÁ TRÌNH NGHỊCH (BACK SUBSTITUTION)")
    print("="*70)
    
    # 1. Tìm chỉ số cột đầu tiên khác 0 (Pivot) với Epsilon
    id = np.zeros(m, dtype=int)
    for i in range(m):
        for j in range(n):
            if abs(A[i, j]) > epsilon:
                id[i] = j + 1  
                break
    
    # 2. Phân loại biến
    all_vars = set(range(1, n + 1))
    pivot_vars = set(id[id > 0])
    free_vars = sorted(all_vars - pivot_vars)
    
    print(f"[*] Các biến cơ sở (Pivot): {[f'x{i}' for i in pivot_vars]}")
    print(f"[*] Các biến tự do: {[f'x{i}' for i in free_vars]}")
    
    # Tạo biến tượng trưng
    t = sym.symbols([f't{i}' for i in free_vars])
    
    solutions = []
    
    # 3. Giải cho từng cột của B
    for col in range(b_cols):
        print(f"\n{'-'*25} GIẢI CỘT B THỨ {col+1} {'-'*25}")
        b = B[:, col]
        
        # Kiểm tra vô nghiệm
        is_no_solution = False
        for i in range(m):
            if id[i] == 0 and abs(b[i]) > epsilon:
                print(f" [!] Hàng {i+1} vô lý (0 = {b[i]:.4f}). HỆ VÔ NGHIỆM!")
                is_no_solution = True
                break
                
        if is_no_solution:
            solutions.append(None)
            continue
            
        x = sym.zeros(n, 1)
        
        # Gán biến tự do
        for idx, var in zip(free_vars, t):
            x[idx - 1] = var
            print(f" -> Đặt biến tự do: x{idx} = {var}")
            
        # Thế ngược tìm biến chính
        for j in sorted(pivot_vars, reverse=True):
            i = np.where(id == j)[0][0] # Tìm hàng chứa pivot
            
            # Tính tổng phần tử đã biết, lọc bỏ hệ số 0 để in ra cho đẹp
            sum_term = 0
            sum_str_parts = []
            for k in range(j, n):
                if abs(A[i, k]) > epsilon:
                    sum_term += A[i, k] * x[k]
                    sum_str_parts.append(f"({A[i, k]:.4f} * x{k+1})")
                    
            sum_str = " + ".join(sum_str_parts) if sum_str_parts else "0"
            
            print(f"\n -> Từ Hàng {i+1}, tính x{j}:")
            print(f"    Phương trình: {A[i, j-1]:.4f} * x{j} + {sum_str} = {b[i]:.4f}")
            
            # Tính toán bằng SymPy và rút gọn
            x[j - 1] = sym.simplify((b[i] - sum_term) / A[i, j - 1])
            
            # Format chuỗi để in kết quả biểu thức (xóa bớt dấu *)
            expr_str = str(x[j-1]).replace("**", "^").replace("*", "")
            print(f"    => x{j} = {expr_str}")
            
        solutions.append(x)
        
    return solutions, free_vars, t

def display_solutions_pro(solutions, free_vars, t, b_cols, n):
    print("\n" + "="*70)
    print(" KẾT LUẬN NGHIỆM TỔNG QUÁT")
    print("="*70)
    
    if all(sol is not None for sol in solutions):
        first_sol = solutions[0]
        for i in range(n):
            coeffs = {t_k: first_sol[i].coeff(t_k) for t_k in t}
            constants = [sol[i].subs({t_k: 0 for t_k in t}) for sol in solutions]
            
            coeffs_float = {t_k: float(coeff) for t_k, coeff in coeffs.items()}
            constants_float = [float(c) for c in constants]
            
            terms = []
            for t_k in t:
                coeff = coeffs_float[t_k]
                if abs(coeff) > 1e-10: # Chỉ in nếu hệ số khác 0
                    coeff_vec = [coeff] * b_cols
                    coeff_str = '[' + ', '.join([f'{c:.4f}' for c in coeff_vec]) + ']'
                    terms.append(f"{coeff_str} * {t_k}")
                    
            const_vec = '[' + ', '.join([f'{c:.4f}' for c in constants_float]) + ']'
            terms.append(const_vec)
            
            display_str = " + ".join(terms)
            print(f" x{i+1} = {display_str}")
    else:
        print(" [!] Tồn tại hệ vô nghiệm, không thể hiển thị vector tổng quát.")

# ==========================================
# KHU VỰC THỬ NGHIỆM
# ==========================================
augmented_matrix = np.array([
[1,  2,  -1,  3,  4],
[2,  4,  -2,  6,  8],
[1,  3,   2,  2,  5]
])

b_cols = 2
n = augmented_matrix.shape[1] - b_cols

solutions, free_vars, t = gauss_inverse_process_pro(augmented_matrix, b_cols)
display_solutions_pro(solutions, free_vars, t, b_cols, n)