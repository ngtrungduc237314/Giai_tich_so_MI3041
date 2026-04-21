"""
=============================================================
  PHƯƠNG PHÁP NEWTON & NEWTON MODIFIED CHO HỆ PT PHI TUYẾN
  F(X) = 0
  Lý thuyết: Hà Thị Ngọc Yến, Hà Nội 2024
=============================================================
"""

import sympy as sp
from sympy import (symbols, diff, simplify, sqrt, sympify,
                   Rational, pretty, pi, E, sin, cos, tan,
                   exp, log, Matrix)
import warnings
warnings.filterwarnings("ignore")

# ── Màu ──────────────────────────────────────────────────────
class C:
    BOLD    = '\033[1m';  CYAN    = '\033[96m';  GREEN   = '\033[92m'
    YELLOW  = '\033[93m'; RED     = '\033[91m';  MAGENTA = '\033[95m'
    BLUE    = '\033[94m'; RESET   = '\033[0m'

def box(title, color=C.CYAN):
    w = 72
    print(f"\n{color}{C.BOLD}{'═'*w}\n  {title}\n{'═'*w}{C.RESET}")

def sep(char='─'):
    print(f"{C.BLUE}{char*72}{C.RESET}")

def banner():
    print(f"""{C.CYAN}{C.BOLD}
  ┌──────────────────────────────────────────────────────────────┐
  │   PHƯƠNG PHÁP NEWTON & NEWTON MODIFIED                       │
  │   Giải hệ F(X) = 0  —  Hà Thị Ngọc Yến, Hà Nội 2024        │
  └──────────────────────────────────────────────────────────────┘
{C.RESET}""")


# ══════════════════════════════════════════════════════════════
#  LÝ THUYẾT
# ══════════════════════════════════════════════════════════════

def in_ly_thuyet(n):
    box("CƠ SỞ LÝ THUYẾT", C.MAGENTA)
    print(f"""
{C.YELLOW}Bài toán:{C.RESET}
  Giải hệ F(X) = 0, X ∈ ℝ^{n}

{C.YELLOW}Ý tưởng Newton:{C.RESET}
  Tuyến tính hóa F quanh điểm hiện tại X_k:
    F(X) ≈ F(X_k) + J(X_k) · (X - X_k) = 0
  Giải ra:
    X_(k+1) = X_k - J(X_k)⁻¹ · F(X_k)

  Đây là lặp đơn với:
    Φ(X) = X - J(X)⁻¹ · F(X)

{C.YELLOW}NEWTON (tiêu chuẩn):{C.RESET}
  Mỗi bước lặp dùng Jacobi tại điểm HIỆN TẠI X_k:
    X_(k+1) = X_k - J(X_k)⁻¹ · F(X_k)
  → Hội tụ BẬC 2 (rất nhanh)
  → Mỗi bước phải tính lại J(X_k) và giải hệ tuyến tính

{C.YELLOW}NEWTON MODIFIED:{C.RESET}
  Chỉ tính Jacobi MỘT LẦN tại X_0, dùng cho tất cả các bước:
    X_(k+1) = X_k - J(X_0)⁻¹ · F(X_k)
  → Hội tụ BẬC 1 (chậm hơn Newton tiêu chuẩn)
  → Tiết kiệm tính toán (không phải tính lại J mỗi bước)

{C.YELLOW}Điều kiện hội tụ (cục bộ):{C.RESET}
  Tồn tại lân cận U(X*) sao cho với X_0 ∈ U(X*):
    ‖J(X*)⁻¹ · F(X)‖ đủ nhỏ  →  dãy lặp hội tụ
  Thực tế: chọn X_0 đủ gần nghiệm.

{C.YELLOW}Đánh giá sai số (hậu nghiệm):{C.RESET}
  ‖X_k - X*‖ ≤ K/(1-K) · ‖X_k - X_(k-1)‖
  với K = ‖J(X_0)⁻¹‖ · ‖J(X_k) - J(X_0)‖  (ước lượng)

{C.YELLOW}Tiêu chuẩn dừng thực tế:{C.RESET}
  ‖F(X_k)‖_∞ < ε   (residual)        — kiểm tra F gần 0
  ‖X_k - X_(k-1)‖_∞ < ε              — bước lặp đủ nhỏ
  (chương trình dùng CẢ HAI)
""")


# ══════════════════════════════════════════════════════════════
#  TÍNH JACOBI TẠI ĐIỂM (numeric)
# ══════════════════════════════════════════════════════════════

def tinh_jacobi_tai_diem(J_sym, all_vars, n, X_val):
    subs = {all_vars[i]: X_val[i] for i in range(n)}
    J_num = []
    for i in range(n):
        row = []
        for j in range(n):
            try:
                v = float(J_sym[i][j].subs(subs).evalf())
            except Exception:
                v = 0.0
            row.append(v)
        J_num.append(row)
    return J_num

def tinh_F_tai_diem(F_list, all_vars, n, X_val):
    subs = {all_vars[i]: X_val[i] for i in range(n)}
    F_val = []
    for fi in F_list:
        try:
            v = float(fi.subs(subs).evalf())
        except Exception:
            v = float('nan')
        F_val.append(v)
    return F_val


# ══════════════════════════════════════════════════════════════
#  GIẢI HỆ TUYẾN TÍNH  J · delta = -F  (Gauss không pivot)
# ══════════════════════════════════════════════════════════════

def giai_he_tuyen_tinh(J_num, F_val, n):
    """
    Giải J · Δ = -F bằng numpy (nhanh, ổn định).
    Trả về Δ = -J⁻¹·F
    """
    import numpy as np
    J_np = np.array(J_num, dtype=float)
    F_np = np.array(F_val, dtype=float)
    try:
        delta = np.linalg.solve(J_np, -F_np)
        return delta.tolist()
    except np.linalg.LinAlgError:
        return None


# ══════════════════════════════════════════════════════════════
#  IN MA TRẬN (đẹp)
# ══════════════════════════════════════════════════════════════

def in_matrix(M, label, n, prec=8):
    fmt = f"{{:>{prec+6}.{prec}f}}"
    print(f"\n  {C.YELLOW}{label}:{C.RESET}")
    for i in range(n):
        row_s = "  [ " + "  ".join(fmt.format(M[i][j]) for j in range(n)) + "  ]"
        print(row_s)

def in_vector(v, label, n, prec=8):
    fmt = f"{{:>{prec+6}.{prec}f}}"
    print(f"\n  {C.YELLOW}{label}:{C.RESET}")
    vals = "  [ " + "  ".join(fmt.format(v[i]) for i in range(n)) + "  ]"
    print(vals)


# ══════════════════════════════════════════════════════════════
#  NEWTON TIÊU CHUẨN
# ══════════════════════════════════════════════════════════════

def newton_chuan(F_list, J_sym, all_vars, n, X0, eps, max_iter=50):
    box("PHƯƠNG PHÁP NEWTON TIÊU CHUẨN", C.CYAN)
    prec = max(8, int(-sp.log(float(eps), 10)) + 3)
    fmt  = f"{{:.{prec}f}}"

    print(f"\n{C.YELLOW}Công thức lặp:{C.RESET}")
    print(f"  X_(k+1) = X_k - J(X_k)⁻¹ · F(X_k)")
    print(f"  (Jacobi tính lại tại MỖI bước k)")
    print(f"\n{C.YELLOW}Điểm xuất phát X⁰:{C.RESET}")
    print("  " + ",  ".join(f"x_{i+1}⁰ = {X0[i]}" for i in range(n)))
    print(f"\n{C.YELLOW}Sai số yêu cầu: ε = {eps}{C.RESET}")
    sep('═')

    # In Jacobi symbolic
    print(f"\n{C.CYAN}Ma trận Jacobi J(X) — symbolic:{C.RESET}")
    for i in range(n):
        parts = ",   ".join(f"∂f_{i+1}/∂x_{j+1} = {J_sym[i][j]}" for j in range(n))
        print(f"  Hàng {i+1}: {parts}")

    X_cur  = [float(v) for v in X0]
    history = [X_cur[:]]
    F_history = []
    converged = False

    for k in range(1, max_iter + 1):
        print(f"\n{C.BOLD}{C.MAGENTA}  {'─'*66}{C.RESET}")
        print(f"{C.BOLD}{C.MAGENTA}  Bước k = {k}:{C.RESET}")

        # Tính F(X_k)
        F_val = tinh_F_tai_diem(F_list, all_vars, n, X_cur)
        norm_F = max(abs(v) for v in F_val)
        F_history.append(F_val[:])

        print(f"\n  {C.CYAN}F(X^({k-1})):{C.RESET}")
        for i in range(n):
            print(f"    f_{i+1}(X^({k-1})) = {fmt.format(F_val[i])}")
        print(f"    ‖F(X^({k-1}))‖_∞ = {fmt.format(norm_F)}")

        # Tính J(X_k)
        J_num = tinh_jacobi_tai_diem(J_sym, all_vars, n, X_cur)
        print(f"\n  {C.CYAN}J(X^({k-1})) (Jacobi tại bước này):{C.RESET}")
        in_matrix(J_num, f"J(X^({k-1}))", n, prec)

        # Giải J·Δ = -F
        delta = giai_he_tuyen_tinh(J_num, F_val, n)
        if delta is None:
            print(f"  {C.RED}✗ Jacobi suy biến (det=0) tại bước k={k}. Dừng.{C.RESET}")
            break
        in_vector(delta, f"Δ^({k-1}) = J(X^({k-1}))⁻¹·(-F(X^({k-1})))", n, prec)

        # Cập nhật
        X_new = [X_cur[i] + delta[i] for i in range(n)]
        norm_delta = max(abs(delta[i]) for i in range(n))

        print(f"\n  {C.CYAN}X^({k}) = X^({k-1}) + Δ^({k-1}):{C.RESET}")
        for i in range(n):
            print(f"    x_{i+1}^({k}) = {fmt.format(X_cur[i])} + ({fmt.format(delta[i])}) = {fmt.format(X_new[i])}")

        # Sai số
        print(f"\n  {C.YELLOW}Kiểm tra dừng:{C.RESET}")
        print(f"    ‖Δ^({k-1})‖_∞ = ‖X^({k}) - X^({k-1})‖_∞ = {fmt.format(norm_delta)}")

        # Tính F mới để kiểm tra residual
        F_new = tinh_F_tai_diem(F_list, all_vars, n, X_new)
        norm_F_new = max(abs(v) for v in F_new)
        print(f"    ‖F(X^({k}))‖_∞ = {fmt.format(norm_F_new)}")

        history.append(X_new[:])
        X_cur = X_new[:]

        if norm_delta < eps and norm_F_new < eps:
            print(f"\n  {C.GREEN}✓ ‖ΔX‖ = {fmt.format(norm_delta)} < ε  VÀ  ‖F‖ = {fmt.format(norm_F_new)} < ε  → DỪNG{C.RESET}")
            converged = True
            break
        elif norm_delta < eps:
            print(f"\n  {C.GREEN}✓ ‖ΔX‖ = {fmt.format(norm_delta)} < ε  → DỪNG (bước đủ nhỏ){C.RESET}")
            converged = True
            break

    return X_cur, k, converged, history


# ══════════════════════════════════════════════════════════════
#  NEWTON MODIFIED
# ══════════════════════════════════════════════════════════════

def newton_modified(F_list, J_sym, all_vars, n, X0, eps, max_iter=100):
    box("PHƯƠNG PHÁP NEWTON MODIFIED", C.CYAN)
    prec = max(8, int(-sp.log(float(eps), 10)) + 3)
    fmt  = f"{{:.{prec}f}}"

    print(f"\n{C.YELLOW}Công thức lặp:{C.RESET}")
    print(f"  X_(k+1) = X_k - J(X_0)⁻¹ · F(X_k)")
    print(f"  (Jacobi chỉ tính MỘT LẦN tại X_0, dùng cho tất cả bước)")
    print(f"\n{C.YELLOW}Điểm xuất phát X⁰:{C.RESET}")
    print("  " + ",  ".join(f"x_{i+1}⁰ = {X0[i]}" for i in range(n)))
    print(f"\n{C.YELLOW}Sai số yêu cầu: ε = {eps}{C.RESET}")
    sep('═')

    # Tính J(X_0) một lần duy nhất
    J0_num = tinh_jacobi_tai_diem(J_sym, all_vars, n, X0)
    print(f"\n{C.CYAN}Tính J(X⁰) — CHỈ MỘT LẦN, dùng cho tất cả bước:{C.RESET}")
    in_matrix(J0_num, "J(X⁰)", n, prec)

    # Kiểm tra J(X0) có khả nghịch không
    import numpy as np
    J0_np = np.array(J0_num, dtype=float)
    det_J0 = np.linalg.det(J0_np)
    print(f"\n  det(J(X⁰)) = {det_J0:.8f}", end="")
    if abs(det_J0) < 1e-12:
        print(f"  {C.RED}→ J(X⁰) suy biến! Không thể dùng Newton Modified với X⁰ này.{C.RESET}")
        return X0, 0, False, [X0[:]]
    else:
        print(f"  {C.GREEN}→ J(X⁰) khả nghịch ✓{C.RESET}")

    # In J(X0)^{-1} để tham khảo
    J0_inv = np.linalg.inv(J0_np)
    print(f"\n{C.CYAN}J(X⁰)⁻¹ (tính một lần, dùng lại mỗi bước):{C.RESET}")
    in_matrix(J0_inv.tolist(), "J(X⁰)⁻¹", n, prec)

    X_cur   = [float(v) for v in X0]
    history = [X_cur[:]]
    converged = False

    for k in range(1, max_iter + 1):
        print(f"\n{C.BOLD}{C.MAGENTA}  {'─'*66}{C.RESET}")
        print(f"{C.BOLD}{C.MAGENTA}  Bước k = {k}:{C.RESET}")

        # Tính F(X_k)
        F_val = tinh_F_tai_diem(F_list, all_vars, n, X_cur)
        norm_F = max(abs(v) for v in F_val)

        print(f"\n  {C.CYAN}F(X^({k-1})):{C.RESET}")
        for i in range(n):
            print(f"    f_{i+1}(X^({k-1})) = {fmt.format(F_val[i])}")
        print(f"    ‖F(X^({k-1}))‖_∞ = {fmt.format(norm_F)}")

        # Δ = J(X0)^{-1} · (-F)  — dùng J0 cố định
        F_np  = np.array(F_val, dtype=float)
        delta_np = J0_inv @ (-F_np)
        delta = delta_np.tolist()
        in_vector(delta, f"Δ^({k-1}) = J(X⁰)⁻¹·(-F(X^({k-1})))", n, prec)

        # Cập nhật
        X_new = [X_cur[i] + delta[i] for i in range(n)]
        norm_delta = max(abs(delta[i]) for i in range(n))

        print(f"\n  {C.CYAN}X^({k}) = X^({k-1}) + Δ^({k-1}):{C.RESET}")
        for i in range(n):
            print(f"    x_{i+1}^({k}) = {fmt.format(X_cur[i])} + ({fmt.format(delta[i])}) = {fmt.format(X_new[i])}")

        # Residual mới
        F_new     = tinh_F_tai_diem(F_list, all_vars, n, X_new)
        norm_F_new = max(abs(v) for v in F_new)

        print(f"\n  {C.YELLOW}Kiểm tra dừng:{C.RESET}")
        print(f"    ‖Δ^({k-1})‖_∞ = ‖X^({k}) - X^({k-1})‖_∞ = {fmt.format(norm_delta)}")
        print(f"    ‖F(X^({k}))‖_∞ = {fmt.format(norm_F_new)}")

        history.append(X_new[:])
        X_cur = X_new[:]

        if norm_delta < eps and norm_F_new < eps:
            print(f"\n  {C.GREEN}✓ ‖ΔX‖ = {fmt.format(norm_delta)} < ε  VÀ  ‖F‖ = {fmt.format(norm_F_new)} < ε  → DỪNG{C.RESET}")
            converged = True
            break
        elif norm_delta < eps:
            print(f"\n  {C.GREEN}✓ ‖ΔX‖ = {fmt.format(norm_delta)} < ε  → DỪNG{C.RESET}")
            converged = True
            break

    return X_cur, k, converged, history


# ══════════════════════════════════════════════════════════════
#  IN KẾT QUẢ CUỐI
# ══════════════════════════════════════════════════════════════

def in_ket_qua(X_final, k, converged, history, n, eps, phuong_phap):
    box(f"KẾT QUẢ — {phuong_phap}", C.GREEN)
    prec = max(8, int(-sp.log(float(eps), 10)) + 3)
    fmt  = f"{{:.{prec}f}}"

    if converged:
        print(f"\n{C.GREEN}✓ Hội tụ sau {k} bước lặp.{C.RESET}\n")
    else:
        print(f"\n{C.YELLOW}⚠ Đạt giới hạn {k} bước, có thể chưa hội tụ.{C.RESET}\n")

    print(f"{C.BOLD}Nghiệm xấp xỉ X* ≈{C.RESET}")
    for i in range(n):
        print(f"  x_{i+1}* ≈ {fmt.format(X_final[i])}")

    # Bảng lịch sử
    print(f"\n{C.CYAN}Bảng tóm tắt:{C.RESET}\n")
    header = f"  {'k':>4} | " + " | ".join(f"{'x_'+str(i+1):>18}" for i in range(n))
    print(f"{C.YELLOW}{header}{C.RESET}")
    sep()
    for step, X in enumerate(history):
        row = f"  {step:>4} | " + " | ".join(f"{fmt.format(X[i]):>18}" for i in range(n))
        print(row)
    sep()


# ══════════════════════════════════════════════════════════════
#  NHẬP X0
# ══════════════════════════════════════════════════════════════

def nhap_X0(n, _locals):
    print(f"\n  Nhập từng thành phần của X⁰:")
    X0 = []
    for i in range(n):
        while True:
            try:
                s = input(f"  x_{i+1}⁰ = ").strip()
                v = float(sympify(s, locals=_locals))
                X0.append(v)
                print(f"  {C.GREEN}✓ x_{i+1}⁰ = {v}{C.RESET}")
                break
            except Exception as e:
                print(f"  {C.RED}Lỗi: {e}{C.RESET}")
    return X0


# ══════════════════════════════════════════════════════════════
#  CHƯƠNG TRÌNH CHÍNH
# ══════════════════════════════════════════════════════════════

def main():
    banner()

    # ── Số phương trình ──────────────────────────────────
    while True:
        try:
            n = int(input(f"{C.CYAN}Hệ có bao nhiêu phương trình? n = {C.RESET}").strip())
            if n < 1: raise ValueError
            break
        except ValueError:
            print(f"{C.RED}  Nhập số nguyên ≥ 1.{C.RESET}")

    in_ly_thuyet(n)

    # ── Tạo biến ─────────────────────────────────────────
    all_vars  = sp.symbols(f'x1:{n+1}')
    var_names = [f'x{i+1}' for i in range(n)]
    _locals   = {name: var for name, var in zip(var_names, all_vars)}
    _locals.update({'sqrt': sqrt, 'sin': sin, 'cos': cos, 'tan': tan,
                    'exp': exp, 'log': log, 'pi': pi, 'E': E,
                    'cbrt': lambda t: t ** Rational(1, 3)})

    # ── Nhập f_i ─────────────────────────────────────────
    box(f"NHẬP HỆ PHƯƠNG TRÌNH  f_i(x1,...,x{n}) = 0", C.CYAN)
    print(f"{C.YELLOW}  Biến: {', '.join(var_names)}{C.RESET}")
    print(f"  Ví dụ: 3*x1 - cos(x2*x3) - 0.5")
    print(f"         x1**2 - 81*(x2+0.1)**2 + sin(x3) + 1.06\n")

    F_list = []
    for i in range(n):
        while True:
            try:
                s = input(f"{C.CYAN}  f_{i+1} = {C.RESET}").strip()
                if not s: continue
                fi = sympify(s, locals=_locals)
                print(f"  {C.GREEN}✓ f_{i+1} = {fi}{C.RESET}")
                F_list.append(fi)
                break
            except Exception as e:
                print(f"  {C.RED}✗ Lỗi: {e}{C.RESET}")

    # ── Tính Jacobi symbolic ─────────────────────────────
    box("MA TRẬN JACOBI J(X) — SYMBOLIC", C.CYAN)
    J_sym = []
    for i, fi in enumerate(F_list):
        row = []
        for j, xj in enumerate(all_vars):
            d = simplify(diff(fi, xj))
            row.append(d)
        J_sym.append(row)
        parts = ",   ".join(f"∂f_{i+1}/∂x_{j+1} = {J_sym[i][j]}" for j in range(n))
        print(f"  Hàng {i+1}: {parts}")

    # ── Sai số ───────────────────────────────────────────
    box("THIẾT LẬP SAI SỐ", C.CYAN)
    print(f"  Nhập số chữ số thập phân (vd: 6 → ε = 10⁻⁶)")
    while True:
        try:
            k_eps = int(input(f"{C.CYAN}  Số chữ số thập phân: {C.RESET}").strip())
            if k_eps < 1: raise ValueError
            eps = 10 ** (-k_eps)
            print(f"  {C.GREEN}✓ ε = {eps} = 10^(-{k_eps}){C.RESET}")
            break
        except ValueError:
            print(f"  {C.RED}  Nhập số nguyên ≥ 1.{C.RESET}")

    # ── Chọn phương pháp ─────────────────────────────────
    box("CHỌN PHƯƠNG PHÁP", C.CYAN)
    print(f"  [1] Newton tiêu chuẩn      — J tính lại mỗi bước, hội tụ bậc 2")
    print(f"  [2] Newton Modified        — J tính 1 lần tại X⁰, hội tụ bậc 1")
    print(f"  [3] Cả hai (so sánh)")
    while True:
        ch = input(f"{C.CYAN}  Chọn [1/2/3]: {C.RESET}").strip()
        ch = ''.join(c for c in ch if c.isdigit())[:1]
        if ch in ('1', '2', '3'): break
        print(f"  {C.RED}Nhập 1, 2 hoặc 3.{C.RESET}")

    # ── Nhập X0 ──────────────────────────────────────────
    box("NHẬP ĐIỂM XUẤT PHÁT X⁰", C.CYAN)
    print(f"  Lưu ý: X⁰ cần đủ gần nghiệm để Newton hội tụ.")
    X0 = nhap_X0(n, _locals)

    # ── Chạy ─────────────────────────────────────────────
    results = {}

    if ch in ('1', '3'):
        X_N, k_N, conv_N, hist_N = newton_chuan(
            F_list, J_sym, all_vars, n, X0, eps)
        results['Newton'] = (X_N, k_N, conv_N, hist_N)
        in_ket_qua(X_N, k_N, conv_N, hist_N, n, eps, "NEWTON TIÊU CHUẨN")

    if ch in ('2', '3'):
        X_M, k_M, conv_M, hist_M = newton_modified(
            F_list, J_sym, all_vars, n, X0, eps)
        results['Modified'] = (X_M, k_M, conv_M, hist_M)
        in_ket_qua(X_M, k_M, conv_M, hist_M, n, eps, "NEWTON MODIFIED")

    # ── So sánh (nếu chạy cả 2) ──────────────────────────
    if ch == '3' and 'Newton' in results and 'Modified' in results:
        box("SO SÁNH HAI PHƯƠNG PHÁP", C.YELLOW)
        prec = max(8, int(-sp.log(float(eps), 10)) + 3)
        fmt  = f"{{:.{prec}f}}"

        X_N, k_N, conv_N, _ = results['Newton']
        X_M, k_M, conv_M, _ = results['Modified']

        print(f"\n  {'':30} {'Newton':>15} {'Newton Modified':>18}")
        sep()
        print(f"  {'Số bước lặp':30} {k_N:>15} {k_M:>18}")
        print(f"  {'Hội tụ':30} {'✓' if conv_N else '✗':>15} {'✓' if conv_M else '✗':>18}")
        for i in range(n):
            label = f"x_{i+1}*"
            print(f"  {label:30} {fmt.format(X_N[i]):>15} {fmt.format(X_M[i]):>18}")
        sep()
        print(f"\n  {C.YELLOW}Nhận xét:{C.RESET}")
        print(f"  • Newton tiêu chuẩn: hội tụ bậc 2, ít bước hơn nhưng tính J mỗi bước")
        print(f"  • Newton Modified:   hội tụ bậc 1, nhiều bước hơn nhưng J tính 1 lần")
        if conv_N and conv_M:
            diff_steps = k_M - k_N
            print(f"  • Bài này Newton tiêu chuẩn nhanh hơn {diff_steps} bước")

    # ── Thử lại ──────────────────────────────────────────
    print()
    tiep = input(f"{C.YELLOW}Giải hệ khác? [y/n]: {C.RESET}").strip().lower()
    if tiep == 'y':
        main()
    else:
        print(f"\n{C.GREEN}{C.BOLD}Hoàn thành!{C.RESET}\n")


if __name__ == "__main__":
    main()