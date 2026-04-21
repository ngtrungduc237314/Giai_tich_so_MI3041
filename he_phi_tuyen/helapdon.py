"""
=============================================================
  GIẢI HỆ PHƯƠNG TRÌNH PHI TUYẾN BẰNG PHƯƠNG PHÁP LẶP ĐƠN
  X = Φ(X)
  Lý thuyết: Hà Thị Ngọc Yến, Hà Nội 2024
=============================================================
"""

import sympy as sp
from sympy import (symbols, diff, simplify, sqrt, sympify,
                   Rational, pretty, Abs, pi, E, sin, cos,
                   tan, exp, log, Float)
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
  │   PHƯƠNG PHÁP LẶP ĐƠN CHO HỆ PHƯƠNG TRÌNH PHI TUYẾN        │
  │   X = Φ(X)   —   Hà Thị Ngọc Yến, Hà Nội 2024              │
  └──────────────────────────────────────────────────────────────┘
{C.RESET}""")


# ══════════════════════════════════════════════════════════════
#  LÝ THUYẾT (in ra đầu)
# ══════════════════════════════════════════════════════════════

def in_ly_thuyet(n):
    box("CƠ SỞ LÝ THUYẾT — PHƯƠNG PHÁP LẶP ĐƠN CHO HỆ", C.MAGENTA)
    print(f"""
{C.YELLOW}Bài toán:{C.RESET}
  Giải hệ F(X) = 0, tức là:
    f_1(x_1, ..., x_{n}) = 0
    f_2(x_1, ..., x_{n}) = 0
    ...
    f_{n}(x_1, ..., x_{n}) = 0

{C.YELLOW}Ý tưởng:{C.RESET}
  Viết lại tương đương:  X = Φ(X), tức là:
    x_1 = φ_1(x_1, ..., x_{n})
    x_2 = φ_2(x_1, ..., x_{n})
    ...
    x_{n} = φ_{n}(x_1, ..., x_{n})

{C.YELLOW}Dãy lặp:{C.RESET}
  Chọn X⁰ = (x_1⁰, ..., x_{n}⁰) làm điểm xuất phát.
  X^(k+1) = Φ(X^(k)),  k = 0, 1, 2, ...
  Tức là:
    x_i^(k+1) = φ_i(x_1^(k), ..., x_{n}^(k))  với mọi i

{C.YELLOW}ĐIỀU KIỆN ĐỦ hội tụ (Định lý Banach — slide):{C.RESET}
  Miền D = [a_1,b_1] × ... × [a_{n},b_{n}]  ⊂ ℝ^{n}
  (1) Φ(D) ⊆ D  (tính bất biến)
  (2) Φ là ánh xạ co:  ‖Φ(X) - Φ(Y)‖ ≤ K·‖X-Y‖,  K < 1

  Điều kiện (2) được kiểm tra qua đạo hàm riêng:
    ∀ i,j: |∂φ_i/∂x_j| ≤ K/n < 1/n  trên D
  → K = n × max_{{i,j}} |∂φ_i/∂x_j|  < 1  (chuẩn slide)
  → K = max_i Σ_j |∂φ_i/∂x_j|       < 1  (chuẩn hàng ∞-norm)
  → K = max_j Σ_i |∂φ_i/∂x_j|       < 1  (chuẩn cột 1-norm)

{C.YELLOW}ĐIỀU KIỆN CẦN:{C.RESET}
  Nếu X* là điểm bất động (nghiệm), thì cần:
    ρ(J_Φ(X*)) < 1   (bán kính phổ của ma trận Jacobi tại nghiệm)
  Thực tế: kiểm tra K < 1 trên D là đủ.

{C.YELLOW}Đánh giá sai số (tiên nghiệm):{C.RESET}
  ‖X^k - X*‖ ≤ K^k / (1-K) · ‖X¹ - X⁰‖

{C.YELLOW}Đánh giá sai số (hậu nghiệm):{C.RESET}
  ‖X^k - X*‖ ≤ K / (1-K) · ‖X^k - X^(k-1)‖
""")


# ══════════════════════════════════════════════════════════════
#  KIỂM TRA ĐIỀU KIỆN TRÊN MIỀN D (symbolic + numeric)
# ══════════════════════════════════════════════════════════════

def kiem_tra_dieu_kien(phi_list, all_vars, n, domain=None):
    """
    domain: list of (a_i, b_i) hoặc None (chỉ symbolic)
    Trả về K_hang, K_cot, K_slide (float nếu có domain, else None)
    """
    box("KIỂM TRA ĐIỀU KIỆN HỘI TỤ", C.MAGENTA)

    # ── Tính ma trận Jacobi Φ (symbolic) ─────────────────
    print(f"\n{C.CYAN}[1] Ma trận Jacobi J_Φ (đạo hàm riêng của Φ):{C.RESET}\n")
    J_sym = []
    for i, phi_i in enumerate(phi_list):
        row = []
        for j, xj in enumerate(all_vars):
            try:
                d = simplify(diff(phi_i, xj))
            except Exception:
                d = sp.Symbol('?')
            row.append(d)
        J_sym.append(row)
        # In hàng
        parts = ",  ".join(f"∂φ_{i+1}/∂x_{j+1} = {J_sym[i][j]}" for j in range(n))
        print(f"  Hàng {i+1}: {parts}")

    print(f"\n{C.CYAN}[2] Giá trị tuyệt đối |∂φ_i/∂x_j|:{C.RESET}\n")
    for i in range(n):
        for j in range(n):
            print(f"  |∂φ_{i+1}/∂x_{j+1}| = |{J_sym[i][j]}|")

    K_hang = K_cot = K_slide = None
    K_val  = None   # K dùng để đánh giá sai số

    if domain:
        print(f"\n{C.CYAN}[3] Tính max |∂φ_i/∂x_j| trên miền D:{C.RESET}")
        print(f"  D = " + " × ".join(f"[{a},{b}]" for a,b in domain))

        # Tính max bằng cách thử tại các đỉnh của D (conservative)
        import itertools
        corners = list(itertools.product(*[(a, b) for a,b in domain]))

        abs_J_max = [[0.0]*n for _ in range(n)]
        for corner in corners:
            subs = {all_vars[k]: float(corner[k]) for k in range(n)}
            for i in range(n):
                for j in range(n):
                    try:
                        val = abs(float(J_sym[i][j].subs(subs).evalf()))
                        if val > abs_J_max[i][j]:
                            abs_J_max[i][j] = val
                    except Exception:
                        pass

        print(f"\n  Bảng max |∂φ_i/∂x_j| tại các đỉnh D:\n")
        header = "         " + "".join(f"  x_{j+1}         " for j in range(n))
        print(f"{C.YELLOW}{header}{C.RESET}")
        sep()
        for i in range(n):
            row_s = "".join(f"  {abs_J_max[i][j]:.6f}    " for j in range(n))
            print(f"  φ_{i+1}:   {row_s}")
        sep()

        # K theo 3 chuẩn
        K_hang  = max(sum(abs_J_max[i][j] for j in range(n)) for i in range(n))
        K_cot   = max(sum(abs_J_max[i][j] for i in range(n)) for j in range(n))
        K_slide = n * max(abs_J_max[i][j] for i in range(n) for j in range(n))

        print(f"\n{C.CYAN}[4] Hệ số co K (kiểm tra K < 1):{C.RESET}\n")
        print(f"  Chuẩn hàng (∞-norm):  K = max_i Σ_j |∂φ_i/∂x_j| = {K_hang:.6f}  ", end="")
        print(f"{C.GREEN}✓ K < 1{C.RESET}" if K_hang < 1 else f"{C.RED}✗ K ≥ 1  (không đảm bảo hội tụ!){C.RESET}")

        print(f"  Chuẩn cột  (1-norm):  K = max_j Σ_i |∂φ_i/∂x_j| = {K_cot:.6f}  ", end="")
        print(f"{C.GREEN}✓ K < 1{C.RESET}" if K_cot < 1 else f"{C.RED}✗ K ≥ 1  (không đảm bảo hội tụ!){C.RESET}")

        print(f"  Chuẩn slide:          K = n × max|∂φ_i/∂x_j|     = {K_slide:.6f}  ", end="")
        print(f"{C.GREEN}✓ K < 1{C.RESET}" if K_slide < 1 else f"{C.RED}✗ K ≥ 1  (không đảm bảo hội tụ!){C.RESET}")

        # Chọn K nhỏ nhất trong các chuẩn < 1 để đánh giá sai số
        candidates = [k for k in [K_hang, K_cot, K_slide] if k < 1]
        K_val = min(candidates) if candidates else min(K_hang, K_cot, K_slide)

        print(f"\n  → Dùng K = {K_val:.6f} để đánh giá sai số (chọn chuẩn nhỏ nhất < 1)")

        if all(k < 1 for k in [K_hang, K_cot, K_slide]):
            print(f"\n{C.GREEN}  ✓ ĐIỀU KIỆN ĐỦ THỎA MÃN: Φ là ánh xạ co, dãy lặp chắc chắn hội tụ.{C.RESET}")
        elif any(k < 1 for k in [K_hang, K_cot, K_slide]):
            print(f"\n{C.YELLOW}  ⚠ MỘT SỐ CHUẨN < 1: Khả năng hội tụ cao, nhưng nên theo dõi dãy lặp.{C.RESET}")
        else:
            print(f"\n{C.RED}  ✗ ĐIỀU KIỆN ĐỦ KHÔNG THỎA: Hội tụ không được đảm bảo lý thuyết.{C.RESET}")
            print(f"  → Thử đổi φ_i hoặc thu nhỏ miền D.")

    else:
        print(f"\n{C.YELLOW}  Không có miền D → chỉ hiển thị symbolic. Kiểm tra bằng tay.{C.RESET}")
        print(f"  Sau khi có X⁰, sẽ ước lượng K tại điểm xuất phát.")

    return K_val, J_sym


# ══════════════════════════════════════════════════════════════
#  TÍNH K TẠI 1 ĐIỂM (dùng khi không có domain)
# ══════════════════════════════════════════════════════════════

def tinh_K_tai_diem(J_sym, all_vars, n, X0_val):
    subs = {all_vars[k]: X0_val[k] for k in range(n)}
    abs_J = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            try:
                abs_J[i][j] = abs(float(J_sym[i][j].subs(subs).evalf()))
            except Exception:
                abs_J[i][j] = 0.0
    K_hang  = max(sum(abs_J[i][j] for j in range(n)) for i in range(n))
    K_cot   = max(sum(abs_J[i][j] for i in range(n)) for j in range(n))
    K_slide = n * max(abs_J[i][j] for i in range(n) for j in range(n))
    return K_hang, K_cot, K_slide


# ══════════════════════════════════════════════════════════════
#  LẶP CHÍNH
# ══════════════════════════════════════════════════════════════

def lap_don_he(phi_list, all_vars, n, X0, eps, K_val, max_iter=200):
    """
    Thực hiện lặp X^(k+1) = Φ(X^(k)).
    In từng bước đầy đủ.
    """
    box("QUÁ TRÌNH LẶP  X^(k+1) = Φ(X^(k))", C.CYAN)

    # Ký hiệu in đẹp
    var_names = [str(v) for v in all_vars]

    print(f"\n{C.YELLOW}Hệ lặp:{C.RESET}")
    for i, phi in enumerate(phi_list):
        print(f"  x_{i+1}^(k+1) = {phi}")

    print(f"\n{C.YELLOW}Điểm xuất phát X⁰:{C.RESET}")
    print("  " + ",  ".join(f"x_{i+1}⁰ = {X0[i]}" for i in range(n)))

    print(f"\n{C.YELLOW}Sai số yêu cầu: ε = {eps}{C.RESET}")
    if K_val and K_val < 1:
        print(f"{C.YELLOW}Hệ số co K = {K_val:.6f}{C.RESET}")
        # Ước lượng số bước cần thiết (tiên nghiệm) — cần X1 trước, tính sau
    sep('═')

    X_cur = [float(x) for x in X0]
    history = [X_cur[:]]

    # Tiêu chuẩn dừng: ‖X^(k+1) - X^k‖_∞ < ε(1-K)/K  (hậu nghiệm)
    # hoặc đơn giản hơn: max|x_i^(k+1) - x_i^k| < ε
    stop_threshold = eps
    if K_val and K_val < 1:
        stop_threshold = eps * (1 - K_val) / K_val
        print(f"\n{C.CYAN}Tiêu chuẩn dừng (hậu nghiệm):{C.RESET}")
        print(f"  ‖X^(k+1) - X^k‖ < ε·(1-K)/K = {eps}·(1-{K_val:.4f})/{K_val:.4f} = {stop_threshold:.2e}")
    else:
        print(f"\n{C.CYAN}Tiêu chuẩn dừng:{C.RESET}")
        print(f"  max_i |x_i^(k+1) - x_i^k| < ε = {eps}")

    print()

    converged = False
    k = 0
    X_new = None

    for k in range(1, max_iter + 1):
        # Thay X_cur vào Φ
        subs = {all_vars[i]: X_cur[i] for i in range(n)}
        X_new = []
        ok = True
        for i, phi_i in enumerate(phi_list):
            try:
                val = float(phi_i.subs(subs).evalf())
                if val != val:   # NaN
                    ok = False; break
                X_new.append(val)
            except Exception:
                ok = False; break

        if not ok:
            print(f"{C.RED}  Lỗi tính toán tại bước k={k}. Dừng.{C.RESET}")
            break

        # Hiệu ‖X^k - X^(k-1)‖
        diff_vec = [abs(X_new[i] - X_cur[i]) for i in range(n)]
        norm_diff = max(diff_vec)

        # ── In bước lặp ──────────────────────────────────
        print(f"{C.BOLD}{C.MAGENTA}  Bước k = {k}:{C.RESET}")

        # Thay số vào phi
        for i, phi_i in enumerate(phi_list):
            val_str = " = ".join(
                f"φ_{i+1}({', '.join(f'{X_cur[j]:.{max(6,int(-sp.log(eps,10))+2)}f}' for j in range(n))})"
                if i == 0 else ""
            )
            prec = max(8, int(-sp.log(float(eps), 10)) + 3) if eps > 0 else 8
            fmt = f"{{:.{prec}f}}"
            print(f"    x_{i+1}^({k}) = φ_{i+1}({', '.join(fmt.format(X_cur[j]) for j in range(n))}) = {fmt.format(X_new[i])}")

        # Hiệu và chuẩn
        prec = max(8, int(-sp.log(float(eps), 10)) + 3) if eps > 0 else 8
        fmt  = f"{{:.{prec}f}}"
        diff_str = ", ".join(f"|x_{i+1}^({k}) - x_{i+1}^({k-1})| = {fmt.format(diff_vec[i])}" for i in range(n))
        print(f"    Hiệu: {diff_str}")
        print(f"    ‖X^({k}) - X^({k-1})‖_∞ = {fmt.format(norm_diff)}")

        # Đánh giá sai số hậu nghiệm
        if K_val and K_val < 1:
            err_post = K_val / (1 - K_val) * norm_diff
            print(f"    Sai số hậu nghiệm: K/(1-K)·‖ΔX‖ = {K_val:.4f}/{1-K_val:.4f} × {fmt.format(norm_diff)} = {fmt.format(err_post)}")

        # Tiêu chuẩn dừng
        if norm_diff < stop_threshold:
            print(f"\n    {C.GREEN}✓ ‖X^({k}) - X^({k-1})‖ = {fmt.format(norm_diff)} < {stop_threshold:.2e}  → DỪNG{C.RESET}")
            converged = True
            history.append(X_new[:])
            X_cur = X_new[:]
            break

        print()
        history.append(X_new[:])
        X_cur = X_new[:]

    return X_cur, k, converged, history, K_val


# ══════════════════════════════════════════════════════════════
#  IN KẾT QUẢ CUỐI
# ══════════════════════════════════════════════════════════════

def in_ket_qua_cuoi(X_final, k, converged, history, n, eps, K_val):
    box("KẾT QUẢ", C.GREEN)
    prec = max(8, int(-sp.log(float(eps), 10)) + 3) if eps > 0 else 8
    fmt  = f"{{:.{prec}f}}"

    if converged:
        print(f"\n{C.GREEN}✓ Phương pháp lặp HỘI TỤ sau {k} bước lặp.{C.RESET}\n")
    else:
        print(f"\n{C.YELLOW}⚠ Đạt giới hạn số bước lặp ({k} bước), có thể chưa đủ chính xác.{C.RESET}\n")

    print(f"{C.BOLD}Nghiệm xấp xỉ X* ≈{C.RESET}")
    for i in range(n):
        print(f"  x_{i+1}* ≈ {fmt.format(X_final[i])}")

    if K_val and K_val < 1 and len(history) >= 2:
        norm_last = max(abs(history[-1][i] - history[-2][i]) for i in range(n))
        err_bound = K_val / (1 - K_val) * norm_last
        print(f"\n{C.CYAN}Đánh giá sai số cuối (hậu nghiệm):{C.RESET}")
        print(f"  ‖X^({k}) - X*‖ ≤ K/(1-K) · ‖X^({k}) - X^({k-1})‖")
        print(f"             ≤ {K_val:.6f}/{1-K_val:.6f} × {norm_last:.2e}")
        print(f"             ≤ {err_bound:.2e}")

    # Bảng lịch sử lặp
    print(f"\n{C.CYAN}Bảng tóm tắt các bước lặp:{C.RESET}\n")
    header = f"  {'k':>4} | " + " | ".join(f"{'x_'+str(i+1):>18}" for i in range(n))
    print(f"{C.YELLOW}{header}{C.RESET}")
    sep()
    for step, X in enumerate(history):
        row = f"  {step:>4} | " + " | ".join(f"{fmt.format(X[i]):>18}" for i in range(n))
        print(row)
    sep()


# ══════════════════════════════════════════════════════════════
#  XÁC ĐỊNH ĐIỂM XUẤT PHÁT TỪ KHOẢNG CÁCH LY
# ══════════════════════════════════════════════════════════════

def chon_X0_tu_khoang(n, domain):
    """Lấy trung điểm của từng khoảng làm X0."""
    X0 = [(a + b) / 2 for a, b in domain]
    print(f"\n{C.CYAN}  Điểm xuất phát X⁰ = trung điểm của D:{C.RESET}")
    for i in range(n):
        a, b = domain[i]
        print(f"    x_{i+1}⁰ = ({a} + {b}) / 2 = {X0[i]}")
    return X0


# ══════════════════════════════════════════════════════════════
#  CHƯƠNG TRÌNH CHÍNH
# ══════════════════════════════════════════════════════════════

def main():
    banner()


    # ── Bước 1: Số phương trình ───────────────────────────
    while True:
        try:
            n = int(input(f"\n{C.CYAN}Hệ có bao nhiêu phương trình? n = {C.RESET}").strip())
            if n < 1:
                raise ValueError
            break
        except ValueError:
            print(f"{C.RED}  Nhập số nguyên ≥ 1.{C.RESET}")

    in_ly_thuyet(n)

    # ── Bước 2: Tạo biến ─────────────────────────────────
    all_vars = sp.symbols(f'x1:{n+1}')
    var_names = [f'x{i+1}' for i in range(n)]
    _locals = {name: var for name, var in zip(var_names, all_vars)}
    _locals.update({'sqrt': sqrt, 'sin': sin, 'cos': cos, 'tan': tan,
                    'exp': exp, 'log': log, 'pi': pi, 'E': E,
                    'cbrt': lambda t: t ** Rational(1, 3)})

    # ── Bước 3: Nhập φ_i ──────────────────────────────────
    box(f"NHẬP HÀM LẶP  φ_i  (đã rút từ f_i = 0)", C.CYAN)
    print(f"{C.YELLOW}  Biến: {', '.join(var_names)}{C.RESET}")
    print(f"  Ví dụ: (cos(x2*x3) + 0.5)/3")
    print(f"         sqrt(x1**2 + sin(x3) + 1.06)/9 - 0.1\n")

    phi_list = []
    for i in range(n):
        while True:
            try:
                s = input(f"{C.CYAN}  φ_{i+1}({', '.join(var_names)}) = {C.RESET}").strip()
                if not s:
                    continue
                phi_i = sympify(s, locals=_locals)
                print(f"  {C.GREEN}✓ φ_{i+1} = {phi_i}{C.RESET}")
                phi_list.append(phi_i)
                break
            except Exception as e:
                print(f"  {C.RED}✗ Lỗi: {e}{C.RESET}")

    # ── Bước 4: Sai số ────────────────────────────────────
    box("THIẾT LẬP SAI SỐ", C.CYAN)
    print(f"  Chọn độ chính xác (số chữ số thập phân sau dấu phẩy):")
    print(f"  Ví dụ: nhập 4 → ε = 0.0001 = 10⁻⁴")
    while True:
        try:
            so_chu_so = int(input(f"{C.CYAN}  Số chữ số thập phân: {C.RESET}").strip())
            if so_chu_so < 1:
                raise ValueError
            eps = 10 ** (-so_chu_so)
            print(f"  {C.GREEN}✓ ε = {eps} = 10^(-{so_chu_so}){C.RESET}")
            break
        except ValueError:
            print(f"  {C.RED}  Nhập số nguyên ≥ 1.{C.RESET}")

    # ── Bước 5: Kiểu nhập điểm xuất phát ─────────────────
    box("KIỂU BÀI TOÁN", C.CYAN)
    print(f"  [1] Cho sẵn khoảng cách ly nghiệm D = [a_1,b_1] × ... × [a_{n},b_{n}]")
    print(f"  [2] Cho sẵn điểm xuất phát (nghiệm xấp xỉ ban đầu) X⁰ trực tiếp")
    while True:
        kieu = input(f"{C.CYAN}  Chọn [1/2]: {C.RESET}").strip()
        # Chỉ lấy ký tự số đầu tiên nếu có
        kieu = ''.join(c for c in kieu if c.isdigit())[:1]
        if kieu in ('1', '2'):
            break
        print(f"  {C.RED}Nhập 1 hoặc 2.{C.RESET}")

    domain = None
    X0     = None

    if kieu == '1':
        # ── Nhập khoảng cách ly ───────────────────────────
        box("NHẬP KHOẢNG CÁCH LY NGHIỆM", C.CYAN)
        print(f"  Nhập [a_i, b_i] cho từng biến x_i.\n")
        domain = []
        for i in range(n):
            while True:
                try:
                    a_s = input(f"  a_{i+1} (cận dưới x_{i+1}): ").strip()
                    b_s = input(f"  b_{i+1} (cận trên x_{i+1}): ").strip()
                    a_v = float(sympify(a_s))
                    b_v = float(sympify(b_s))
                    if a_v >= b_v:
                        print(f"  {C.RED}Cần a < b.{C.RESET}")
                        continue
                    domain.append((a_v, b_v))
                    print(f"  {C.GREEN}✓ x_{i+1} ∈ [{a_v}, {b_v}]{C.RESET}\n")
                    break
                except Exception as e:
                    print(f"  {C.RED}Lỗi: {e}{C.RESET}")

        # Kiểm tra điều kiện hội tụ trên D
        K_val, J_sym = kiem_tra_dieu_kien(phi_list, all_vars, n, domain)

        # Chọn X0 = trung điểm
        X0 = chon_X0_tu_khoang(n, domain)
        print(f"\n  {C.YELLOW}(Bạn có thể đổi X⁰ nếu muốn. Nhấn Enter để dùng trung điểm.){C.RESET}")
        doi = input(f"  Đổi X⁰? [y/Enter]: ").strip().lower()
        if doi == 'y':
            X0 = nhap_X0(n)

    else:
        # ── Nhập X0 trực tiếp ─────────────────────────────
        box("NHẬP ĐIỂM XUẤT PHÁT X⁰", C.CYAN)
        X0 = nhap_X0(n)

        # Kiểm tra điều kiện symbolic + tại điểm X0
        K_val, J_sym = kiem_tra_dieu_kien(phi_list, all_vars, n, domain=None)

        print(f"\n{C.CYAN}Ước lượng K tại X⁰ = ({', '.join(str(v) for v in X0)}):{C.RESET}")
        K_hang, K_cot, K_slide = tinh_K_tai_diem(J_sym, all_vars, n, X0)
        print(f"  Chuẩn hàng:  K = {K_hang:.6f}  {'✓ < 1' if K_hang<1 else '✗ ≥ 1'}")
        print(f"  Chuẩn cột:   K = {K_cot:.6f}  {'✓ < 1' if K_cot<1 else '✗ ≥ 1'}")
        print(f"  Chuẩn slide: K = {K_slide:.6f}  {'✓ < 1' if K_slide<1 else '✗ ≥ 1'}")

        candidates = [k for k in [K_hang, K_cot, K_slide] if k < 1]
        K_val = min(candidates) if candidates else min(K_hang, K_cot, K_slide)
        print(f"\n  → Dùng K = {K_val:.6f} để đánh giá sai số.")

        if all(k >= 1 for k in [K_hang, K_cot, K_slide]):
            print(f"\n  {C.RED}✗ K ≥ 1 tại X⁰. Hội tụ không được đảm bảo!{C.RESET}")
            print(f"  {C.YELLOW}  Thử đổi φ_i hoặc chọn X⁰ khác.{C.RESET}")
            cont = input(f"  Tiếp tục lặp thử? [y/n]: ").strip().lower()
            if cont != 'y':
                return

    # ── Bước 6: Thực hiện lặp ────────────────────────────
    X_final, k_final, converged, history, K_used = lap_don_he(
        phi_list, all_vars, n, X0, eps, K_val
    )

    # ── Bước 7: In kết quả cuối ───────────────────────────
    in_ket_qua_cuoi(X_final, k_final, converged, history, n, eps, K_used)

    # ── Hỏi thử lại ──────────────────────────────────────
    print()
    tiep = input(f"{C.YELLOW}Giải hệ khác? [y/n]: {C.RESET}").strip().lower()
    if tiep == 'y':
        main()
    else:
        print(f"\n{C.GREEN}{C.BOLD}Hoàn thành!{C.RESET}\n")


def nhap_X0(n):
    print(f"  Nhập từng thành phần của X⁰:\n")
    X0 = []
    for i in range(n):
        while True:
            try:
                s = input(f"  x_{i+1}⁰ = ").strip()
                v = float(sympify(s))
                X0.append(v)
                print(f"  {C.GREEN}✓ x_{i+1}⁰ = {v}{C.RESET}")
                break
            except Exception as e:
                print(f"  {C.RED}Lỗi: {e}{C.RESET}")
    return X0


if __name__ == "__main__":
    main()