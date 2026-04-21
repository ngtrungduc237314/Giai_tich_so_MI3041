"""
=============================================================
  CÔNG CỤ RÚT Φ(X) CHO HỆ PHƯƠNG TRÌNH PHI TUYẾN
  F(X) = 0  →  X = Φ(X)
  Lý thuyết: Hà Thị Ngọc Yến, Hà Nội 2024
=============================================================
  Nhập hệ f_1, f_2, ..., f_n = 0
  → Rút phi_i cho từng phương trình
  → Kiểm tra điều kiện hội tụ (điều kiện cần & đủ)
=============================================================
"""

import sympy as sp
from sympy import (symbols, diff, simplify, sqrt, sympify,
                   Rational, pretty, Abs, Matrix)
import warnings
warnings.filterwarnings("ignore")

# ── Màu ──────────────────────────────────────────────────
class C:
    BOLD    = '\033[1m';  CYAN    = '\033[96m';  GREEN   = '\033[92m'
    YELLOW  = '\033[93m'; RED     = '\033[91m';  MAGENTA = '\033[95m'
    BLUE    = '\033[94m'; RESET   = '\033[0m'

def box(title, color=C.CYAN):
    print(f"\n{color}{C.BOLD}{'═'*70}\n  {title}\n{'═'*70}{C.RESET}")

def sep():
    print(f"{C.BLUE}{'─'*70}{C.RESET}")

def banner():
    print(f"""{C.CYAN}{C.BOLD}
  ┌───────────────────────────────────────────────────────────┐
  │   CÔNG CỤ RÚT Φ(X) CHO HỆ PHƯƠNG TRÌNH PHI TUYẾN        │
  │   F(X) = 0  →  X = Φ(X)   (Phương pháp Lặp Đơn)         │
  └───────────────────────────────────────────────────────────┘
{C.RESET}""")


# ══════════════════════════════════════════════════════════════
#  RÚT phi_i TỪ f_i = 0 THEO BIẾN x_i (biến "chính")
# ══════════════════════════════════════════════════════════════

def rut_phi_mot(fi_expr, xi, all_vars):
    """
    Từ f_i(x_1,...,x_n) = 0, rút phi_i theo biến x_i.

    Chiến lược (theo đúng slide):
      [C1] Rút x_i bậc 1:   a*x_i + g(...) = 0  →  x_i = -g/a
      [C2] Rút x_i bậc cao: a*x_i^k + h(...) = 0  →  x_i = (−h/a)^(1/k)
      [C3] Newton thành phần: phi_i = x_i - f_i / (∂f_i/∂x_i)
    """
    results = []
    seen = set()

    def add(phi, ten, giai_thich):
        try:
            phi_s = simplify(phi)
            if phi_s == xi:         return
            if phi_s.has(sp.I):     return
            key = str(phi_s)
            if key in seen:         return
            seen.add(key)
            results.append((phi_s, ten, giai_thich))
        except Exception:
            pass

    # ── C1: Rút x_i bậc 1 ────────────────────────────────
    try:
        a1 = fi_expr.coeff(xi, 1)
        if a1 not in (0, sp.Integer(0)):
            g = fi_expr - a1 * xi
            phi_c1 = simplify(-g / a1)
            giai_thich = (
                f"  f_i = {a1}·{xi} + ({g}) = 0\n"
                f"  ↔  {xi} = ({simplify(-g)}) / ({a1})\n"
                f"  ↔  {xi} = {phi_c1}"
            )
            add(phi_c1, f"C1 — Rút {xi} bậc 1", giai_thich)
    except Exception:
        pass

    # ── C2: Rút x_i bậc cao ──────────────────────────────
    try:
        try:
            poly   = sp.Poly(fi_expr, xi)
            deg_max = poly.degree()
        except Exception:
            deg_max = 5

        for deg in range(2, deg_max + 1):
            an = fi_expr.coeff(xi, deg)
            if an in (0, sp.Integer(0)):
                continue
            h   = fi_expr - an * xi**deg
            rhs = simplify(-h / an)

            if deg % 2 == 1:
                phi_r = simplify(rhs ** Rational(1, deg))
                if phi_r.has(sp.I):
                    continue
                giai_thich = (
                    f"  f_i = {an}·{xi}^{deg} + ({h}) = 0\n"
                    f"  ↔  {xi}^{deg} = {rhs}\n"
                    f"  ↔  {xi} = ({rhs})^(1/{deg}) = {phi_r}\n"
                    f"  [Căn bậc lẻ: xác định với mọi số thực]"
                )
                add(phi_r, f"C2 — Rút {xi}^{deg} (căn bậc lẻ)", giai_thich)
            else:
                phi_r = simplify(sqrt(rhs))
                if phi_r.has(sp.I):
                    continue
                giai_thich = (
                    f"  f_i = {an}·{xi}^{deg} + ({h}) = 0\n"
                    f"  ↔  {xi}^{deg} = {rhs}\n"
                    f"  ↔  {xi} = sqrt({rhs}) = {phi_r}\n"
                    f"  ⚠  Bậc chẵn: cần {rhs} ≥ 0 trên miền D!"
                )
                add(phi_r, f"C2 — Rút {xi}^{deg} (căn bậc chẵn, cần ≥ 0)", giai_thich)
    except Exception:
        pass

    # ── C3: Newton thành phần ─────────────────────────────
    try:
        dfi = simplify(diff(fi_expr, xi))
        if dfi not in (0, sp.Integer(0)):
            phi_n = simplify(xi - fi_expr / dfi)
            if not phi_n.has(sp.I):
                giai_thich = (
                    f"  ∂f_i/∂{xi} = {dfi}\n"
                    f"  phi_i = {xi} - f_i / (∂f_i/∂{xi})\n"
                    f"        = {xi} - ({fi_expr}) / ({dfi})\n"
                    f"        = {phi_n}\n"
                    f"  [Newton thành phần: hội tụ nhanh hơn]"
                )
                add(phi_n, f"C3 — Newton thành phần cho {xi}", giai_thich)
    except Exception:
        pass

    return results


# ══════════════════════════════════════════════════════════════
#  KIỂM TRA ĐIỀU KIỆN HỘI TỤ THEO SLIDE
# ══════════════════════════════════════════════════════════════

def kiem_tra_hoi_tu(phi_list, all_vars, domain_str=None):
    """
    Kiểm tra điều kiện hội tụ cho hệ X = Φ(X).

    Theo slide (slide 7):
      Điều kiện ĐỦ: ∀ i, j:  |∂φ_i/∂x_j| ≤ K/n < 1/n  trên D
      Tức là K < 1.

    Tính 3 chuẩn thực tế (symbolic):
      - p=∞ (hàng): K = max_i Σ_j |∂φ_i/∂x_j|
      - p=1  (cột): K = max_j Σ_i |∂φ_i/∂x_j|
      - p=entry:    K = max_{i,j} |∂φ_i/∂x_j| × n   (dạng slide)
    """
    n = len(phi_list)
    box("KIỂM TRA ĐIỀU KIỆN HỘI TỤ", C.MAGENTA)

    print(f"\n{C.YELLOW}Lý thuyết (slide):{C.RESET}")
    print(f"  Điều kiện ĐỦ (Banach): Φ là ánh xạ co trên D")
    print(f"  ‖Φ(X) - Φ(Y)‖ ≤ K·‖X - Y‖   với K < 1")
    print(f"\n  Điều kiện thực tế (kiểm tra qua đạo hàm riêng):")
    print(f"  ∀ i,j: |∂φ_i/∂x_j| ≤ K/n < 1/n  trên D")
    print(f"\n  3 chuẩn hay dùng:")
    print(f"  • p=∞ (chuẩn hàng): K = max_i  Σ_j |∂φ_i/∂x_j|  < 1")
    print(f"  • p=1  (chuẩn cột): K = max_j  Σ_i |∂φ_i/∂x_j|  < 1")
    print(f"  • Dạng slide:       K = n × max_{{i,j}} |∂φ_i/∂x_j| < 1")

    sep()
    print(f"\n{C.CYAN}Ma trận Jacobi của Φ  (J_Φ):{C.RESET}")

    # Tính ma trận Jacobi Φ
    J = []
    for i, phi_i in enumerate(phi_list):
        row = []
        for j, xj in enumerate(all_vars):
            try:
                d = simplify(diff(phi_i, xj))
            except Exception:
                d = sp.Symbol('?')
            row.append(d)
        J.append(row)

    # In bảng Jacobi
    header = "  " + " | ".join(f"  ∂φ_{i+1}/∂x_{j+1}  " for j in range(n))
    print(f"\n{C.YELLOW}{header}{C.RESET}")
    sep()
    for i in range(n):
        row_str = " | ".join(f"  {str(J[i][j]):<20}" for j in range(n))
        print(f"  φ_{i+1}: {row_str}")

    sep()
    print(f"\n{C.CYAN}Biểu thức |∂φ_i/∂x_j| (symbolic):{C.RESET}")
    for i in range(n):
        for j in range(n):
            expr = Abs(J[i][j])
            print(f"  |∂φ_{i+1}/∂x_{j+1}| = |{J[i][j]}|")

    print(f"\n{C.YELLOW}Lưu ý:{C.RESET}")
    print(f"  → Để kiểm tra K < 1 chính xác, bạn cần cho biết miền D.")
    print(f"  → Tìm max của |∂φ_i/∂x_j| trên D, rồi tính K theo công thức trên.")

    # Nếu user cho domain, thử tính số
    if domain_str:
        print(f"\n{C.GREEN}Bạn đã cung cấp miền D = {domain_str}{C.RESET}")
        print(f"  → Hãy thay giá trị biên vào các biểu thức ∂φ_i/∂x_j bên trên")
        print(f"     để tìm giá trị lớn nhất, rồi tính K.")

    # In công thức K
    print(f"\n{C.CYAN}Công thức tính K (áp dụng sau khi có max trên D):{C.RESET}")
    print(f"  Đặt M = max_{{i,j}} |∂φ_i/∂x_j|  trên D")
    print(f"  • Chuẩn slide:  K = {n} × M        (cần K < 1, tức M < 1/{n} = {1/n:.4f})")
    print(f"  • Chuẩn hàng:   K = max_i Σ_j |∂φ_i/∂x_j|  (cần K < 1)")
    print(f"  • Chuẩn cột:    K = max_j Σ_i |∂φ_i/∂x_j|  (cần K < 1)")

    return J


# ══════════════════════════════════════════════════════════════
#  IN KẾT QUẢ
# ══════════════════════════════════════════════════════════════

def in_ket_qua_he(he_results, all_vars, phi_chon):
    """
    he_results: list of list, he_results[i] = list các (phi, ten, gt) cho phương trình i
    phi_chon:   list phi_i đã chọn (1 cho mỗi pt)
    """
    n = len(he_results)
    box("TÓM TẮT HỆ Φ(X) ĐÃ CHỌN", C.GREEN)

    print(f"\n{C.YELLOW}Hệ X = Φ(X):{C.RESET}")
    for i, phi in enumerate(phi_chon):
        print(f"  x_{i+1} = {phi}")

    print(f"\n{C.CYAN}Đây là hàm lặp để dùng trong chương trình giải:{C.RESET}")
    sep()
    for i, phi in enumerate(phi_chon):
        var_str = ", ".join(str(v) for v in all_vars)
        print(f"  phi_{i+1}({var_str}) = {phi}")
    sep()


# ══════════════════════════════════════════════════════════════
#  CHƯƠNG TRÌNH CHÍNH
# ══════════════════════════════════════════════════════════════

def main():
    banner()

    # ── Bước 1: Hỏi số phương trình ──────────────────────
    while True:
        try:
            n = int(input(f"{C.CYAN}Hệ có bao nhiêu phương trình? n = {C.RESET}").strip())
            if n < 2:
                print(f"{C.RED}  Cần ít nhất 2 phương trình cho hệ.{C.RESET}")
                continue
            break
        except ValueError:
            print(f"{C.RED}  Nhập số nguyên.{C.RESET}")

    # ── Bước 2: Tạo biến x_1, ..., x_n ──────────────────
    all_vars = symbols(f'x1:{n+1}')   # x1, x2, ..., xn
    var_names = [f'x{i+1}' for i in range(n)]
    var_dict  = {name: var for name, var in zip(var_names, all_vars)}

    # Thêm các hàm toán học vào local dict
    _locals = {**var_dict,
               'sqrt': sqrt, 'sin': sp.sin, 'cos': sp.cos,
               'tan': sp.tan, 'exp': sp.exp, 'log': sp.log,
               'pi': sp.pi, 'E': sp.E,
               'cbrt': lambda t: t ** Rational(1, 3)}

    print(f"\n{C.YELLOW}Biến: {', '.join(var_names)}{C.RESET}")
    print(f"{C.CYAN}Ví dụ nhập: 3*x1 - cos(x2*x3) - 0.5{C.RESET}")
    sep()

    # ── Bước 3: Nhập từng f_i ────────────────────────────
    box(f"NHẬP {n} PHƯƠNG TRÌNH f_i(x1,...,x{n}) = 0", C.MAGENTA)
    equations = []
    for i in range(n):
        while True:
            try:
                s = input(f"{C.CYAN}f_{i+1}(x1,...,x{n}) = {C.RESET}").strip()
                if not s:
                    continue
                fi = sympify(s, locals=_locals)
                print(f"{C.GREEN}  ✓ f_{i+1} = {pretty(fi, use_unicode=True)}{C.RESET}")
                equations.append(fi)
                break
            except Exception as e:
                print(f"{C.RED}  ✗ Lỗi: {e}. Thử lại.{C.RESET}")

    # ── Bước 4: Rút phi_i cho từng phương trình ──────────
    box("RÚT φ_i TỪ TỪNG PHƯƠNG TRÌNH", C.CYAN)

    he_results  = []   # he_results[i] = list (phi, ten, gt)
    phi_chon    = []   # phi chọn cuối cùng cho mỗi pt

    for i, fi in enumerate(equations):
        xi = all_vars[i]
        print(f"\n{C.BOLD}{C.MAGENTA}═══ PHƯƠNG TRÌNH {i+1}:  f_{i+1} = {fi} = 0 ═══{C.RESET}")
        print(f"{C.YELLOW}  Rút theo biến chính: {xi}{C.RESET}\n")

        results = rut_phi_mot(fi, xi, all_vars)

        if not results:
            print(f"{C.RED}  Không tìm được phi_{i+1} tự động. Cần biến đổi thủ công.{C.RESET}")
            # Dùng Newton như fallback
            try:
                dfi = simplify(diff(fi, xi))
                if dfi not in (0, sp.Integer(0)):
                    phi_fallback = simplify(xi - fi / dfi)
                    print(f"{C.YELLOW}  → Dùng Newton: phi_{i+1} = {phi_fallback}{C.RESET}")
                    he_results.append([(phi_fallback, "Newton (fallback)", "")])
                    phi_chon.append(phi_fallback)
                else:
                    he_results.append([])
                    phi_chon.append(xi)  # không đổi
            except Exception:
                he_results.append([])
                phi_chon.append(xi)
            continue

        he_results.append(results)

        # In tất cả phi tìm được
        for k, (phi, ten, gt) in enumerate(results):
            print(f"{C.BOLD}  [{k+1}] {ten}{C.RESET}")
            print(f"{C.CYAN}       φ_{i+1} = {phi}{C.RESET}")
            if gt:
                sep()
                for line in gt.split('\n'):
                    print(f"    {line}")
            # Đạo hàm gợi ý
            try:
                dp = simplify(diff(phi, xi))
                print(f"\n{C.YELLOW}       ∂φ_{i+1}/∂{xi} = {dp}{C.RESET}")
            except Exception:
                pass
            sep()

        # Chọn phi: nếu chỉ có 1 thì tự chọn, nếu nhiều hỏi user
        if len(results) == 1:
            phi_chon.append(results[0][0])
            print(f"{C.GREEN}  → Tự động chọn: φ_{i+1} = {results[0][0]}{C.RESET}")
        else:
            while True:
                try:
                    ch = int(input(f"{C.YELLOW}  Chọn phi_{i+1} [1-{len(results)}]: {C.RESET}").strip())
                    if 1 <= ch <= len(results):
                        phi_chon.append(results[ch-1][0])
                        print(f"{C.GREEN}  → Đã chọn: φ_{i+1} = {results[ch-1][0]}{C.RESET}")
                        break
                    print(f"{C.RED}  Nhập số từ 1 đến {len(results)}.{C.RESET}")
                except ValueError:
                    print(f"{C.RED}  Nhập số nguyên.{C.RESET}")

    # ── Bước 5: Kiểm tra điều kiện hội tụ ────────────────
    J = kiem_tra_hoi_tu(phi_chon, all_vars)

    # ── Bước 6: Tóm tắt Φ(X) đã chọn ────────────────────
    in_ket_qua_he(he_results, all_vars, phi_chon)

    # ── Bước 7: In hướng dẫn cuối ────────────────────────
    box("HƯỚNG DẪN TIẾP THEO", C.YELLOW)
    print(f"""
{C.CYAN}Hệ lặp đã rút:{C.RESET}""")
    for i, phi in enumerate(phi_chon):
        print(f"  x_{i+1}^(k+1) = {phi}  (thay x_j = x_j^(k))")

    print(f"""
{C.YELLOW}Kiểm tra hội tụ:{C.RESET}
  1. Xác định miền D = [a1,b1] × [a2,b2] × ... × [a{n},b{n}]
  2. Tính max |∂φ_i/∂x_j| trên D cho tất cả i, j
  3. Tính K theo chuẩn bạn chọn (chuẩn hàng khuyên dùng):
       K = max_i  Σ_j max_D |∂φ_i/∂x_j|   →  cần K < 1
  4. Nếu K < 1: phương pháp lặp hội tụ ✓
  5. Copy hệ Φ(X) sang chương trình lap_don_he.py để giải

{C.GREEN}Đánh giá sai số (slide 7):{C.RESET}
  ‖X_k - X*‖ ≤ K^n / (1-K) · ‖X_1 - X_0‖
""")

    # ── Hỏi thử lại ──────────────────────────────────────
    tiep = input(f"{C.YELLOW}Thử hệ khác? [y/n]: {C.RESET}").strip().lower()
    if tiep == 'y':
        main()
    else:
        print(f"\n{C.GREEN}{C.BOLD}Xong! Copy Φ(X) sang chương trình giải.{C.RESET}\n")


if __name__ == "__main__":
    main()