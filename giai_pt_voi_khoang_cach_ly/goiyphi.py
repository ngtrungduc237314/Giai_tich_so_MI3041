"""
=============================================================
  CÔNG CỤ RÚT φ(x) TỪ f(x) = 0  →  x = φ(x)
  Dựa trên lý thuyết: Hà Thị Ngọc Yến, Hà Nội 2024
=============================================================
  Chương trình này CHỈ làm một việc:
    Nhập f(x)  →  In ra tất cả φ(x) hợp lệ + giải thích

  Sau đó copy φ(x) sang chương trình lap_don.py để giải.
=============================================================
"""

import sympy as sp
from sympy import (symbols, diff, simplify, sqrt, sympify,
                   factor, expand, Rational, pretty)
import warnings
warnings.filterwarnings("ignore")

# ── Màu ──────────────────────────────────────────────────
class C:
    BOLD    = '\033[1m';  CYAN    = '\033[96m';  GREEN   = '\033[92m'
    YELLOW  = '\033[93m'; RED     = '\033[91m';  MAGENTA = '\033[95m'
    BLUE    = '\033[94m'; RESET   = '\033[0m'

def box(title, color=C.CYAN):
    print(f"\n{color}{C.BOLD}{'═'*66}\n  {title}\n{'═'*66}{C.RESET}")

def sep():
    print(f"{C.BLUE}{'─'*66}{C.RESET}")

def banner():
    print(f"""{C.CYAN}{C.BOLD}
  ┌─────────────────────────────────────────────────────┐
  │   CÔNG CỤ RÚT φ(x) TỪ f(x) = 0  →  x = φ(x)      │
  │   Dành cho Phương pháp Lặp Đơn                      │
  └─────────────────────────────────────────────────────┘
{C.RESET}""")


# ══════════════════════════════════════════════════════════
#  HÀM CHÍNH: RÚT φ(x)
# ══════════════════════════════════════════════════════════

def rut_phi(fx_expr, x):
    """
    Từ f(x) = 0, rút ra tất cả φ(x) hợp lệ bằng biến đổi đại số:

      ┌─────────────────────────────────────────────────────┐
      │  Nguyên tắc: f(x) = 0                              │
      │    → cô lập x ở vế trái                            │
      │    → x = φ(x)  ←  đây là hàm lặp cần tìm          │
      └─────────────────────────────────────────────────────┘

    3 phương pháp hợp lệ duy nhất:
      [C1] Rút x bậc 1:   a·x + g(x) = 0  →  x = -g(x)/a
      [C2] Rút x bậc cao: a·x^n + h(x) = 0  →  x = (-h(x)/a)^(1/n)
           • Bậc lẻ (3, 5, 7...): luôn xác định với mọi số thực
           • Bậc chẵn (2, 4...):  cần -h(x)/a ≥ 0 trên khoảng giải
      [C3] Công thức Newton: φ(x) = x - f(x)/f'(x)
           (Đây là dạng điểm cố định đặc biệt, hội tụ bậc 2)
    """
    results = []   # [(phi, ten_cach, giai_thich_day_du)]
    seen   = set()

    def add(phi, ten, giai_thich):
        try:
            phi_s = simplify(phi)
            if phi_s == x:          return   # φ(x)=x → vô nghĩa
            if phi_s.has(sp.I):     return   # phức → bỏ
            if phi_s.is_number:
                pass  # hằng số: điểm cố định trực tiếp, vẫn hợp lệ
            key = str(phi_s)
            if key in seen:         return
            seen.add(key)
            results.append((phi_s, ten, giai_thich))
        except Exception:
            pass

    # ──────────────────────────────────────────────────────
    #  C1: Rút x bậc 1
    #  f(x) = a·x + g(x) = 0
    #       ↔  a·x = -g(x)
    #       ↔  x   = -g(x) / a   ← φ(x)
    # ──────────────────────────────────────────────────────
    try:
        a1 = fx_expr.coeff(x, 1)          # hệ số của x^1

        if a1 not in (0, sp.Integer(0)):
            g  = fx_expr - a1 * x         # phần còn lại (không chứa x^1 riêng lẻ)
            phi_c1  = simplify(-g / a1)
            neg_g   = simplify(-g)

            giai_thich = (
                f"  f(x) = {a1}·x + ({g}) = 0\n"
                f"  ↔  {a1}·x = {neg_g}\n"
                f"  ↔  x = ({neg_g}) / ({a1})\n"
                f"  ↔  x = {phi_c1}"
            )
            add(phi_c1, "C1 — Rút x bậc 1", giai_thich)
    except Exception:
        pass

    # ──────────────────────────────────────────────────────
    #  C2: Rút x bậc cao (n = 2, 3, 4, 5, ...)
    #  f(x) = a·x^n + h(x) = 0
    #       ↔  a·x^n = -h(x)
    #       ↔  x^n   = -h(x) / a
    #       ↔  x     = (-h(x)/a)^(1/n)
    # ──────────────────────────────────────────────────────
    try:
        # Xác định bậc cao nhất của f(x) theo x
        try:
            poly   = sp.Poly(fx_expr, x)
            deg_max = poly.degree()
        except Exception:
            deg_max = 6   # fallback: thử đến bậc 6

        for deg in range(2, deg_max + 1):
            an = fx_expr.coeff(x, deg)

            if an in (0, sp.Integer(0)):
                continue

            # Phần còn lại: h(x) = f(x) - a·x^n
            h       = fx_expr - an * x**deg
            rhs     = simplify(-h / an)          # x^n = rhs
            neg_h   = simplify(-h)

            if deg % 2 == 1:
                # ── Căn bậc lẻ: luôn xác định ─────────────
                phi_r = simplify(rhs ** Rational(1, deg))

                if phi_r.has(sp.I):
                    continue

                # Viết rõ biểu thức căn
                rhs_str = str(simplify(neg_h / an))
                giai_thich = (
                    f"  f(x) = {an}·x^{deg} + ({h}) = 0\n"
                    f"  ↔  {an}·x^{deg} = {neg_h}\n"
                    f"  ↔  x^{deg} = ({neg_h}) / ({an})  =  {rhs_str}\n"
                    f"  ↔  x = ({rhs_str})^(1/{deg})  =  {phi_r}\n"
                    f"  [Bậc lẻ: căn bậc {deg} xác định với mọi số thực]"
                )
                add(phi_r, f"C2 — Rút x^{deg} (căn bậc lẻ {deg})", giai_thich)

            else:
                # ── Căn bậc chẵn: cần kiểm tra rhs ≥ 0 ────
                phi_r = simplify(sqrt(rhs))

                if phi_r.has(sp.I):
                    continue

                rhs_str = str(rhs)
                giai_thich = (
                    f"  f(x) = {an}·x^{deg} + ({h}) = 0\n"
                    f"  ↔  {an}·x^{deg} = {neg_h}\n"
                    f"  ↔  x^{deg} = {rhs_str}\n"
                    f"  ↔  x = sqrt({rhs_str})  =  {phi_r}\n"
                    f"  ⚠  Bậc chẵn: phải kiểm tra {rhs_str} ≥ 0 trên khoảng [a,b]!"
                )
                add(phi_r, f"C2 — Rút x^{deg} (căn bậc chẵn {deg}, cần ≥ 0)", giai_thich)

    except Exception:
        pass

    # ──────────────────────────────────────────────────────
    #  C3: Công thức Newton
    #  φ(x) = x - f(x)/f'(x)
    #
    #  Lý do: nếu đặt g(x) = x - f(x)/f'(x) thì
    #    g(α) = α  (α là nghiệm)
    #  → đây là dạng điểm cố định hợp lệ
    #  → hội tụ bậc 2 (nhanh gấp đôi so với lặp đơn thông thường)
    # ──────────────────────────────────────────────────────
    try:
        df      = diff(fx_expr, x)
        df_simp = simplify(df)

        if df_simp not in (0, sp.Integer(0)):
            phi_n = simplify(x - fx_expr / df_simp)

            if not phi_n.has(sp.I):
                giai_thich = (
                    f"  f'(x) = {df_simp}\n"
                    f"  φ(x)  = x - f(x)/f'(x)\n"
                    f"        = x - ({fx_expr}) / ({df_simp})\n"
                    f"        = {phi_n}\n"
                    f"\n"
                    f"  ─ Ý nghĩa: Newton viết lại dưới dạng điểm cố định ─\n"
                    f"    Từ f(α) = 0 ta có α = α - f(α)/f'(α)\n"
                    f"    → Tại nghiệm đúng, φ(α) = α  ✓\n"
                    f"\n"
                    f"  ─ Ưu điểm: hội tụ BẬC 2 (rất nhanh) ─\n"
                    f"    |x_{{n+1}} - α| ≤ C · |xₙ - α|²\n"
                    f"\n"
                    f"  ─ Lưu ý quan trọng: ─\n"
                    f"    • Đây KHÔNG phải là lặp đơn thuần túy\n"
                    f"    • Mỗi bước lặp cần tính thêm f'(x) → tốn hơn\n"
                    f"    • Phương pháp Newton là một trường hợp riêng\n"
                    f"      của lặp đơn với φ(x) = x - f(x)/f'(x)"
                )
                add(phi_n, "C3 — Newton  φ(x) = x - f(x)/f'(x)  [bậc 2]", giai_thich)
    except Exception:
        pass

    return results


# ══════════════════════════════════════════════════════════
#  IN KẾT QUẢ ĐẸP
# ══════════════════════════════════════════════════════════

def in_ket_qua(results, fx_expr, x):
    if not results:
        print(f"\n{C.RED}Không tìm được φ(x) hợp lệ nào bằng các phương pháp đại số.{C.RESET}")
        print(f"{C.YELLOW}Hãy thử biến đổi thủ công hoặc kiểm tra lại f(x).{C.RESET}")
        return

    box(f"KẾT QUẢ:  Tìm được {len(results)} φ(x) từ f(x) = {fx_expr}", C.GREEN)

    print(f"\n{C.YELLOW}Nhắc lại lý thuyết:{C.RESET}")
    print(f"  f(x) = 0  ↔  x = φ(x)")
    print(f"  Dãy lặp: x_{{n+1}} = φ(x_n),  x₀ ∈ [a,b]")
    print(f"  Điều kiện hội tụ: max|φ'(x)| = q < 1 trên [a,b]")
    print(f"\n{C.CYAN}{'─'*66}{C.RESET}")

    for i, (phi, ten, giai_thich) in enumerate(results):
        print(f"\n{C.BOLD}{C.MAGENTA}[{i+1}] {ten}{C.RESET}")
        print(f"{C.CYAN}     φ(x) = {phi}{C.RESET}")
        sep()
        print(f"{C.YELLOW}  Cách rút:{C.RESET}")
        for line in giai_thich.split('\n'):
            print(f"  {line}")

        # Tính đạo hàm và ghi chú hội tụ sơ bộ
        try:
            dphi = diff(phi, x)
            dphi_s = simplify(dphi)
            print(f"\n{C.YELLOW}  Đạo hàm:{C.RESET}")
            print(f"  φ'(x) = {dphi_s}")
            print(f"  → Cần kiểm tra max|φ'(x)| < 1 trên khoảng [a,b] của bạn")
        except Exception:
            pass

        # Cách copy để dùng trong lap_don.py
        print(f"\n{C.GREEN}  ✎ Copy vào lap_don.py:{C.RESET}")
        # In dạng Python string (sympy printing đôi khi không dùng được trực tiếp)
        phi_str = str(phi)
        # Đổi ** về dạng Python nếu cần (sympy đã dùng ** rồi)
        print(f"     {C.BOLD}{phi_str}{C.RESET}")
        sep()

    # Tóm tắt bảng
    box("BẢNG TÓM TẮT CÁC φ(x)", C.CYAN)
    print(f"\n  {'STT':<5} {'Cách':<40} {'φ(x)'}")
    sep()
    for i, (phi, ten, _) in enumerate(results):
        ten_ngan = ten.split('—')[1].strip() if '—' in ten else ten
        phi_str  = str(phi)
        if len(phi_str) > 35:
            phi_str = phi_str[:32] + "..."
        print(f"  [{i+1}]   {ten_ngan:<40} {phi_str}")
    sep()

    print(f"""
{C.YELLOW}HƯỚNG DẪN TIẾP THEO:{C.RESET}
  1. Xác định khoảng cách ly nghiệm [a, b]
  2. Với từng φ(x) ở trên, kiểm tra:
       • max|φ'(x)| = q < 1  trên [a,b]   ← điều kiện hội tụ
       • φ([a,b]) ⊆ [a,b]                 ← điều kiện bất biến (Định lý 1)
         hoặc (a;b) là khoảng cách ly nghiệm  (Định lý 2, thoải mái hơn)
  3. Chọn φ(x) có q nhỏ nhất → hội tụ nhanh nhất
  4. Copy φ(x) đã chọn vào chương trình lap_don.py để giải
""")


# ══════════════════════════════════════════════════════════
#  CHƯƠNG TRÌNH CHÍNH
# ══════════════════════════════════════════════════════════

def main():
    banner()

    # Hướng dẫn nhập
    box("HƯỚNG DẪN NHẬP f(x)", C.MAGENTA)
    print(f"""
{C.YELLOW}Toán tử:{C.RESET}
  **   lũy thừa   (vd: x**5, x**2)
  *    nhân        (vd: 3*x, 25*x)
  /    chia        (vd: x/2)
  +  - cộng trừ

{C.YELLOW}Hàm số:{C.RESET}
  sqrt(x)          căn bậc hai
  x**(1/3)         căn bậc ba
  sin(x), cos(x), tan(x)
  exp(x)           e^x
  log(x)           ln(x)
  log(x, 10)       log₁₀(x)
  pi, E            hằng số

{C.CYAN}Ví dụ:{C.RESET}
  x**5 + 25*x - 17          (bài tập slide)
  x**3 + 3*x**2 - 1
  x**2 + 4*sin(x) - 1
  1.4**x - x
""")

    x = symbols('x')
    _locals = {
        'x': x, 'sqrt': sqrt, 'sin': sp.sin, 'cos': sp.cos,
        'tan': sp.tan, 'exp': sp.exp, 'log': sp.log,
        'cbrt': lambda t: t ** Rational(1, 3),
        'pi': sp.pi, 'E': sp.E
    }

    # Vòng lặp: cho phép thử nhiều f(x)
    while True:
        # Nhập f(x)
        while True:
            try:
                s = input(f"{C.CYAN}Nhập f(x) = {C.RESET}").strip()
                if not s:
                    continue
                fx_expr = sympify(s, locals=_locals)
                print(f"{C.GREEN}  ✓ f(x) = {pretty(fx_expr, use_unicode=True)}{C.RESET}")
                break
            except Exception as e:
                print(f"{C.RED}  ✗ Lỗi: {e}. Thử lại.{C.RESET}")

        # Rút và in φ(x)
        results = rut_phi(fx_expr, x)
        in_ket_qua(results, fx_expr, x)

        # Hỏi có muốn thử f(x) khác không
        tiep = input(f"{C.YELLOW}Thử f(x) khác? [y/n]: {C.RESET}").strip().lower()
        if tiep != 'y':
            break

    print(f"\n{C.GREEN}{C.BOLD}Xong! Bây giờ copy φ(x) sang lap_don.py để giải.{C.RESET}\n")


if __name__ == "__main__":
    main()