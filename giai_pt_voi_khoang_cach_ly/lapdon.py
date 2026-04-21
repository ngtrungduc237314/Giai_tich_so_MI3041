"""
=============================================================
  PHƯƠNG PHÁP LẶP ĐƠN (Simple Iteration / Fixed-Point Method)
  Dựa trên tài liệu: Hà Thị Ngọc Yến, Hà Nội 2024
=============================================================
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify, simplify, Abs
from sympy import sqrt, sin, cos, exp, log, tan
from sympy import sympify
import warnings
warnings.filterwarnings("ignore")

# ── Màu sắc cho terminal ──────────────────────────────────
class C:
    BOLD      = '\033[1m'
    BLUE      = '\033[94m'
    CYAN      = '\033[96m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    MAGENTA   = '\033[95m'
    RESET     = '\033[0m'
    UNDERLINE = '\033[4m'

def box(title, color=C.CYAN):
    w = 62
    print(f"\n{color}{C.BOLD}{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}{C.RESET}")

def sep(color=C.BLUE):
    print(f"{color}{'─'*62}{C.RESET}")

def banner():
    print(f"""
{C.CYAN}{C.BOLD}
 ██╗      █████╗ ██████╗      ██████╗  ██████╗ ███╗   ██╗
 ██║     ██╔══██╗██╔══██╗     ██╔══██╗██╔═══██╗████╗  ██║
 ██║     ███████║██████╔╝     ██║  ██║██║   ██║██╔██╗ ██║
 ██║     ██╔══██║██╔═══╝      ██║  ██║██║   ██║██║╚██╗██║
 ███████╗██║  ██║██║          ██████╔╝╚██████╔╝██║ ╚████║
 ╚══════╝╚═╝  ╚═╝╚═╝          ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝
{C.RESET}
{C.YELLOW}       PHƯƠNG PHÁP LẶP ĐƠN  ─  Fixed-Point Iteration
       Hà Thị Ngọc Yến, Hà Nội 2024{C.RESET}
""")


# ══════════════════════════════════════════════════════════
#  PHẦN 1: NHẬP HÀM φ(x)
# ══════════════════════════════════════════════════════════

def huong_dan_nhap():
    box("HƯỚNG DẪN NHẬP HÀM", C.MAGENTA)
    print(f"""
{C.YELLOW}Các toán tử và hàm hợp lệ:{C.RESET}
  +  -  *  /  **      ← cộng, trừ, nhân, chia, lũy thừa
  sqrt(x)             ← căn bậc hai
  x**(1/3)            ← căn bậc ba  (hoặc cbrt(x) nếu x>0)
  sin(x), cos(x), tan(x)
  exp(x)              ← e^x
  log(x)              ← ln(x)
  log(x, 10)          ← log cơ số 10
  pi, E               ← hằng số π và e

{C.CYAN}Ví dụ φ(x):{C.RESET}
  (17 - x**5) / 25
  (1 - x**3) / 3
  (1 - 4*sin(x)) / x
  1.4**x
""")


def nhap_ham():
    x = symbols('x')
    huong_dan_nhap()

    box("NHẬP HÀM φ(x)", C.GREEN)

    _locals = {'x': x, 'sqrt': sqrt, 'sin': sp.sin, 'cos': sp.cos,
               'tan': sp.tan, 'exp': sp.exp, 'log': sp.log,
               'cbrt': lambda t: t ** sp.Rational(1, 3)}

    while True:
        try:
            s = input(f"\n{C.CYAN}Nhập φ(x) = {C.RESET}").strip()
            phi_expr = sympify(s, locals=_locals)
            print(f"{C.GREEN}  ✓ φ(x) = {sp.pretty(phi_expr, use_unicode=True)}{C.RESET}")
            break
        except Exception as e:
            print(f"{C.RED}  ✗ Lỗi: {e}. Thử lại.{C.RESET}")

    return x, phi_expr


# ══════════════════════════════════════════════════════════
#  PHẦN 2: NHẬP KHOẢNG VÀ KIỂM TRA ĐIỀU KIỆN
# ══════════════════════════════════════════════════════════

def nhap_khoang():
    box("NHẬP KHOẢNG CÁCH LY NGHIỆM [a, b]", C.CYAN)
    print(f"\n{C.YELLOW}Đây là khoảng [a,b] mà bạn đã xác định nghiệm nằm trong đó.{C.RESET}")
    while True:
        try:
            a = float(input(f"  a = "))
            b = float(input(f"  b = "))
            if a >= b:
                print(f"{C.RED}  Cần a < b!{C.RESET}")
                continue
            default_x0 = (a + b) / 2
            inp = input(f"  x₀ ∈ [{a}, {b}] (điểm bắt đầu lặp, mặc định = {default_x0:.4f}): ").strip()
            x0 = float(inp) if inp else default_x0
            if not (a <= x0 <= b):
                print(f"{C.RED}  x₀ phải nằm trong [{a}, {b}]!{C.RESET}")
                continue
            return a, b, x0
        except:
            print(f"{C.RED}  Nhập số hợp lệ!{C.RESET}")


def kiem_tra_dieu_kien(x_sym, phi_expr, a, b):
    """
    Kiểm tra:
    1. φ: [a,b] → [a,b]  (ánh xạ bất biến)
    2. |φ'(x)| ≤ q < 1   (ánh xạ co)
    Trả về (ok, q_val, dphi)
    """
    box("KIỂM TRA ĐIỀU KIỆN HỘI TỤ", C.YELLOW)

    dphi   = diff(phi_expr, x_sym)
    phi_f  = lambdify(x_sym, phi_expr, modules=['numpy', {'sqrt': np.sqrt,
                                                          'sin': np.sin, 'cos': np.cos,
                                                          'tan': np.tan, 'exp': np.exp,
                                                          'log': np.log}])
    dphi_f = lambdify(x_sym, Abs(dphi),  modules=['numpy', {'sqrt': np.sqrt,
                                                            'sin': np.sin, 'cos': np.cos,
                                                            'tan': np.tan, 'exp': np.exp,
                                                            'log': np.log}])
    xs = np.linspace(a, b, 1000)

    # ── Điều kiện 1: φ([a,b]) ⊆ [a,b] ───────────────────
    print(f"\n{C.BOLD}Điều kiện 1:{C.RESET} φ: [{a}, {b}] → [{a}, {b}]")
    try:
        phi_vals = phi_f(xs)
        phi_min  = float(np.nanmin(phi_vals))
        phi_max  = float(np.nanmax(phi_vals))
        print(f"  φ(x) ∈ [{phi_min:.6f}, {phi_max:.6f}] với x ∈ [{a}, {b}]")
        dc1 = (phi_min >= a - 1e-9) and (phi_max <= b + 1e-9)
        if dc1:
            print(f"  {C.GREEN}✓ Thỏa mãn: φ([a,b]) ⊆ [a,b]{C.RESET}")
        else:
            print(f"  {C.YELLOW}⚠ Không chắc: φ ra ngoài [{a},{b}]{C.RESET}")
            print(f"    → Tài liệu cho phép thay bằng điều kiện (a;b) là khoảng cách ly nghiệm (Định lý 2).")
    except Exception as e:
        dc1 = False
        print(f"  {C.RED}✗ Không tính được: {e}{C.RESET}")

    # ── Điều kiện 2: |φ'(x)| ≤ q < 1 ────────────────────
    print(f"\n{C.BOLD}Điều kiện 2:{C.RESET} |φ'(x)| ≤ q < 1  với mọi x ∈ [{a},{b}]")
    print(f"  φ'(x) = {dphi}")
    try:
        dphi_vals = dphi_f(xs)
        q_val = float(np.nanmax(np.abs(dphi_vals)))
        print(f"  max|φ'(x)| ≈ {q_val:.8f}")

        if q_val < 1.0:
            print(f"  {C.GREEN}✓ Thỏa mãn: q = {q_val:.6f} < 1  →  ÁNH XẠ CO{C.RESET}")
            dc2 = True
        else:
            print(f"  {C.RED}✗ KHÔNG thỏa mãn: q = {q_val:.6f} ≥ 1  →  DÃY SẼ PHÂN KỲ!{C.RESET}")
            print(f"  {C.YELLOW}  Hãy chọn lại φ(x) khác!{C.RESET}")
            dc2 = False
    except Exception as e:
        q_val = 999.0
        dc2 = False
        print(f"  {C.RED}✗ Không tính được: {e}{C.RESET}")

    # ── Tốc độ hội tụ ─────────────────────────────────────
    if dc2:
        print(f"\n{C.BOLD}Đánh giá tốc độ hội tụ:{C.RESET}")
        print(f"  Hệ số co q = {q_val:.6f}")
        if q_val < 0.1:
            print(f"  {C.GREEN}→ Rất nhanh (q < 0.1){C.RESET}")
        elif q_val < 0.5:
            print(f"  {C.GREEN}→ Nhanh (q < 0.5){C.RESET}")
        elif q_val < 0.9:
            print(f"  {C.YELLOW}→ Trung bình (0.5 ≤ q < 0.9){C.RESET}")
        else:
            print(f"  {C.RED}→ Chậm (q ≥ 0.9){C.RESET}")

    ok = dc2
    return ok, q_val, dphi


# ══════════════════════════════════════════════════════════
#  PHẦN 3: CẤU HÌNH CHẠY
# ══════════════════════════════════════════════════════════

def cau_hinh_chay():
    box("CẤU HÌNH CHẠY THUẬT TOÁN", C.CYAN)

    # Số chữ số thập phân
    while True:
        try:
            nd = int(input(f"\n{C.YELLOW}Số chữ số sau dấu phẩy muốn in ra (vd: 7): {C.RESET}"))
            if nd >= 0: break
        except: pass
        print(f"{C.RED}Nhập số nguyên không âm!{C.RESET}")

    # Kiểu dừng
    print(f"\n{C.BOLD}Điều kiện dừng:{C.RESET}")
    print(f"  [1] Sai số TIÊN NGHIỆM  →  dừng khi qⁿ/(1-q)·|x₁-x₀| < ε")
    print(f"  [2] Sai số HẬU NGHIỆM   →  dừng khi q/(1-q)·|xₙ-x_{{n-1}}| < ε")
    print(f"  [3] Dừng sau n lần lặp cố định")
    while True:
        kieu = input(f"{C.YELLOW}Chọn [1/2/3]: {C.RESET}").strip()
        if kieu in ['1', '2', '3']: break

    eps   = None
    n_lap = None

    if kieu in ['1', '2']:
        while True:
            try:
                s = input(f"  Nhập ε (vd: 0.5e-7 hoặc 1e-7): ").strip()
                eps = float(s)
                if eps > 0: break
            except: pass
            print(f"{C.RED}  Nhập số dương!{C.RESET}")
    else:
        while True:
            try:
                n_lap = int(input(f"  Số lần lặp n = "))
                if n_lap > 0: break
            except: pass
            print(f"{C.RED}  Nhập số nguyên dương!{C.RESET}")

    return nd, kieu, eps, n_lap


# ══════════════════════════════════════════════════════════
#  PHẦN 4: THỰC HIỆN LẶP + IN TỪNG BƯỚC
# ══════════════════════════════════════════════════════════

def lap_don(x_sym, phi_expr, x0, a, b, q_val, nd, kieu, eps, n_lap):
    box("QUÁ TRÌNH LẶP - TỪNG BƯỚC", C.GREEN)

    phi_f = lambdify(x_sym, phi_expr, modules=['numpy', {'sqrt': np.sqrt,
                                                         'sin': np.sin, 'cos': np.cos,
                                                         'tan': np.tan, 'exp': np.exp,
                                                         'log': np.log}])
    fmt = f".{nd}f"

    results = []   # lưu (n, xn, |xn - x_{n-1}|, sai_so_tien, sai_so_hau)

    # Tính x₁ từ x₀
    x_prev = x0
    x_cur  = float(phi_f(x0))
    d0     = abs(x_cur - x_prev)   # |x₁ - x₀|

    # ── In thông tin điều kiện dừng đang dùng ─────────────
    if kieu == '1':
        print(f"\n{C.CYAN}Điều kiện dừng TIÊN NGHIỆM:{C.RESET}")
        print(f"  |xₙ - α| ≤ qⁿ/(1-q) · |x₁-x₀| < ε")
        print(f"  q = {q_val:.6f},  |x₁-x₀| = {d0:.{nd}f},  ε = {eps}")
        # Ước lượng số bước cần
        if q_val < 1 and d0 > 1e-30:
            import math
            rhs = eps * (1 - q_val) / d0
            if rhs > 0 and rhs < 1:
                n_pred = math.ceil(math.log(rhs) / math.log(q_val))
                n_pred = max(n_pred, 1)
                print(f"\n{C.MAGENTA}  [Ước tính số bước] n ≥ log(ε·(1-q)/|x₁-x₀|) / log(q)")
                print(f"                   = log({eps}·{1-q_val:.4f}/{d0:.{nd}f}) / log({q_val:.4f})")
                print(f"                   ≈ {n_pred}  →  Cần khoảng {n_pred} bước lặp{C.RESET}")
    elif kieu == '2':
        print(f"\n{C.CYAN}Điều kiện dừng HẬU NGHIỆM:{C.RESET}")
        print(f"  |xₙ - α| ≤ q/(1-q) · |xₙ-x_{{n-1}}| < ε")
        print(f"  q = {q_val:.6f},  q/(1-q) = {q_val/(1-q_val):.6f},  ε = {eps}")
        print(f"  Dừng khi: {q_val/(1-q_val):.6f} · |Δxₙ| < {eps}")
    else:
        print(f"\n{C.CYAN}Điều kiện dừng: sau {n_lap} lần lặp cố định.{C.RESET}")

    sep()

    w = nd + 8

    # ── Header bảng ───────────────────────────────────────
    print(f"\n{'n':>4} │ {'x_n':{w}} │ {'x_{n+1} = φ(x_n)':{w}} │ {'|xₙ - xₙ₋₁|':<18} │ {'Tiên nghiệm ≤':<20} │ {'Hậu nghiệm ≤':<20}")
    sep()

    # Bước 0
    results.append((0, x_prev, 0.0, float('inf'), float('inf')))
    print(f"{0:>4} │ {x_prev:{w}{fmt}} │ {x_cur:{w}{fmt}} │ {'─':^18} │ {'─':^20} │ {'─':^20}")

    n        = 1
    MAX_ITER = 100000

    while True:
        d = abs(x_cur - x_prev)   # |xₙ - xₙ₋₁|

        # Sai số tiên nghiệm
        sai_so_tien = (q_val**n / (1 - q_val)) * d0 if q_val < 1 else float('inf')
        # Sai số hậu nghiệm
        sai_so_hau  = (q_val / (1 - q_val)) * d     if q_val < 1 else float('inf')

        x_next = float(phi_f(x_cur))

        print(f"{n:>4} │ {x_cur:{w}{fmt}} │ {x_next:{w}{fmt}} │ {d:<18.{nd}f} │ {sai_so_tien:<20.{nd}f} │ {sai_so_hau:<20.{nd}f}")

        results.append((n, x_cur, d, sai_so_tien, sai_so_hau))

        # ── Kiểm tra điều kiện dừng ───────────────────────
        if kieu == '1':
            if sai_so_tien < eps:
                print(f"\n{C.GREEN}  ✓ Dừng (tiên nghiệm): q^{n}/(1-q)·|x₁-x₀| = {sai_so_tien:.{nd}f} < ε = {eps}{C.RESET}")
                break

        elif kieu == '2':
            if sai_so_hau < eps:
                print(f"\n{C.GREEN}  ✓ Dừng (hậu nghiệm): q/(1-q)·|xₙ-xₙ₋₁| = {sai_so_hau:.{nd}f} < ε = {eps}{C.RESET}")
                break

        else:
            if n >= n_lap:
                # Thêm bước cuối
                x_prev2 = x_cur
                x_cur   = x_next
                n += 1
                d_last  = abs(x_cur - x_prev2)
                st_last = (q_val**n / (1 - q_val)) * d0 if q_val < 1 else float('inf')
                sh_last = (q_val / (1 - q_val)) * d_last if q_val < 1 else float('inf')
                results.append((n, x_cur, d_last, st_last, sh_last))
                print(f"\n{C.GREEN}  ✓ Dừng sau {n_lap} lần lặp.{C.RESET}")
                break

        x_prev = x_cur
        x_cur  = x_next
        n += 1

        if n > MAX_ITER:
            print(f"{C.RED}  ✗ Vượt {MAX_ITER} lần lặp, dừng.{C.RESET}")
            break

    sep()
    nghiem = x_cur
    print(f"\n{C.BOLD}{C.GREEN}KẾT QUẢ: x* ≈ {nghiem:{fmt}}{C.RESET}")

    # ── Tóm tắt đánh giá sai số ───────────────────────────
    box("TÓM TẮT ĐÁNH GIÁ SAI SỐ", C.MAGENTA)
    if len(results) >= 2:
        last  = results[-1]
        n_fin = last[0]
        d_fin = last[2]

        sau_tien_fin = (q_val**n_fin / (1 - q_val)) * d0 if q_val < 1 else 0
        sau_hau_fin  = (q_val / (1 - q_val)) * d_fin      if q_val < 1 else 0

        print(f"\n{C.CYAN}Công thức sai số TIÊN NGHIỆM (tại bước n = {n_fin}):{C.RESET}")
        print(f"  |x_n - α| ≤ qⁿ/(1-q) · |x₁ - x₀|")
        print(f"           ≤ {q_val:.6f}^{n_fin} / (1-{q_val:.6f}) · {d0:.{nd}f}")
        print(f"           ≤ {sau_tien_fin:.{nd}f}")

        print(f"\n{C.CYAN}Công thức sai số HẬU NGHIỆM (tại bước n = {n_fin}):{C.RESET}")
        print(f"  |x_n - α| ≤ q/(1-q) · |xₙ - x_{{n-1}}|")
        print(f"           ≤ {q_val:.6f}/(1-{q_val:.6f}) · {d_fin:.{nd}f}")
        print(f"           ≤ {sau_hau_fin:.{nd}f}")

    return nghiem, results


# ══════════════════════════════════════════════════════════
#  PHẦN 5: VẼ HÌNH
# ══════════════════════════════════════════════════════════

def ve_hinh(x_sym, phi_expr, a, b, results, nghiem):
    box("VẼ HÌNH MINH HỌA", C.CYAN)

    phi_f = lambdify(x_sym, phi_expr,
                     modules=['numpy', {'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos,
                                        'tan': np.tan, 'exp': np.exp, 'log': np.log}])

    margin = (b - a) * 0.3
    xs = np.linspace(a - margin, b + margin, 800)

    try:
        phi_vals = phi_f(xs)
    except:
        phi_vals = np.array([float(phi_expr.subs(x_sym, xi)) for xi in xs])

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.patch.set_facecolor('#0f1117')

    colors = {'bg': '#0f1117', 'panel': '#1a1d2e', 'phi': '#00d4ff',
              'diag': '#ff6b6b', 'traj': '#ffd700', 'sol': '#00ff88',
              'grid': '#2a2d3e', 'text': '#e0e0e0'}

    # ── Hình 1: Cobweb (Mạng nhện) ───────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(colors['panel'])
    ax1.spines[:].set_color(colors['grid'])

    ax1.plot(xs, phi_vals, color=colors['phi'],  linewidth=2.5, label='y = φ(x)', zorder=3)
    ax1.plot(xs, xs,       color=colors['diag'], linewidth=1.5, linestyle='--', label='y = x', zorder=3)

    xn_list = [r[1] for r in results]
    for i in range(len(xn_list) - 1):
        xi  = xn_list[i]
        xi1 = xn_list[i + 1]
        try:
            ax1.plot([xi, xi],  [xi, xi1],  color=colors['traj'], linewidth=0.9, alpha=0.7, zorder=4)
            ax1.plot([xi, xi1], [xi1, xi1], color=colors['traj'], linewidth=0.9, alpha=0.7, zorder=4)
        except:
            pass

    if xn_list:
        ax1.axvline(x=xn_list[0], color='#aaaaaa', linestyle=':', alpha=0.5)
        ax1.scatter([xn_list[0]], [xn_list[0]], color='white', s=60, zorder=6,
                    label=f'x₀={xn_list[0]:.4f}')
    ax1.axvline(x=nghiem, color=colors['sol'], linewidth=1.5, linestyle='--', alpha=0.8, zorder=5)
    ax1.scatter([nghiem], [nghiem], color=colors['sol'], s=100, zorder=7,
                label=f'x*≈{nghiem:.6f}')

    ax1.axvspan(a, b, alpha=0.08, color=colors['phi'])
    ax1.set_xlim(a - margin, b + margin)
    ax1.set_ylim(a - margin, b + margin)
    ax1.set_title('Cobweb (Mạng nhện)', color=colors['text'], fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=8, facecolor=colors['panel'], labelcolor=colors['text'], edgecolor=colors['grid'])
    ax1.tick_params(colors=colors['text'])
    ax1.set_xlabel('x', color=colors['text'])
    ax1.set_ylabel('y', color=colors['text'])
    ax1.grid(True, color=colors['grid'], alpha=0.5, linewidth=0.5)

    # ── Hình 2: Hội tụ của dãy xₙ ────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(colors['panel'])
    ax2.spines[:].set_color(colors['grid'])

    n_vals = [r[0] for r in results]
    x_vals = [r[1] for r in results]

    ax2.plot(n_vals, x_vals, 'o-', color=colors['traj'], markersize=4,
             linewidth=1.5, label='xₙ', zorder=3)
    ax2.axhline(y=nghiem, color=colors['sol'], linewidth=2, linestyle='--',
                label=f'x*≈{nghiem:.6f}', zorder=4)
    ax2.set_title('Dãy xₙ hội tụ về nghiệm', color=colors['text'],
                  fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=8, facecolor=colors['panel'], labelcolor=colors['text'], edgecolor=colors['grid'])
    ax2.tick_params(colors=colors['text'])
    ax2.set_xlabel('n (số bước lặp)', color=colors['text'])
    ax2.set_ylabel('xₙ', color=colors['text'])
    ax2.grid(True, color=colors['grid'], alpha=0.5, linewidth=0.5)

    fig.suptitle('PHƯƠNG PHÁP LẶP ĐƠN  ─  Fixed-Point Iteration',
                 color=colors['text'], fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('lap_don_result.png', dpi=150, bbox_inches='tight',
                facecolor=colors['bg'])
    print(f"\n{C.GREEN}  ✓ Đã lưu hình vào 'lap_don_result.png'{C.RESET}")
    plt.show()


# ══════════════════════════════════════════════════════════
#  CHƯƠNG TRÌNH CHÍNH
# ══════════════════════════════════════════════════════════

def main():
    banner()

    x_sym, phi_expr = nhap_ham()
    a, b, x0        = nhap_khoang()

    ok, q_val, dphi = kiem_tra_dieu_kien(x_sym, phi_expr, a, b)

    if not ok:
        print(f"\n{C.RED}{C.BOLD}✗ φ(x) không thỏa mãn điều kiện hội tụ!{C.RESET}")
        print(f"{C.YELLOW}  Hãy chạy lại và chọn φ(x) khác.{C.RESET}")
        return

    nd, kieu, eps, n_lap = cau_hinh_chay()

    nghiem, results = lap_don(x_sym, phi_expr, x0, a, b,
                              q_val, nd, kieu, eps, n_lap)

    hoi = input(f"\n{C.YELLOW}Vẽ hình minh họa? [y/n]: {C.RESET}").strip().lower()
    if hoi == 'y':
        ve_hinh(x_sym, phi_expr, a, b, results, nghiem)

    box("HOÀN THÀNH", C.GREEN)
    print(f"\n  Nghiệm gần đúng:  x* ≈ {nghiem:.{nd}f}")
    print(f"  Hệ số co:         q  = {q_val:.8f}")
    print(f"  Số bước lặp:      {results[-1][0]}\n")


if __name__ == "__main__":
    main()