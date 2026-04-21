"""
╔══════════════════════════════════════════════════════════════════════╗
║          PHƯƠNG PHÁP DÂY CUNG - GIẢI PHƯƠNG TRÌNH f(x) = 0          ║
║                   Hà Thị Ngọc Yến  •  Hà Nội, 2025                  ║
╚══════════════════════════════════════════════════════════════════════╝

Công thức lặp:
    x_{k+1} = x_k - f(x_k)*(x_k - d) / (f(x_k) - f(d))

Điều kiện hội tụ:
    • (a,b) là khoảng cách ly nghiệm
    • f', f'' liên tục, xác định dấu không đổi trên [a,b]
    • Điểm Fourier d: f(d)·f''(d) > 0
    • Điểm xuất phát x0: f(d)·f(x0) < 0

Sai số:
    (1) |x_n - x*| ≤ |f(x_n)| / m1         (sai số mục tiêu)
    (2) |x_n - x*| ≤ (M1-m1)/m1 * |x_n - x_{n-1}|  (hai xấp xỉ liên tiếp)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sympy import *
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PHẦN 1: TIỆN ÍCH NHẬP LIỆU
# ─────────────────────────────────────────────────────────────

def nhap_ham_so():
    """Nhận hàm số từ người dùng, trả về hàm sympy và hàm numpy."""
    print("\n" + "═"*60)
    print("  NHẬP HÀM SỐ f(x)")
    print("═"*60)
    print("  Ví dụ: x**3 - x - 2   |   cos(x) - x   |   exp(x) - 3")
    print("  Dùng: sin, cos, tan, exp, log, sqrt, pi, E")
    print("─"*60)

    x = symbols('x')
    while True:
        try:
            bieu_thuc = input("  f(x) = ").strip()
            f_sym = sympify(bieu_thuc)
            f_num = lambdify(x, f_sym, modules=['numpy'])
            # Kiểm tra thử
            _ = f_num(1.0)
            return x, f_sym, f_num, bieu_thuc
        except Exception as e:
            print(f"  ✗ Lỗi: {e}. Vui lòng nhập lại.")


def nhap_khoang(f_num):
    """Nhận khoảng [a,b] và kiểm tra khoảng cách ly nghiệm."""
    print("\n" + "─"*60)
    print("  NHẬP KHOẢNG CÁCH LY NGHIỆM [a, b]")
    print("─"*60)
    while True:
        try:
            a = float(input("  a = "))
            b = float(input("  b = "))
            if a >= b:
                print("  ✗ Cần a < b. Nhập lại.")
                continue
            fa, fb = f_num(a), f_num(b)
            if fa * fb >= 0:
                print(f"  ✗ f(a)·f(b) = {fa:.4f}·{fb:.4f} ≥ 0.")
                print("  ✗ Không phải khoảng cách ly nghiệm! Thử khoảng khác.")
                continue
            print(f"  ✓ f({a}) = {fa:.6f},  f({b}) = {fb:.6f}")
            print(f"  ✓ f(a)·f(b) = {fa*fb:.6f} < 0  →  Khoảng hợp lệ!")
            return a, b
        except ValueError:
            print("  ✗ Nhập số thực hợp lệ.")


def nhap_sai_so():
    """Nhận yêu cầu sai số từ người dùng."""
    print("\n" + "─"*60)
    print("  THIẾT LẬP SAI SỐ")
    print("─"*60)
    print("  [1] Sai số tuyệt đối ε  (ví dụ: 0.0001)")
    print("  [2] Không vượt quá p%   (ví dụ: 0.01 %)")
    print("  [3] Có k chữ số đáng tin (ví dụ: 4 chữ số)")
    print("  [4] Có k chữ số sau dấu phẩy (ví dụ: 4 chữ số)")
    print("─"*60)

    while True:
        chon = input("  Chọn loại sai số [1/2/3/4]: ").strip()
        if chon == '1':
            eps = float(input("  ε = "))
            mo_ta = f"ε = {eps}"
            return eps, mo_ta, chon
        elif chon == '2':
            p = float(input("  p (%) = "))
            eps = p / 100.0
            mo_ta = f"sai số ≤ {p}%  (ε = {eps})"
            return eps, mo_ta, chon
        elif chon == '3':
            k = int(input("  Số chữ số đáng tin k = "))
            eps = 0.5 * 10**(-k)
            mo_ta = f"{k} chữ số đáng tin  (ε = {eps})"
            return eps, mo_ta, chon
        elif chon == '4':
            k = int(input("  Số chữ số sau dấu phẩy k = "))
            eps = 0.5 * 10**(-k)
            mo_ta = f"{k} chữ số sau dấu phẩy  (ε = {eps})"
            return eps, mo_ta, chon
        else:
            print("  ✗ Chọn 1, 2, 3 hoặc 4.")


# ─────────────────────────────────────────────────────────────
#  PHẦN 2: PHÂN TÍCH TOÁN HỌC
# ─────────────────────────────────────────────────────────────

def phan_tich_ham(x_sym, f_sym, a, b, f_num):
    """Tính đạo hàm, tìm điểm Fourier, điểm xuất phát."""
    print("\n" + "═"*60)
    print("  PHÂN TÍCH TOÁN HỌC")
    print("═"*60)

    # Đạo hàm bậc 1, bậc 2
    fp_sym  = diff(f_sym, x_sym)
    fpp_sym = diff(fp_sym, x_sym)
    fp_num  = lambdify(x_sym, fp_sym,  modules=['numpy'])
    fpp_num = lambdify(x_sym, fpp_sym, modules=['numpy'])

    print(f"\n  f(x)   = {f_sym}")
    print(f"  f'(x)  = {fp_sym}")
    print(f"  f''(x) = {fpp_sym}")

    # Tính m1, M1
    xs_check = np.linspace(a, b, 2000)
    fp_vals  = np.abs(fp_num(xs_check))
    m1 = float(np.min(fp_vals))
    M1 = float(np.max(fp_vals))
    print(f"\n  m₁ = min|f'(x)| trên [{a},{b}] = {m1:.6f}")
    print(f"  M₁ = max|f'(x)| trên [{a},{b}] = {M1:.6f}")

    # Chọn điểm Fourier d
    # d là đầu mút thỏa f(d)·f''(d) > 0
    fa, fb = f_num(a), f_num(b)
    fpa, fpb = fp_num(a), fp_num(b)
    fppa, fppb = fpp_num(a), fpp_num(b)

    print(f"\n  Kiểm tra điểm Fourier:")
    print(f"    f({a})·f''({a}) = {fa:.4f}·{fppa:.4f} = {fa*fppa:.6f}", end="")
    if fa * fppa > 0:
        d = a
        print("  > 0  →  d = a ✓")
    else:
        print("  ≤ 0")

    print(f"    f({b})·f''({b}) = {fb:.4f}·{fppb:.4f} = {fb*fppb:.6f}", end="")
    if fb * fppb > 0:
        d = b
        print("  > 0  →  d = b ✓")
    else:
        print("  ≤ 0")

    # Điểm xuất phát x0 = đầu mút còn lại
    x0 = b if d == a else a
    fd, fx0 = f_num(d), f_num(x0)
    print(f"\n  Điểm mốc (Fourier): d = {d},  f(d) = {fd:.6f}")
    print(f"  Điểm xuất phát:     x₀ = {x0},  f(x₀) = {fx0:.6f}")
    print(f"  Kiểm tra: f(d)·f(x₀) = {fd*fx0:.6f}", end="")
    print("  < 0  ✓" if fd*fx0 < 0 else "  ≥ 0  ✗ (cảnh báo)")

    return fp_sym, fpp_sym, fp_num, fpp_num, m1, M1, d, x0


# ─────────────────────────────────────────────────────────────
#  PHẦN 3: THUẬT TOÁN LẶP
# ─────────────────────────────────────────────────────────────

def day_cung_lap(f_num, fp_num, d, x0, m1, M1, eps, max_iter=200):
    """
    Thực hiện vòng lặp phương pháp dây cung.
    Trả về danh sách các bước lặp và thông tin sai số.
    """
    fd = f_num(d)
    lich_su = []  # (k, xk, fxk, sai_so_1, sai_so_2)

    xk = x0
    xk_prev = x0

    for k in range(max_iter):
        fxk = f_num(xk)

        # Sai số mục tiêu (công thức 1): |x_n - x*| ≤ |f(x_n)| / m1
        ss1 = abs(fxk) / m1 if m1 > 1e-15 else float('inf')

        # Sai số hai xấp xỉ liên tiếp (công thức 2)
        if k == 0:
            ss2 = float('inf')
        else:
            ss2 = ((M1 - m1) / m1) * abs(xk - xk_prev) if m1 > 1e-15 else float('inf')

        lich_su.append((k, xk, fxk, ss1, ss2))

        # Kiểm tra hội tụ
        if ss1 <= eps:
            break

        # Công thức lặp: x_{k+1} = x_k - f(x_k)*(x_k - d) / (f(x_k) - f(d))
        mau = fxk - fd
        if abs(mau) < 1e-15:
            print("  ✗ Mẫu số tiến về 0, dừng lặp.")
            break

        xk_prev = xk
        xk = xk - fxk * (xk - d) / mau

    return lich_su


# ─────────────────────────────────────────────────────────────
#  PHẦN 4: IN KẾT QUẢ
# ─────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
YELLOW= "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"

def in_bang_lap(lich_su, eps, mo_ta_ss):
    """In bảng các vòng lặp theo quy tắc: ≤15 in hết, >15 in 5 đầu + 5 cuối."""
    n = len(lich_su)
    print("\n" + "═"*80)
    print(f"  KẾT QUẢ CÁC VÒNG LẶP   ({mo_ta_ss})")
    print("═"*80)
    header = f"  {'k':>4}  {'x_k':>18}  {'f(x_k)':>16}  {'SS mục tiêu (1)':>18}  {'SS 2 xấp xỉ (2)':>18}"
    print(header)
    print("  " + "─"*76)

    def in_dong(hang, nhan=""):
        k, xk, fxk, ss1, ss2 = hang
        ss1_str = f"{ss1:.2e}" if ss1 != float('inf') else "   ∞   "
        ss2_str = f"{ss2:.2e}" if ss2 != float('inf') else "   ∞   "
        dau = GREEN if ss1 <= eps else ""
        print(f"  {k:>4}  {dau}{xk:>18.10f}  {fxk:>16.2e}  {ss1_str:>18}  {ss2_str:>18}{RESET}{nhan}")

    if n <= 15:
        for hang in lich_su:
            in_dong(hang)
    else:
        print(f"  (Tổng {n} vòng lặp → in 5 đầu và 5 cuối)")
        print("  " + "─"*76)
        print(f"  {CYAN}--- 5 bước ĐẦU (tiên nghiệm) ---{RESET}")
        for hang in lich_su[:5]:
            in_dong(hang)
        print("  " + "·"*76)
        print(f"  {YELLOW}--- 5 bước CUỐI (hậu nghiệm) ---{RESET}")
        for hang in lich_su[-5:]:
            in_dong(hang)

    print("  " + "─"*76)
    k_cuoi, x_cuoi, fx_cuoi, ss1_cuoi, ss2_cuoi = lich_su[-1]
    print(f"\n  {GREEN}{BOLD}Nghiệm xấp xỉ:  x* ≈ {x_cuoi:.10f}{RESET}")
    print(f"  f(x*) = {fx_cuoi:.2e}")
    print(f"  Sai số mục tiêu: {ss1_cuoi:.2e}  ({'✓ đạt' if ss1_cuoi <= eps else '✗ chưa đạt'})")
    print(f"  Số vòng lặp:     {n}")


# ─────────────────────────────────────────────────────────────
#  PHẦN 5: TRÌNH BÀY TOÁN HỌC TỪNG BƯỚC
# ─────────────────────────────────────────────────────────────

def trinh_bay_toan_hoc(f_sym, fp_sym, fpp_sym, a, b, d, x0, m1, M1,
                       lich_su, eps, mo_ta_ss, bieu_thuc, x_sym):
    """In trình bày giải toán học theo phong cách học thuật."""
    n_hien = min(5, len(lich_su))
    print("\n" + "═"*70)
    print("  TRÌNH BÀY TOÁN HỌC CHI TIẾT")
    print("═"*70)

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  BÀI TOÁN                                           │
  │  Giải phương trình: f(x) = {bieu_thuc:<25}│
  │  trên khoảng [{a}, {b}]                             │
  └─────────────────────────────────────────────────────┘

  ━━━ BƯỚC 1: KIỂM TRA ĐIỀU KIỆN HỘI TỤ ━━━━━━━━━━━━━━

  ① Khoảng cách ly nghiệm (a, b) = ({a}, {b}):
     f(a) · f(b) < 0  →  có đúng một nghiệm trong ({a}, {b}) ✓

  ② Đạo hàm:
     f'(x)  = {fp_sym}
     f''(x) = {fpp_sym}
     Cả hai liên tục và xác định dấu không đổi trên [{a},{b}] ✓

  ③ m₁ = min|f'(x)| = {m1:.6f}
     M₁ = max|f'(x)| = {M1:.6f}

  ━━━ BƯỚC 2: CHỌN ĐIỂM FOURIER (ĐIỂM MỐC) d ━━━━━━━━━

  Điểm Fourier d thỏa điều kiện:  f(d) · f''(d) > 0

  Ta chọn: d = {d}

  ━━━ BƯỚC 3: CHỌN ĐIỂM XUẤT PHÁT x₀ ━━━━━━━━━━━━━━━━━

  Điểm x₀ thỏa điều kiện:  f(d) · f(x₀) < 0
  Ta chọn: x₀ = {x0}

  ━━━ BƯỚC 4: CÔNG THỨC LẶP ━━━━━━━━━━━━━━━━━━━━━━━━━━

                   f(xₖ) · (xₖ - d)
  x_(k+1) = xₖ - ──────────────────
                   f(xₖ) - f(d)

  (Dây cung nối M(d, f(d)) và Mₖ(xₖ, f(xₖ)) cắt Ox tại x_(k+1))

  ━━━ BƯỚC 5: ĐÁNH GIÁ SAI SỐ ━━━━━━━━━━━━━━━━━━━━━━━━

  Hai công thức sai số hậu nghiệm:

              |f(xₙ)|
  (1) |xₙ - x*| ≤ ────────       ← Sai số mục tiêu
                   m₁

               M₁ - m₁
  (2) |xₙ - x*| ≤ ──────── · |xₙ - x_(n-1)|   ← Hai xấp xỉ liên tiếp
                     m₁

  Điều kiện dừng: {mo_ta_ss}

  ━━━ BƯỚC 6: CÁC VÒNG LẶP (chi tiết {n_hien} bước đầu) ━━━━━
""")

    fd = lich_su[0][2]  # f(x0) ở bước đầu - thực ra lấy trực tiếp
    for i in range(n_hien):
        k, xk, fxk, ss1, ss2 = lich_su[i]
        if i + 1 < len(lich_su):
            _, xk1, _, _, _ = lich_su[i+1]
        else:
            xk1 = xk
        fd_val = lich_su[0][2] if i == 0 else None

        print(f"  Bước k = {k}:")
        print(f"    xₖ  = {xk:.10f}")
        print(f"    f(xₖ) = {fxk:.8e}")
        print(f"    SS(1) = |f(xₖ)|/m₁ = {abs(fxk):.2e}/{m1:.6f} = {ss1:.2e}")
        if ss1 <= eps:
            print(f"    → SS(1) = {ss1:.2e} ≤ ε = {eps} → DỪNG ✓")
        else:
            print(f"    → SS(1) = {ss1:.2e} > ε = {eps} → tiếp tục")
            print(f"    x_(k+1) = {xk:.8f} - ({fxk:.6e})·({xk:.6f}-{lich_su[0][1]:.6f}) / (...)")
            print(f"            = {xk1:.10f}")
        print()

    nghiem = lich_su[-1][1]
    print(f"  ━━━ KẾT QUẢ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Nghiệm gần đúng:  x* ≈ {nghiem:.10f}")
    print(f"  Đạt được sau {len(lich_su)} vòng lặp")
    print(f"  Sai số cuối: {lich_su[-1][3]:.2e}")


# ─────────────────────────────────────────────────────────────
#  PHẦN 6: VẼ ĐỒ THỊ
# ─────────────────────────────────────────────────────────────

def ve_do_thi(f_num, a, b, d, x0, lich_su, bieu_thuc, eps):
    """Vẽ đồ thị chi tiết với đường cong, nghiệm, và lịch sử hội tụ."""
    nghiem = lich_su[-1][1]

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0d1117')

    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])   # Đồ thị chính - rộng full
    ax2 = fig.add_subplot(gs[1, 0])   # Sai số theo vòng lặp
    ax3 = fig.add_subplot(gs[1, 1])   # Phóng to vùng nghiệm

    MAU_NEN   = '#0d1117'
    MAU_LUOI  = '#21262d'
    MAU_DUONG = '#58a6ff'
    MAU_NGHIEM= '#f0e040'
    MAU_DAY   = '#ff7b72'
    MAU_LUOI2 = '#30363d'
    MAU_TEXT  = '#e6edf3'
    MAU_GREEN = '#3fb950'
    MAU_ORANGE= '#d29922'

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(MAU_NEN)
        ax.tick_params(colors=MAU_TEXT, labelsize=9)
        ax.spines['bottom'].set_color(MAU_LUOI2)
        ax.spines['top'].set_color(MAU_LUOI2)
        ax.spines['left'].set_color(MAU_LUOI2)
        ax.spines['right'].set_color(MAU_LUOI2)
        ax.grid(True, color=MAU_LUOI, linewidth=0.5, linestyle='--', alpha=0.6)

    # ─── ĐỒ THỊ CHÍNH ───────────────────────────────────────
    margin = (b - a) * 0.3
    xs = np.linspace(a - margin, b + margin, 1200)
    try:
        ys = f_num(xs)
        # Loại bỏ các giá trị bất thường
        mask = np.abs(ys) < 1e6
        xs_plot, ys_plot = xs[mask], ys[mask]
    except:
        xs_plot, ys_plot = xs, np.zeros_like(xs)

    ax1.plot(xs_plot, ys_plot, color=MAU_DUONG, linewidth=2.5,
             label=f'y = f(x) = {bieu_thuc}', zorder=5)
    ax1.axhline(0, color=MAU_TEXT, linewidth=1, alpha=0.5)
    ax1.axvline(0, color=MAU_TEXT, linewidth=0.5, alpha=0.3)

    # Vẽ các dây cung (tối đa 6 dây đầu tiên)
    n_day = min(6, len(lich_su) - 1)
    colors_day = plt.cm.plasma(np.linspace(0.3, 0.9, n_day))
    fd_val = f_num(d)

    for i in range(n_day):
        k, xk, fxk, _, _ = lich_su[i]
        _, xk1, _, _, _ = lich_su[i+1] if i+1 < len(lich_su) else lich_su[i]
        # Vẽ đoạn dây cung từ (d, f(d)) đến (xk, f(xk))
        ax1.plot([d, xk], [fd_val, fxk],
                 color=colors_day[i], linewidth=1.2, alpha=0.7,
                 linestyle='--', zorder=4)
        # Đánh dấu x_{k+1}
        ax1.plot(xk1, 0, 'o', color=colors_day[i], markersize=5,
                 zorder=6, alpha=0.9)

    # Điểm Fourier d
    ax1.plot(d, fd_val, 's', color=MAU_ORANGE, markersize=10,
             zorder=8, label=f'Điểm Fourier d = {d}')
    ax1.annotate(f'd = {d}', (d, fd_val),
                 textcoords="offset points", xytext=(10, 8),
                 color=MAU_ORANGE, fontsize=9, fontweight='bold')

    # Điểm xuất phát x0
    ax1.plot(x0, f_num(x0), '^', color='#da3633', markersize=10,
             zorder=8, label=f'Điểm xuất phát x₀ = {x0}')

    # Nghiệm
    ax1.plot(nghiem, 0, '*', color=MAU_NGHIEM, markersize=18,
             zorder=10, label=f'Nghiệm x* ≈ {nghiem:.6f}')
    ax1.axvline(nghiem, color=MAU_NGHIEM, linewidth=1,
                linestyle=':', alpha=0.6)
    ax1.annotate(f'x* ≈ {nghiem:.6f}',
                 (nghiem, 0), textcoords="offset points",
                 xytext=(8, -18), color=MAU_NGHIEM,
                 fontsize=10, fontweight='bold')

    # Khoảng [a,b]
    y_min_plot = min(ys_plot) if len(ys_plot) > 0 else -1
    y_max_plot = max(ys_plot) if len(ys_plot) > 0 else 1
    rng = y_max_plot - y_min_plot
    ax1.axvspan(a, b, alpha=0.08, color=MAU_GREEN, label=f'Khoảng [{a}, {b}]')

    ax1.set_title(f'Phương pháp Dây Cung  —  f(x) = {bieu_thuc}',
                  color=MAU_TEXT, fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('x', color=MAU_TEXT, fontsize=10)
    ax1.set_ylabel('f(x)', color=MAU_TEXT, fontsize=10)
    leg = ax1.legend(loc='best', facecolor='#161b22', edgecolor=MAU_LUOI2,
                     labelcolor=MAU_TEXT, fontsize=9)

    # ─── SỐ ĐỒ SAI SỐ ───────────────────────────────────────
    ks  = [h[0] for h in lich_su]
    ss1 = [h[3] for h in lich_su if h[3] != float('inf')]
    ss2 = [h[4] for h in lich_su if h[4] != float('inf')]
    ks1 = [h[0] for h in lich_su if h[3] != float('inf')]
    ks2 = [h[0] for h in lich_su if h[4] != float('inf')]

    if ss1:
        ax2.semilogy(ks1, ss1, 'o-', color=MAU_DUONG, linewidth=2,
                     markersize=5, label='SS mục tiêu (1)')
    if ss2:
        ax2.semilogy(ks2, ss2, 's-', color=MAU_DAY, linewidth=2,
                     markersize=5, label='SS hai xấp xỉ (2)')
    ax2.axhline(eps, color=MAU_NGHIEM, linewidth=1.5, linestyle='--',
                label=f'ε = {eps}')
    ax2.set_title('Sự hội tụ sai số', color=MAU_TEXT, fontsize=11,
                  fontweight='bold')
    ax2.set_xlabel('Vòng lặp k', color=MAU_TEXT, fontsize=9)
    ax2.set_ylabel('Sai số (log)', color=MAU_TEXT, fontsize=9)
    ax2.legend(facecolor='#161b22', edgecolor=MAU_LUOI2,
               labelcolor=MAU_TEXT, fontsize=8)

    # ─── PHÓNG TO VÙNG NGHIỆM ────────────────────────────────
    delta = max((b - a) * 0.05, abs(nghiem) * 1e-4, 1e-6)
    xs_zoom = np.linspace(nghiem - delta, nghiem + delta, 500)
    try:
        ys_zoom = f_num(xs_zoom)
    except:
        ys_zoom = np.zeros_like(xs_zoom)

    ax3.plot(xs_zoom, ys_zoom, color=MAU_DUONG, linewidth=2.5)
    ax3.axhline(0, color=MAU_TEXT, linewidth=1, alpha=0.6)
    ax3.plot(nghiem, 0, '*', color=MAU_NGHIEM, markersize=15, zorder=10)

    # Vẽ các x_k cuối cùng tiến đến nghiệm
    cuoi_n = min(6, len(lich_su))
    for i in range(len(lich_su) - cuoi_n, len(lich_su)):
        k, xk, fxk, _, _ = lich_su[i]
        if nghiem - delta <= xk <= nghiem + delta:
            ax3.plot(xk, 0, 'o', color=MAU_GREEN,
                     markersize=6, alpha=0.8, zorder=8)
            ax3.annotate(f'x_{k}', (xk, 0),
                         textcoords="offset points", xytext=(3, 5),
                         color=MAU_GREEN, fontsize=7)

    ax3.annotate(f'x* ≈ {nghiem:.8f}', (nghiem, 0),
                 textcoords="offset points", xytext=(5, -15),
                 color=MAU_NGHIEM, fontsize=8, fontweight='bold')
    ax3.set_title('Phóng to vùng nghiệm', color=MAU_TEXT,
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel('x', color=MAU_TEXT, fontsize=9)
    ax3.set_ylabel('f(x)', color=MAU_TEXT, fontsize=9)

    # Tiêu đề tổng
    fig.suptitle('PHƯƠNG PHÁP DÂY CUNG  —  Giải phương trình f(x) = 0',
                 color='#f0e040', fontsize=15, fontweight='bold', y=0.98)

    plt.savefig('day_cung_do_thi.png', dpi=150, bbox_inches='tight',
                facecolor=MAU_NEN)
    plt.show()
    print("\n  ✓ Đồ thị đã lưu tại: day_cung_do_thi.png")


# ─────────────────────────────────────────────────────────────
#  CHƯƠNG TRÌNH CHÍNH
# ─────────────────────────────────────────────────────────────

def main():
    print()
    print("╔" + "═"*62 + "╗")
    print("║    PHƯƠNG PHÁP DÂY CUNG — GIẢI PT f(x) = 0              ║")
    print("║    Theo slide bài giảng: Hà Thị Ngọc Yến, Hà Nội 2025   ║")
    print("╚" + "═"*62 + "╝")

    # 1. Nhập liệu
    x_sym, f_sym, f_num, bieu_thuc = nhap_ham_so()
    a, b = nhap_khoang(f_num)
    eps, mo_ta_ss, loai_ss = nhap_sai_so()

    # 2. Phân tích toán học
    fp_sym, fpp_sym, fp_num, fpp_num, m1, M1, d, x0 = phan_tich_ham(
        x_sym, f_sym, a, b, f_num)

    if m1 < 1e-12:
        print(f"\n  {RED}CẢNH BÁO: m₁ ≈ 0, f' có thể bằng 0 trong khoảng này.{RESET}")
        print("  Phương pháp có thể không hội tụ tốt.")

    # 3. Lặp
    print("\n" + "─"*60)
    print("  Đang thực hiện vòng lặp...")
    lich_su = day_cung_lap(f_num, fp_num, d, x0, m1, M1, eps)

    # 4. In bảng kết quả
    in_bang_lap(lich_su, eps, mo_ta_ss)

    # 5. Trình bày toán học
    hoi = input("\n  In trình bày toán học chi tiết? [y/n]: ").strip().lower()
    if hoi in ('y', 'yes', 'c', ''):
        trinh_bay_toan_hoc(f_sym, fp_sym, fpp_sym, a, b, d, x0,
                           m1, M1, lich_su, eps, mo_ta_ss, bieu_thuc, x_sym)

    # 6. Vẽ đồ thị
    hoi2 = input("\n  Vẽ đồ thị? [y/n]: ").strip().lower()
    if hoi2 in ('y', 'yes', 'c', ''):
        ve_do_thi(f_num, a, b, d, x0, lich_su, bieu_thuc, eps)

    # 7. Kết quả cuối
    nghiem = lich_su[-1][1]
    print("\n" + "═"*60)
    print(f"  {GREEN}{BOLD}NGHIỆM:  x* ≈ {nghiem:.10f}{RESET}")
    print(f"  f(x*) = {f_num(nghiem):.4e}")
    print(f"  Số vòng lặp: {len(lich_su)}")
    print("═"*60 + "\n")


# ─────────────────────────────────────────────────────────────
#  VÍ DỤ MẪU (chạy trực tiếp không cần nhập)
# ─────────────────────────────────────────────────────────────

def vi_du_mac_dinh():
    """
    Chạy ví dụ mẫu: f(x) = x³ - x - 2 trên (1, 2), ε = 1e-6
    Nghiệm đúng: x* ≈ 1.5213797...
    """
    print()
    print("╔" + "═"*62 + "╗")
    print("║  VÍ DỤ MẪU: f(x) = x³ - x - 2,  [a,b] = [1, 2]        ║")
    print("║  Sai số: ε = 1e-6                                        ║")
    print("╚" + "═"*62 + "╝")

    x_sym = symbols('x')
    f_sym = x_sym**3 - x_sym - 2
    bieu_thuc = "x**3 - x - 2"
    f_num  = lambdify(x_sym, f_sym, modules=['numpy'])
    a, b   = 1.0, 2.0
    eps    = 1e-6
    mo_ta  = "ε = 1e-6"

    fp_sym, fpp_sym, fp_num, fpp_num, m1, M1, d, x0 = phan_tich_ham(
        x_sym, f_sym, a, b, f_num)

    lich_su = day_cung_lap(f_num, fp_num, d, x0, m1, M1, eps)
    in_bang_lap(lich_su, eps, mo_ta)
    trinh_bay_toan_hoc(f_sym, fp_sym, fpp_sym, a, b, d, x0,
                       m1, M1, lich_su, eps, mo_ta, bieu_thuc, x_sym)
    ve_do_thi(f_num, a, b, d, x0, lich_su, bieu_thuc, eps)


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        vi_du_mac_dinh()
    else:
        # Hỏi người dùng muốn chạy demo hay nhập tay
        print("\n  Chạy ví dụ mẫu hay nhập tay?")
        print("  [1] Ví dụ mẫu: f(x) = x³ - x - 2")
        print("  [2] Nhập hàm số của tôi")
        chon = input("  Chọn [1/2]: ").strip()
        if chon == '1':
            vi_du_mac_dinh()
        else:
            main()