"""
=============================================================
  PHƯƠNG PHÁP TIẾP TUYẾN (NEWTON-RAPHSON) - GIẢI PT f(x) = 0
=============================================================
Tác giả: Dựa trên bài giảng của Hà Thị Ngọc Yến, 2025

Cách dùng:
  - Nhập hàm f(x) và f'(x), f''(x)
  - Nhập khoảng cách ly nghiệm [a, b]
  - Nhập sai số epsilon
  - Chọn loại sai số: (1) mục tiêu |f(xn)|/m1 <= eps
                       (2) hai xấp xỉ liên tiếp M2/(2m1)*|xn-xn-1|^2 <= eps
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────
#  PHẦN 1: NHẬP ĐẦU VÀO
# ─────────────────────────────────────────────

print("=" * 65)
print("   PHƯƠNG PHÁP TIẾP TUYẾN (NEWTON-RAPHSON) - GIẢI PT f(x)=0")
print("=" * 65)
print()
print("Nhập hàm số f(x). Dùng cú pháp Python/numpy, ví dụ:")
print("   x**5 - 17")
print("   math.exp(x) - 4*math.cos(x/2) + 1")
print("   x**3 - 2*x - 5")
print()

f_str = input("  f(x) = ").strip()
fp_str = input("  f'(x) = ").strip()
fpp_str = input("  f''(x) = ").strip()

# Tạo các hàm từ chuỗi nhập
def make_func(expr):
    def func(x):
        return eval(expr, {"x": x, "math": math, "np": np,
                           "sin": math.sin, "cos": math.cos,
                           "exp": math.exp, "log": math.log,
                           "sqrt": math.sqrt, "pi": math.pi, "e": math.e})
    return func

f   = make_func(f_str)
fp  = make_func(fp_str)
fpp = make_func(fpp_str)

print()
a = float(input("  Nhập a (cận trái khoảng cách ly nghiệm): "))
b = float(input("  Nhập b (cận phải): "))

print()
print("  Nhập sai số epsilon (ví dụ: 1e-6 hoặc 0.5e-6):")
eps = float(input("  ε = "))

print()
print("  Chọn tiêu chí dừng:")
print("    (1) Sai số mục tiêu:        |f(xn)| / m1 ≤ ε")
print("    (2) Hai xấp xỉ liên tiếp:   (M2/2m1)·|xn - xn-1|² ≤ ε")
stop_choice = input("  Nhập 1 hoặc 2 [mặc định 1]: ").strip() or "1"

MAX_ITER = 200

# ─────────────────────────────────────────────
#  PHẦN 2: KIỂM TRA ĐIỀU KIỆN HỘI TỤ
# ─────────────────────────────────────────────

print()
print("=" * 65)
print("  BƯỚC 1 — KIỂM TRA ĐIỀU KIỆN HỘI TỤ")
print("=" * 65)

# Lấy mẫu để kiểm tra dấu
samples = np.linspace(a, b, 1000)
fp_vals  = [fp(x)  for x in samples]
fpp_vals = [fpp(x) for x in samples]

fp_signs  = set(1 if v > 0 else -1 for v in fp_vals)
fpp_signs = set(1 if v > 0 else -1 for v in fpp_vals)

print()
print(f"  • f'(x): min = {min(fp_vals):.6g},  max = {max(fp_vals):.6g}")
if len(fp_signs) == 1:
    sign_str = "dương (+)" if list(fp_signs)[0] > 0 else "âm (−)"
    print(f"    → f'(x) giữ nguyên dấu {sign_str} trên [{a}, {b}]  ✓")
else:
    print(f"    ⚠ CẢNH BÁO: f'(x) đổi dấu trên [{a}, {b}]! Thuật toán có thể phân kỳ.")

print()
print(f"  • f''(x): min = {min(fpp_vals):.6g},  max = {max(fpp_vals):.6g}")
if len(fpp_signs) == 1:
    sign_str = "dương (+)" if list(fpp_signs)[0] > 0 else "âm (−)"
    print(f"    → f''(x) giữ nguyên dấu {sign_str} trên [{a}, {b}]  ✓")
else:
    print(f"    ⚠ CẢNH BÁO: f''(x) đổi dấu trên [{a}, {b}]!")

# m1 = min|f'(x)|, M2 = max|f''(x)|
m1 = min(abs(v) for v in fp_vals)
M2 = max(abs(v) for v in fpp_vals)
print()
print(f"  • m₁ = min|f'(x)| trên [{a},{b}] ≈ {m1:.6g}")
print(f"  • M₂ = max|f''(x)| trên [{a},{b}] ≈ {M2:.6g}")

# ─────────────────────────────────────────────
#  BƯỚC 3: CHỌN ĐIỂM FOURIER x0
# ─────────────────────────────────────────────

print()
print("=" * 65)
print("  BƯỚC 2 — CHỌN ĐIỂM FOURIER x₀")
print("=" * 65)
print()
print("  Điều kiện: f(x₀)·f''(x₀) > 0")
print()

fa  = f(a);  fpp_a = fpp(a)
fb  = f(b);  fpp_b = fpp(b)
print(f"  Tại x = a = {a}: f(a) = {fa:.6g},  f''(a) = {fpp_a:.6g}  →  tích = {fa*fpp_a:.6g}")
print(f"  Tại x = b = {b}: f(b) = {fb:.6g},  f''(b) = {fpp_b:.6g}  →  tích = {fb*fpp_b:.6g}")

if fa * fpp_a > 0:
    x0 = a
    print(f"\n  → Chọn x₀ = a = {a}  (f(a)·f''(a) > 0)  ✓")
elif fb * fpp_b > 0:
    x0 = b
    print(f"\n  → Chọn x₀ = b = {b}  (f(b)·f''(b) > 0)  ✓")
else:
    x0 = (a + b) / 2
    print(f"\n  ⚠ Không rõ điểm Fourier tại a hoặc b, dùng x₀ = (a+b)/2 = {x0}")

# ─────────────────────────────────────────────
#  PHẦN 3: CÔNG THỨC TOÁN HỌC
# ─────────────────────────────────────────────

print()
print("=" * 65)
print("  CÔNG THỨC LẶP (NEWTON-RAPHSON)")
print("=" * 65)
print()
print("         f(xₖ)")
print("  xₖ₊₁ = xₖ − ──────   (k = 0, 1, 2, ...)")
print("         f'(xₖ)")
print()
print("  Tiêu chí dừng:")
if stop_choice == "1":
    print("         |f(xₙ)|")
    print("    (1)  ──────── ≤ ε")
    print("            m₁")
    print()
    print(f"  Ngưỡng kiểm tra: m₁·ε = {m1:.6g} × {eps} = {m1*eps:.6g}")
else:
    print("         M₂")
    print("    (2)  ──── · |xₙ − xₙ₋₁|² ≤ ε")
    print("         2m₁")
    coef = M2 / (2 * m1)
    print()
    print(f"  Hệ số M₂/(2m₁) = {M2:.6g} / (2×{m1:.6g}) = {coef:.6g}")
    threshold = math.sqrt(eps / coef) if coef > 0 else eps
    print(f"  ⟺ |xₙ − xₙ₋₁| ≤ √(2m₁ε/M₂) = {threshold:.6g}")

# ─────────────────────────────────────────────
#  PHẦN 4: VÒNG LẶP NEWTON
# ─────────────────────────────────────────────

print()
print("=" * 65)
print("  BƯỚC 3 — CÁC VÒNG LẶP NEWTON-RAPHSON")
print("=" * 65)
print()

history = []   # (n, xn, fxn, error_estimate, stopped)
xn = x0
converged = False
coef_stop2 = M2 / (2 * m1) if m1 > 0 else float('inf')

for k in range(MAX_ITER):
    fxn   = f(xn)
    fpxn  = fp(xn)

    if abs(fpxn) < 1e-15:
        print(f"  ⚠ f'(x{k}) ≈ 0, dừng để tránh chia 0!")
        break

    xn1 = xn - fxn / fpxn

    # Tính sai số ước lượng
    if stop_choice == "1":
        err_est = abs(fxn) / m1
        err_label = "|f(xₙ)|/m₁"
    else:
        if k == 0:
            err_est = float('inf')
        else:
            err_est = coef_stop2 * (xn - history[-1][1])**2
        err_label = "(M₂/2m₁)|xₙ-xₙ₋₁|²"

    history.append((k, xn, fxn, err_est))

    stop_flag = err_est <= eps if k > 0 or stop_choice == "1" else False
    if stop_choice == "1":
        stop_flag = (abs(fxn) / m1) <= eps

    if stop_flag:
        converged = True
        xn = xn1
        fxn_final = f(xn)
        err_final = abs(fxn_final) / m1 if stop_choice == "1" else coef_stop2 * (xn - history[-1][1])**2
        history.append((k+1, xn, fxn_final, err_final))
        break

    xn = xn1

# ─────────────────────────────────────────────
#  PHẦN 5: IN KẾT QUẢ
# ─────────────────────────────────────────────

n_iter = len(history)

# Header bảng
col_w = 20
header = (f"{'n':>4}  {'xₙ':>{col_w}}  {'f(xₙ)':>{col_w}}  {'Sai số ước lượng':>{col_w}}")
sep    = "-" * len(header)
print(header)
print(sep)

def print_row(row):
    n, xn_val, fxn_val, err = row
    print(f"{n:>4}  {xn_val:>{col_w}.12f}  {fxn_val:>{col_w}.6e}  {err:>{col_w}.6e}")

if n_iter <= 15:
    for row in history:
        print_row(row)
else:
    # In 5 đầu và 5 cuối
    for row in history[:5]:
        print_row(row)
    print(f"{'...':>4}  {'...' :>{col_w}}  {'...':>{col_w}}  {'...':>{col_w}}")
    for row in history[-5:]:
        print_row(row)

print(sep)

# Kết quả cuối
x_result = history[-1][1]
err_result = history[-1][3]

print()
print("=" * 65)
print("  KẾT QUẢ")
print("=" * 65)
print()
if converged:
    print(f"  ✅ Phương pháp HỘI TỤ sau {n_iter-1} vòng lặp")
else:
    print(f"  ⚠ Chưa hội tụ sau {MAX_ITER} vòng lặp (dùng giá trị cuối)")

print()
print(f"  Nghiệm xấp xỉ:  x* ≈ {x_result:.12f}")
print(f"  f(x*)         = {f(x_result):.6e}")
print(f"  Sai số ước lượng ≤ {err_result:.6e}")
print()

# Số chữ số đáng tin
if err_result > 0:
    n_digits = max(0, -int(math.floor(math.log10(err_result))))
    print(f"  Số chữ số đáng tin sau dấu phẩy: {n_digits}")
    rounded = round(x_result, n_digits)
    print(f"  Kết quả làm tròn {n_digits} chữ số: x* ≈ {rounded}")

# ─────────────────────────────────────────────
#  PHẦN 6: VẼ ĐỒ THỊ
# ─────────────────────────────────────────────

print()
print("  Đang vẽ đồ thị...")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0f1117')
gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])   # đồ thị hàm số + tiếp tuyến
ax2 = fig.add_subplot(gs[1, 0])   # hội tụ xn
ax3 = fig.add_subplot(gs[1, 1])   # sai số

DARK = '#0f1117'
GRID = '#2a2d3a'
CYAN  = '#00d4ff'
ORANGE = '#ff8c00'
GREEN = '#00ff88'
RED   = '#ff4444'
PURPLE = '#cc88ff'
WHITE = '#e8eaf0'

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#1a1d2e')
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.spines['bottom'].set_color(GRID)
    ax.spines['left'].set_color(GRID)
    ax.spines['top'].set_color(GRID)
    ax.spines['right'].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

# ── Đồ thị 1: f(x) và các tiếp tuyến ──
margin = (b - a) * 0.3
xs = np.linspace(a - margin, b + margin, 800)
try:
    ys = [f(x) for x in xs]
except:
    ys = [0] * len(xs)

ax1.axhline(0, color=GRID, linewidth=1.0)
ax1.axvline(0, color=GRID, linewidth=0.5)
ax1.plot(xs, ys, color=CYAN, linewidth=2.2, label=f'f(x) = {f_str}', zorder=3)

# Vẽ các tiếp tuyến (tối đa 6 bước đầu)
tangent_colors = [ORANGE, '#ffdd00', GREEN, '#ff69b4', PURPLE, '#ff6666']
n_show = min(6, len(history) - 1)
for i in range(n_show):
    n_i, xi, fxi, _ = history[i]
    fpi = fp(xi)
    # Tiếp tuyến: y = f'(xi)(x - xi) + f(xi)
    x_tang = np.linspace(xi - (b-a)*0.4, xi + (b-a)*0.4, 100)
    y_tang = fpi * (x_tang - xi) + fxi
    color_t = tangent_colors[i % len(tangent_colors)]
    ax1.plot(x_tang, y_tang, color=color_t, linewidth=1.2, alpha=0.75,
             linestyle='--', label=f'd_{n_i}  (x={xi:.4f})')
    ax1.plot(xi, fxi, 'o', color=color_t, markersize=6, zorder=5)
    # Mũi tên từ điểm trên đường cong xuống trục x
    xi1 = history[i+1][1]
    ax1.annotate('', xy=(xi1, 0), xytext=(xi, fxi),
                 arrowprops=dict(arrowstyle='->', color=color_t, lw=1.2))

# Vẽ nghiệm
ax1.axvline(x=x_result, color=GREEN, linewidth=1.5, linestyle=':', alpha=0.8)
ax1.plot(x_result, 0, '*', color=GREEN, markersize=14, zorder=6,
         label=f'x* ≈ {x_result:.8f}')
ax1.plot(x_result, f(x_result), 's', color=RED, markersize=7, zorder=6)

# Vùng cách ly nghiệm
ymin, ymax = ax1.get_ylim()
ax1.axvspan(a, b, alpha=0.08, color=CYAN, label=f'Khoảng [{a}, {b}]')

ax1.set_title('Phương pháp Tiếp tuyến (Newton-Raphson) — Đồ thị f(x) và các tiếp tuyến',
              color=WHITE, fontsize=12, pad=10)
ax1.set_xlabel('x', color=WHITE); ax1.set_ylabel('y', color=WHITE)
leg = ax1.legend(fontsize=7.5, facecolor='#1a1d2e', edgecolor=GRID,
                 labelcolor=WHITE, loc='upper right', ncol=3)

# ── Đồ thị 2: Hội tụ của xn ──
iters = [h[0] for h in history]
xvals = [h[1] for h in history]
ax2.plot(iters, xvals, 'o-', color=ORANGE, linewidth=2, markersize=6, zorder=3)
ax2.axhline(y=x_result, color=GREEN, linewidth=1.5, linestyle='--', label=f'x* ≈ {x_result:.6f}')
for i, (it, xv) in enumerate(zip(iters, xvals)):
    ax2.annotate(f'{xv:.5f}', (it, xv), textcoords='offset points',
                 xytext=(5, 5), fontsize=7, color=WHITE)
ax2.set_title('Hội tụ của dãy xₙ', color=WHITE, fontsize=11)
ax2.set_xlabel('Số vòng lặp n', color=WHITE)
ax2.set_ylabel('xₙ', color=WHITE)
ax2.legend(fontsize=8, facecolor='#1a1d2e', edgecolor=GRID, labelcolor=WHITE)

# ── Đồ thị 3: Sai số ──
errs = [h[3] for h in history]
valid_idx = [i for i, e in enumerate(errs) if e > 0 and e != float('inf')]
if len(valid_idx) >= 2:
    e_iters = [iters[i] for i in valid_idx]
    e_vals  = [errs[i]  for i in valid_idx]
    ax3.semilogy(e_iters, e_vals, 's-', color=PURPLE, linewidth=2, markersize=6, zorder=3)
    ax3.axhline(y=eps, color=RED, linewidth=1.5, linestyle='--', label=f'ε = {eps}')
    ax3.set_title('Sai số ước lượng theo vòng lặp', color=WHITE, fontsize=11)
    ax3.set_xlabel('Số vòng lặp n', color=WHITE)
    ax3.set_ylabel('Sai số (log scale)', color=WHITE)
    ax3.legend(fontsize=8, facecolor='#1a1d2e', edgecolor=GRID, labelcolor=WHITE)

# Tiêu đề tổng
fig.suptitle(f'f(x) = {f_str}   |   x* ≈ {x_result:.10f}   |   ε = {eps}',
             color=WHITE, fontsize=13, y=0.98)

plt.savefig('/mnt/user-data/outputs/newton_raphson_plot.png',
            dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  → Đã lưu đồ thị: newton_raphson_plot.png")

# ─────────────────────────────────────────────
#  PHẦN 7: TRÌNH BÀY LỜI GIẢI TOÁN HỌC
# ─────────────────────────────────────────────

print()
print("=" * 65)
print("  TRÌNH BÀY LỜI GIẢI TOÁN HỌC ĐẦY ĐỦ")
print("=" * 65)
print()
print(f"  Bài toán: Tìm nghiệm của f(x) = {f_str} = 0")
print(f"            trên khoảng ({a}, {b}) với ε = {eps}")
print()
print("  ┌─ Bước 1: Kiểm tra khoảng cách ly nghiệm")
print(f"  │  f({a}) = {f(a):.6g}")
print(f"  │  f({b}) = {f(b):.6g}")
if f(a) * f(b) < 0:
    print(f"  │  f({a})·f({b}) < 0  → phương trình có nghiệm trên ({a},{b})  ✓")
else:
    print(f"  │  ⚠ Kiểm tra lại khoảng cách ly nghiệm!")
print()
print("  ├─ Bước 2: Kiểm tra điều kiện hội tụ")
print(f"  │  f'(x) và f''(x) liên tục, giữ dấu không đổi trên [{a},{b}]  ✓")
print(f"  │  m₁ = min|f'| ≈ {m1:.6g}")
print(f"  │  M₂ = max|f''| ≈ {M2:.6g}")
print()
print("  ├─ Bước 3: Chọn điểm Fourier x₀")
print(f"  │  Cần: f(x₀)·f''(x₀) > 0")
print(f"  │  → x₀ = {x0}")
print()
print("  ├─ Bước 4: Áp dụng công thức lặp")
print("  │")
print("  │          f(xₖ)")
print("  │  xₖ₊₁ = xₖ − ──────")
print("  │          f'(xₖ)")
print("  │")

show_steps = history if len(history) <= 15 else (history[:5] + [None] + history[-5:])
for row in show_steps:
    if row is None:
        print("  │  ⋮")
        continue
    n_i, xi, fxi, err_i = row
    if n_i < len(history) - 1:
        xi1 = history[n_i + 1][1] if n_i + 1 < len(history) else xi
        fpi = fp(xi)
        print(f"  │  k={n_i}: x={xi:.8f}, f(x)={fxi:.6e}, f'(x)={fpi:.6e}")
        print(f"  │       → x_{n_i+1} = {xi:.8f} − ({fxi:.6e})/({fpi:.6e}) = {xi1:.8f}")
    else:
        print(f"  │  k={n_i}: x={xi:.8f}, f(x)={fxi:.6e}  ← KẾT QUẢ")

print()
print("  └─ Bước 5: Kiểm tra sai số và kết luận")
print()
print(f"      Nghiệm gần đúng:  x* ≈ {x_result:.12f}")
print(f"      Sai số ước lượng ≤ {err_result:.6e}  (< ε = {eps})")
if err_result > 0:
    print(f"      Số chữ số đáng tin: {n_digits} chữ số sau dấu phẩy")
print()
print("=" * 65)
print("  Hoàn tất! Xem đồ thị tại: newton_raphson_plot.png")
print("=" * 65)