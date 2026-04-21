"""
====================================================
TÌM KHOẢNG CÁCH LI NGHIỆM
Cho phương trình phi tuyến f(x) = 0
Dựa theo slide bài giảng - HUST 2025
====================================================
"""

import math
import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
    CAN_PLOT = True
except ImportError:
    CAN_PLOT = False
    print("[!] matplotlib / numpy chưa cài. Bỏ qua phần vẽ đồ thị.")
    print("    Cài bằng: pip install matplotlib numpy\n")


# ══════════════════════════════════════════════════
#  HÀM TIỆN ÍCH
# ══════════════════════════════════════════════════

_NS_BASE = {
    "__builtins__": {},
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "log10": math.log10, "log2": math.log2,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "abs": abs, "pi": math.pi, "e": math.e,
}

def tao_ham(bieu_thuc: str):
    def f(x_val):
        ns = dict(_NS_BASE)
        ns["x"] = x_val
        return eval(bieu_thuc, ns)
    return f


def kiem_tra_bieu_thuc(bieu_thuc: str) -> bool:
    try:
        f = tao_ham(bieu_thuc)
        f(1.0)
        return True
    except Exception as err:
        print(f"  [!] Lỗi cú pháp: {err}")
        return False


# ══════════════════════════════════════════════════
#  PHẦN 1: NHẬP LIỆU
# ══════════════════════════════════════════════════

def nhap_ham_so():
    print("\n" + "═"*60)
    print("  TÌM KHOẢNG CÁCH LI NGHIỆM  —  f(x) = 0")
    print("═"*60)
    print("\n[1] NHẬP HÀM SỐ f(x)")
    print("-"*40)
    print("  Hỗ trợ: +  -  *  /  **")
    print("          sqrt()  exp()  log()  log10()  log2()")
    print("          sin()  cos()  tan()  asin()  acos()  atan()")
    print("          abs()  pi  e")
    print("  Ví dụ:  x**5 - 0.2*x + 15")
    print("          exp(x) - cos(2*x)")
    print("          x**3 - 2*x - 5")
    print("-"*40)

    while True:
        bieu_thuc = input("  f(x) = ").strip()
        if not bieu_thuc:
            print("  [!] Chưa nhập hàm. Vui lòng thử lại.")
            continue
        if kiem_tra_bieu_thuc(bieu_thuc):
            break
    return bieu_thuc


def nhap_mien_tim():
    print("\n[2] NHẬP MIỀN TÌM KIẾM [L, R]")
    print("-"*40)
    print("  Chương trình sẽ quét toàn bộ khoảng [L, R]")
    print("  để tìm tất cả khoảng cách li nghiệm.")
    print("-"*40)

    while True:
        try:
            L = float(input("  L (cận trái)  = "))
            R = float(input("  R (cận phải) = "))
            if L >= R:
                print("  [!] Cần L < R. Nhập lại.\n")
                continue
            break
        except ValueError:
            print("  [!] Vui lòng nhập số thực.\n")

    print("\n[3] NHẬP SỐ BƯỚC CHIA ĐỂ QUÉT")
    print("-"*40)
    print("  Càng nhiều bước → tìm được nhiều nghiệm gần nhau hơn")
    print(f"  (Mặc định: 1000 bước)")
    print("-"*40)

    while True:
        try:
            s = input("  Số bước N (Enter = 1000): ").strip()
            N = int(s) if s else 1000
            if N < 2:
                print("  [!] N phải >= 2.")
                continue
            break
        except ValueError:
            print("  [!] Vui lòng nhập số nguyên.\n")

    return L, R, N


# ══════════════════════════════════════════════════
#  PHẦN 2: THUẬT TOÁN QUÉT TÌM KHOẢNG
# ══════════════════════════════════════════════════

def tim_khoang_cach_li(f, L: float, R: float, N: int):
    """
    Quét [L, R] với N bước đều.
    Trả về danh sách các khoảng (a, b) thoả f(a)*f(b) < 0.
    Cũng phát hiện điểm nghiệm chính xác (f(x)=0).
    """
    h = (R - L) / N
    xs = [L + i * h for i in range(N + 1)]

    # Tính f tại tất cả các điểm, bỏ qua điểm lỗi (hàm không xác định)
    fxs = []
    for x in xs:
        try:
            fxs.append(f(x))
        except Exception:
            fxs.append(None)   # không xác định tại điểm này

    khoang_list   = []   # [(a, fa, b, fb), ...]
    nghiem_chinh_xac = []

    for i in range(N):
        xi, xi1 = xs[i], xs[i + 1]
        fi, fi1 = fxs[i], fxs[i + 1]

        # Bỏ qua nếu hàm không xác định tại một trong hai đầu
        if fi is None or fi1 is None:
            continue

        # Nghiệm chính xác
        if fi == 0:
            nghiem_chinh_xac.append(xi)
            continue
        if fi1 == 0:
            nghiem_chinh_xac.append(xi1)
            continue

        # Đổi dấu → có nghiệm trong (xi, xi1)
        if fi * fi1 < 0:
            khoang_list.append((xi, fi, xi1, fi1))

    # Loại trùng nghiệm chính xác
    nghiem_chinh_xac = sorted(set(round(x, 12) for x in nghiem_chinh_xac))

    return khoang_list, nghiem_chinh_xac


# ══════════════════════════════════════════════════
#  PHẦN 3: THU HẸP KHOẢNG (tuỳ chọn)
# ══════════════════════════════════════════════════

def thu_hep_khoang(f, a: float, b: float, so_buoc_thu_hep: int = 5):
    """
    Dùng chia đôi nhanh để thu hẹp khoảng cách li.
    Trả về (a', b') chặt hơn.
    """
    for _ in range(so_buoc_thu_hep):
        c = (a + b) / 2
        try:
            fc = f(c)
        except Exception:
            break
        fa = f(a)
        if fa * fc < 0:
            b = c
        elif fc * f(b) < 0:
            a = c
        else:
            break
    return a, b


# ══════════════════════════════════════════════════
#  PHẦN 4: IN KẾT QUẢ
# ══════════════════════════════════════════════════

def in_ket_qua(bieu_thuc, L, R, N, khoang_list, nghiem_cx, thu_hep=False, f=None):
    h = (R - L) / N

    print("\n" + "═"*70)
    print("  KẾT QUẢ TÌM KHOẢNG CÁCH LI NGHIỆM")
    print("═"*70)
    print(f"""
  Hàm số  : f(x) = {bieu_thuc}
  Miền quét: [{L}, {R}]
  Bước quét: h = (R-L)/N = ({R}-{L})/{N} = {h:.6g}
""")

    # ── Nghiệm chính xác (nếu có) ──
    if nghiem_cx:
        print("  ┌─ NGHIỆM CHÍNH XÁC (f(x) = 0) " + "─"*35 + "┐")
        for x in nghiem_cx:
            print(f"  │   x = {x}")
        print("  └" + "─"*67 + "┘\n")

    # ── Khoảng cách li ──
    if not khoang_list:
        print("  [!] Không tìm thấy khoảng cách li nghiệm nào trong miền quét.")
        print("      Gợi ý: mở rộng [L, R] hoặc tăng số bước N.\n")
        return

    tong = len(khoang_list)
    print(f"  Tìm được {tong} khoảng cách li nghiệm:\n")

    # Header bảng
    w = 70
    print("  " + "─"*w)
    print(f"  {'STT':>4}  {'a':>16}  {'f(a)':>14}  {'b':>16}  {'f(b)':>14}  {'b-a':>10}")
    print("  " + "─"*w)

    for idx, (a, fa, b, fb) in enumerate(khoang_list, 1):
        do_rong = b - a
        print(f"  {idx:>4}  {a:>16.8f}  {fa:>14.6e}  {b:>16.8f}  {fb:>14.6e}  {do_rong:>10.4e}")

    print("  " + "─"*w)

    # ── Thu hẹp khoảng (tuỳ chọn) ──
    if thu_hep and f is not None:
        print(f"\n  SAU KHI THU HẸP (10 bước chia đôi mỗi khoảng):\n")
        print("  " + "─"*w)
        col_a, col_fa, col_b, col_fb = "a'", "f(a')", "b'", "f(b')"
        print(f"  {'STT':>4}  {col_a:>16}  {col_fa:>14}  {col_b:>16}  {col_fb:>14}  {'b-a':>10}")
        print("  " + "─"*w)
        for idx, (a, fa, b, fb) in enumerate(khoang_list, 1):
            a2, b2 = thu_hep_khoang(f, a, b, so_buoc_thu_hep=10)
            try:
                fa2, fb2 = f(a2), f(b2)
            except Exception:
                fa2, fb2 = float("nan"), float("nan")
            do_rong2 = b2 - a2
            print(f"  {idx:>4}  {a2:>16.10f}  {fa2:>14.6e}  {b2:>16.10f}  {fb2:>14.6e}  {do_rong2:>10.4e}")
        print("  " + "─"*w)

    print(f"""
  GHI CHÚ:
  ─────────────────────────────────────────────────────────────────────
  • Mỗi khoảng trên đảm bảo f(a)·f(b) < 0 → tồn tại ít nhất 1 nghiệm.
  • Nếu f liên tục và đơn điệu trên khoảng → đúng 1 nghiệm.
  • Khoảng này có thể dùng trực tiếp làm đầu vào cho phương pháp
    chia đôi (bisection), dây cung, tiếp tuyến, v.v.
  • Để tìm nghiệm gần nhau hơn, tăng số bước N lên.
  ─────────────────────────────────────────────────────────────────────
""")


# ══════════════════════════════════════════════════
#  PHẦN 5: VẼ ĐỒ THỊ
# ══════════════════════════════════════════════════

def ve_do_thi(f, bieu_thuc, L, R, N, khoang_list, nghiem_cx):
    if not CAN_PLOT:
        return

    xs = np.linspace(L, R, max(N * 5, 2000))
    ys = []
    for xv in xs:
        try:
            ys.append(f(xv))
        except Exception:
            ys.append(float("nan"))
    ys = np.array(ys)

    # Giới hạn trục y để tránh spike
    y_finite = ys[np.isfinite(ys)]
    if len(y_finite) == 0:
        print("  [!] Không thể vẽ đồ thị.")
        return
    y_med  = np.median(np.abs(y_finite))
    y_clip = max(y_med * 10, 1.0)
    ys_clipped = np.clip(ys, -y_clip, y_clip)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Tìm khoảng cách li nghiệm  —  f(x) = {bieu_thuc} = 0",
                 fontsize=13, fontweight="bold")

    ax.plot(xs, ys_clipped, color="steelblue", linewidth=1.8,
            label=f"f(x) = {bieu_thuc}", zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.axvline(0, color="black", linewidth=0.3, zorder=2)

    # Tô màu từng khoảng cách li
    colors = plt.cm.Set2.colors
    for idx, (a, fa, b, fb) in enumerate(khoang_list):
        color = colors[idx % len(colors)]
        ax.axvspan(a, b, alpha=0.35, color=color,
                   label=f"[{a:.4f}, {b:.4f}]", zorder=1)
        # Đánh dấu 2 đầu
        ax.scatter([a, b], [fa, fb], color=color, s=50, zorder=5,
                   edgecolors="black", linewidths=0.5)

    # Đánh dấu nghiệm chính xác
    for xc in nghiem_cx:
        try:
            ax.plot(xc, 0, "r*", markersize=12, zorder=6,
                    label=f"Nghiệm cx: x={xc:.6f}")
        except Exception:
            pass

    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("f(x)", fontsize=11)
    ax.set_title("Đồ thị hàm số và các khoảng cách li nghiệm (vùng tô màu)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fname = "khoang_cach_li.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    print(f"  [✓] Đồ thị đã lưu vào: {fname}")
    plt.show()


# ══════════════════════════════════════════════════
#  CHƯƠNG TRÌNH CHÍNH
# ══════════════════════════════════════════════════

def main():
    # 1. Nhập hàm
    bieu_thuc = nhap_ham_so()
    f = tao_ham(bieu_thuc)

    # 2. Nhập miền tìm kiếm
    L, R, N = nhap_mien_tim()

    # 3. Quét tìm khoảng
    print("\n" + "─"*40)
    print("  Đang quét tìm khoảng cách li nghiệm...")
    khoang_list, nghiem_cx = tim_khoang_cach_li(f, L, R, N)

    # 4. Hỏi có thu hẹp không
    thu_hep = False
    if khoang_list:
        ans = input("\n  Thu hẹp khoảng bằng chia đôi nhanh? (y/n, Enter=y): ").strip().lower()
        thu_hep = ans not in ("n", "no", "không", "khong")

    # 5. In kết quả
    in_ket_qua(bieu_thuc, L, R, N, khoang_list, nghiem_cx,
               thu_hep=thu_hep, f=f)

    # 6. Vẽ đồ thị
    if CAN_PLOT and (khoang_list or nghiem_cx):
        ans2 = input("  Xem đồ thị? (y/n, Enter=y): ").strip().lower()
        if ans2 not in ("n", "no", "không", "khong"):
            ve_do_thi(f, bieu_thuc, L, R, N, khoang_list, nghiem_cx)
    else:
        print("  [!] Thiếu matplotlib/numpy — bỏ qua đồ thị.")


if __name__ == "__main__":
    main()