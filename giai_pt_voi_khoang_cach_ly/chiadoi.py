"""
====================================================
PHƯƠNG PHÁP CHIA ĐÔI (BISECTION METHOD)
Giải phương trình phi tuyến f(x) = 0
Tác giả: Dựa theo slide bài giảng - HUST 2025
====================================================
"""

import math
import sys

# ─────────────────────────────────────────────────
# Thư viện vẽ đồ thị (tuỳ chọn)
# ─────────────────────────────────────────────────
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    CAN_PLOT = True
except ImportError:
    CAN_PLOT = False
    print("[!] matplotlib / numpy chưa cài. Bỏ qua phần vẽ đồ thị.")
    print("    Cài bằng: pip install matplotlib numpy\n")


# ══════════════════════════════════════════════════
#  PHẦN 1: NHẬP LIỆU
# ══════════════════════════════════════════════════

def nhap_ham_so():
    """Nhập hàm số f(x) từ người dùng."""
    print("\n" + "═"*60)
    print("  PHƯƠNG PHÁP CHIA ĐÔI — BISECTION METHOD")
    print("═"*60)
    print("\n[1] NHẬP HÀM SỐ f(x)")
    print("-"*40)
    print("  Hỗ trợ: +  -  *  /  **  sqrt()  exp()  log()")
    print("          sin()  cos()  tan()  pi  e")
    print("  Ví dụ:  x**5 - 0.2*x + 15")
    print("          exp(x) - cos(2*x)")
    print("          x**3 - 2*x - 5")
    print("-"*40)

    while True:
        bieu_thuc = input("  f(x) = ").strip()
        if not bieu_thuc:
            print("  [!] Chưa nhập hàm số. Vui lòng thử lại.")
            continue
        # Kiểm tra hàm có hợp lệ không
        try:
            x = 1.0
            _kiem_tra = eval(
                bieu_thuc,
                {"__builtins__": {}},
                {"x": x, "sqrt": math.sqrt, "exp": math.exp,
                 "log": math.log, "sin": math.sin, "cos": math.cos,
                 "tan": math.tan, "pi": math.pi, "e": math.e,
                 "abs": abs, "log10": math.log10, "log2": math.log2}
            )
            break
        except Exception as err:
            print(f"  [!] Lỗi khi phân tích hàm: {err}")
            print("  [!] Vui lòng kiểm tra lại cú pháp.\n")

    return bieu_thuc


def tao_ham(bieu_thuc: str):
    """Tạo hàm Python từ chuỗi biểu thức."""
    NS = {"__builtins__": {}, "x": 0,
          "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
          "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "pi": math.pi, "e": math.e, "abs": abs,
          "log10": math.log10, "log2": math.log2}

    def f(x_val):
        ns = dict(NS)
        ns["x"] = x_val
        return eval(bieu_thuc, ns)

    return f


def nhap_khoang(f, bieu_thuc: str):
    """Nhập khoảng cách li nghiệm [a, b]."""
    print("\n[2] NHẬP KHOẢNG CÁCH LI NGHIỆM (a, b)")
    print("-"*40)
    print("  Yêu cầu: f(a) và f(b) phải TRÁI DẤU")
    print("-"*40)

    while True:
        try:
            a = float(input("  a = "))
            b = float(input("  b = "))
        except ValueError:
            print("  [!] Vui lòng nhập số thực.\n")
            continue

        if a >= b:
            print("  [!] Cần a < b. Vui lòng nhập lại.\n")
            continue

        fa, fb = f(a), f(b)
        print(f"\n  f({a}) = {fa:.6f}")
        print(f"  f({b}) = {fb:.6f}")

        if fa * fb > 0:
            print("  [!] f(a) và f(b) cùng dấu → không đảm bảo có nghiệm.")
            print("  [!] Vui lòng chọn khoảng khác.\n")
            continue
        if fa == 0:
            print(f"\n  ✓ x = {a} là nghiệm chính xác!")
            sys.exit(0)
        if fb == 0:
            print(f"\n  ✓ x = {b} là nghiệm chính xác!")
            sys.exit(0)
        break

    return a, b


def nhap_sai_so():
    """Nhập điều kiện dừng theo loại sai số."""
    print("\n[3] ĐIỀU KIỆN SAI SỐ")
    print("-"*40)
    print("  Chọn cách biểu diễn sai số ε:")
    print("  [1] Sai số tuyệt đối: |xₙ − x*| < ε")
    print("  [2] Sai số tương đối (%):  |xₙ − x*| / |xₙ| × 100 < ε%")
    print("  [3] Số chữ số đáng tin (significant digits)")
    print("  [4] Số chữ số sau dấu phẩy (decimal places)")
    print("-"*40)

    while True:
        lua_chon = input("  Lựa chọn (1/2/3/4): ").strip()
        if lua_chon not in ("1", "2", "3", "4"):
            print("  [!] Chỉ nhập 1, 2, 3 hoặc 4.")
            continue
        break

    eps = None
    loai = int(lua_chon)

    if loai == 1:
        while True:
            try:
                eps = float(input("  ε = "))
                if eps <= 0:
                    print("  [!] ε phải dương.")
                    continue
                break
            except ValueError:
                print("  [!] Nhập số thực.")
        mo_ta = f"Sai số tuyệt đối ε = {eps}"

    elif loai == 2:
        while True:
            try:
                pct = float(input("  ε (%) = "))
                if pct <= 0:
                    print("  [!] ε phải dương.")
                    continue
                eps = pct / 100.0
                break
            except ValueError:
                print("  [!] Nhập số thực.")
        mo_ta = f"Sai số tương đối ε = {pct}%"

    elif loai == 3:
        while True:
            try:
                k = int(input("  Số chữ số đáng tin k = "))
                if k <= 0:
                    print("  [!] k phải là số nguyên dương.")
                    continue
                eps = 0.5 * 10 ** (-(k - 1))
                break
            except ValueError:
                print("  [!] Nhập số nguyên.")
        mo_ta = f"{k} chữ số đáng tin  →  ε = {eps:.2e}"

    else:  # loai == 4
        while True:
            try:
                d = int(input("  Số chữ số sau dấu phẩy d = "))
                if d < 0:
                    print("  [!] d phải không âm.")
                    continue
                eps = 0.5 * 10 ** (-d)
                break
            except ValueError:
                print("  [!] Nhập số nguyên.")
        mo_ta = f"{d} chữ số sau dấu phẩy  →  ε = {eps:.2e}"

    print(f"\n  → {mo_ta}")
    print(f"  → Sẽ dừng khi |bₙ − aₙ| < {eps:.2e}  (hậu nghiệm)")
    print(f"     hoặc (b−a)/2ⁿ < {eps:.2e}            (tiên nghiệm)")

    return eps, loai, mo_ta


# ══════════════════════════════════════════════════
#  PHẦN 2: THUẬT TOÁN CHIA ĐÔI
# ══════════════════════════════════════════════════

def so_buoc_tien_nghiem(a: float, b: float, eps: float) -> int:
    """Tính n tối thiểu để (b-a)/2^n < eps  (tiên nghiệm)."""
    if eps <= 0:
        return 0
    n = math.ceil(math.log2((b - a) / eps))
    return max(n, 1)


def chia_doi(f, a: float, b: float, eps: float):
    """
    Thuật toán chia đôi.
    Trả về: (nghiem, lich_su_lap)
    lich_su_lap = list of dict per iteration.
    """
    lich_su = []
    fa = f(a)
    n = 0

    while True:
        c = (a + b) / 2.0
        fc = f(c)
        do_dai = b - a
        sai_so_tien = do_dai / 2.0  # (b-a)/2^(n+1) ≈ độ rộng hiện tại / 2
        sai_so_hau = abs(b - a)     # |bₙ - aₙ|

        lich_su.append({
            "n": n + 1,
            "a": a, "b": b, "c": c,
            "f(a)": fa, "f(b)": f(b), "f(c)": fc,
            "do_dai": do_dai,
            "sai_so_tien": sai_so_tien,
            "sai_so_hau": sai_so_hau,
        })

        # Dừng chính xác
        if fc == 0:
            return c, lich_su

        # Cập nhật khoảng
        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc

        n += 1

        # Kiểm tra điều kiện dừng hậu nghiệm
        if abs(b - a) < eps:
            c = (a + b) / 2.0
            lich_su.append({
                "n": n + 1,
                "a": a, "b": b, "c": c,
                "f(a)": fa, "f(b)": f(b), "f(c)": f(c),
                "do_dai": abs(b - a),
                "sai_so_tien": abs(b - a) / 2.0,
                "sai_so_hau": abs(b - a),
            })
            return c, lich_su


# ══════════════════════════════════════════════════
#  PHẦN 3: IN KẾT QUẢ
# ══════════════════════════════════════════════════

W_N  = 5
W_AB = 14
W_C  = 16
W_FC = 14
W_TN = 14
W_HN = 14

def in_dau_bang():
    tong = W_N + 2*W_AB + W_C + W_FC + W_TN + W_HN + 13
    print("\n" + "─"*tong)
    print(
        f"{'n':>{W_N}} │"
        f"{'a':>{W_AB}} │"
        f"{'b':>{W_AB}} │"
        f"{'c=(a+b)/2':>{W_C}} │"
        f"{'f(c)':>{W_FC}} │"
        f"{'Tiên nghiệm':>{W_TN}} │"
        f"{'Hậu nghiệm':>{W_HN}}"
    )
    print("─"*tong)


def in_dong(row: dict):
    tien = row["sai_so_tien"]
    hau  = row["sai_so_hau"]
    print(
        f"{row['n']:>{W_N}} │"
        f"{row['a']:>{W_AB}.8f} │"
        f"{row['b']:>{W_AB}.8f} │"
        f"{row['c']:>{W_C}.10f} │"
        f"{row['f(c)']:>{W_FC}.6e} │"
        f"{tien:>{W_TN}.6e} │"
        f"{hau:>{W_HN}.6e}"
    )


def in_ket_qua(bieu_thuc, a0, b0, eps, mo_ta_sai_so, nghiem, lich_su, n_tien):
    tong_buoc = len(lich_su)
    nhieu = tong_buoc > 15

    # ── Tiêu đề ──
    print("\n" + "═"*70)
    print("  TRÌNH BÀY TOÁN HỌC — PHƯƠNG PHÁP CHIA ĐÔI")
    print("═"*70)

    print(f"""
  Bài toán: Giải phương trình  f(x) = {bieu_thuc} = 0

  Điều kiện:
    • Khoảng cách li nghiệm: [{a0}, {b0}]
    • f({a0}) × f({b0}) < 0  ✓
    • Sai số yêu cầu: {mo_ta_sai_so}

  ┌─ CÔNG THỨC SAI SỐ ────────────────────────────────────┐
  │                                                         │
  │  TIÊN NGHIỆM (a priori):                               │
  │                    b − a                               │
  │    |xₙ − x*| ≤  ────────  →  0   khi n → ∞           │
  │                     2ⁿ                                 │
  │                                                         │
  │  → Số bước cần thiết:  n ≥ log₂((b−a)/ε)              │
  │    n_min = ⌈log₂(({b0}−{a0})/{eps:.2e})⌉ = {n_tien}   │
  │                                                         │
  │  HẬU NGHIỆM (a posteriori):                            │
  │                                                         │
  │    |xₙ − x*| ≤ |bₙ − aₙ|                              │
  │                                                         │
  │  → Dừng khi |bₙ − aₙ| < ε = {eps:.2e}                │
  └─────────────────────────────────────────────────────────┘

  Thuật toán (mỗi bước):
    Bước 1: c := (a + b) / 2
    Bước 2: z := f(c)
    Bước 3: Nếu z = 0 → nghiệm x = c, dừng
    Bước 4: Nếu z·f(a) < 0 → b := c
            Ngược lại        → a := c
    Bước 5: Nếu |b − a| < ε → dừng, nghiệm ≈ c
    Bước 6: Quay lại Bước 1
""")

    # ── Bảng lặp ──
    print("─"*70)
    print("  BẢNG CÁC VÒNG LẶP")
    print("─"*70)

    if not nhieu:
        # In toàn bộ
        in_dau_bang()
        for row in lich_su:
            in_dong(row)
        print("─"*(W_N + 2*W_AB + W_C + W_FC + W_TN + W_HN + 13))
    else:
        # In 5 tiên nghiệm + dấu ... + 5 hậu nghiệm
        print(f"\n  [Tổng cộng {tong_buoc} vòng lặp — hiển thị 5 đầu + 5 cuối]\n")

        print("  ── 5 bước đầu (minh hoạ TIÊN NGHIỆM) ──")
        in_dau_bang()
        for row in lich_su[:5]:
            in_dong(row)
        tong = W_N + 2*W_AB + W_C + W_FC + W_TN + W_HN + 13
        print("─"*tong)
        print(f"{'   ...':>{tong//2}}  (bỏ {tong_buoc - 10} bước giữa)")
        print("─"*tong)

        print("\n  ── 5 bước cuối (minh hoạ HẬU NGHIỆM) ──")
        in_dau_bang()
        for row in lich_su[-5:]:
            in_dong(row)
        print("─"*tong)

    # ── Kết luận ──
    print(f"""
  KẾT LUẬN:
  ─────────────────────────────────────────────────
  • Nghiệm gần đúng:    x* ≈ {nghiem:.10f}
  • f(x*)            ≈ {lich_su[-1]['f(c)']:.4e}
  • Số vòng lặp đã dùng: {tong_buoc}
  • Số vòng ước lượng tiên nghiệm: {n_tien}
  • Sai số hậu nghiệm cuối: {lich_su[-1]['sai_so_hau']:.4e}  < ε = {eps:.2e}  ✓
  ─────────────────────────────────────────────────
""")


# ══════════════════════════════════════════════════
#  PHẦN 4: VẼ ĐỒ THỊ
# ══════════════════════════════════════════════════

def ve_do_thi(f, bieu_thuc: str, a0: float, b0: float,
              nghiem: float, lich_su: list):
    if not CAN_PLOT:
        return

    # Mở rộng khoảng vẽ
    span = b0 - a0
    x_min = a0 - 0.3 * span
    x_max = b0 + 0.3 * span

    xs = np.linspace(x_min, x_max, 1000)
    ys = []
    for xv in xs:
        try:
            ys.append(f(xv))
        except Exception:
            ys.append(float("nan"))
    ys = np.array(ys)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Phương pháp chia đôi  —  f(x) = {bieu_thuc} = 0",
                 fontsize=13, fontweight="bold")

    # ── Đồ thị hàm số ──
    ax1 = axes[0]
    ax1.plot(xs, ys, "steelblue", linewidth=2, label=f"f(x) = {bieu_thuc}")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axvline(0, color="black", linewidth=0.8)

    # Đánh dấu nghiệm
    try:
        yn = f(nghiem)
    except Exception:
        yn = 0
    ax1.plot(nghiem, 0, "ro", markersize=9, zorder=5, label=f"Nghiệm ≈ {nghiem:.6f}")
    ax1.plot([nghiem, nghiem], [0, yn], "r--", linewidth=1)

    # Khoảng ban đầu
    ax1.axvspan(a0, b0, alpha=0.12, color="orange", label=f"Khoảng [{a0}, {b0}]")
    ax1.scatter([a0, b0], [f(a0), f(b0)], color="orange", s=60, zorder=4)

    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title("Đồ thị hàm số và nghiệm")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ── Biểu đồ hội tụ ──
    ax2 = axes[1]
    buocs = [r["n"] for r in lich_su]
    tien  = [r["sai_so_tien"] for r in lich_su]
    hau   = [r["sai_so_hau"]  for r in lich_su]
    eps_val = lich_su[-1]["sai_so_hau"]  # ε cuối

    ax2.semilogy(buocs, tien, "b.-", linewidth=1.5, markersize=5,
                 label="Tiên nghiệm $(b-a)/2^n$")
    ax2.semilogy(buocs, hau,  "r.-", linewidth=1.5, markersize=5,
                 label="Hậu nghiệm $|b_n - a_n|$")
    ax2.axhline(eps_val, color="green", linestyle="--",
                linewidth=1.2, label=f"ε = {eps_val:.2e}")

    ax2.set_xlabel("Số bước lặp n")
    ax2.set_ylabel("Sai số (log scale)")
    ax2.set_title("Sự hội tụ của sai số")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("bisection_result.png", dpi=130, bbox_inches="tight")
    print("  [✓] Đồ thị đã lưu vào: bisection_result.png")
    plt.show()


# ══════════════════════════════════════════════════
#  CHƯƠNG TRÌNH CHÍNH
# ══════════════════════════════════════════════════

def main():
    # 1. Nhập hàm số
    bieu_thuc = nhap_ham_so()
    f = tao_ham(bieu_thuc)

    # 2. Nhập khoảng
    a0, b0 = nhap_khoang(f, bieu_thuc)

    # 3. Nhập sai số
    eps, loai_sai_so, mo_ta = nhap_sai_so()

    # 4. Ước lượng tiên nghiệm
    n_tien = so_buoc_tien_nghiem(a0, b0, eps)
    print(f"\n  → Ước lượng tiên nghiệm: cần ít nhất {n_tien} bước lặp")

    # 5. Chạy thuật toán
    print("\n" + "─"*40)
    print("  Đang chạy thuật toán chia đôi...")
    nghiem, lich_su = chia_doi(f, a0, b0, eps)

    # 6. In kết quả
    in_ket_qua(bieu_thuc, a0, b0, eps, mo_ta, nghiem, lich_su, n_tien)

    # 7. Vẽ đồ thị
    if CAN_PLOT:
        hoi = input("  Bạn có muốn xem đồ thị không? (y/n): ").strip().lower()
        if hoi in ("y", "yes", "co", "có", ""):
            ve_do_thi(f, bieu_thuc, a0, b0, nghiem, lich_su)
    else:
        print("  [!] Không thể vẽ đồ thị (thiếu matplotlib/numpy).")


if __name__ == "__main__":
    main()