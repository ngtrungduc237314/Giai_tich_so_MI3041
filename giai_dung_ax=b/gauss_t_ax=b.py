import numpy as np
from io import StringIO
# Đặt giới hạn hiển thị dòng không bị cắt
np.set_printoptions(linewidth=np.inf, precision=10, suppress=True)

def gauss_elimination(A, B):
    """
    Thực hiện khử Gauss để biến ma trận mở rộng thành dạng bậc thang.
    In pivot (vị trí và giá trị) và ma trận sau mỗi bước khử.
    
    Parameters:
    A (np.array): Ma trận hệ số
    B (np.array): Ma trận cột hằng số
    """
    # Ghép ma trận A và B thành ma trận mở rộng
    matrix = np.hstack((A, B))
    n, m = matrix.shape  # Số dòng và số cột
    current_row = 0
    epsilon = 1e-10  # Ngưỡng so sánh với 0
    
    print("Ma trận mở rộng ban đầu:")
    print(matrix)
    
    for col in range(m - 1):  # Duyệt qua các cột, trừ cột B
        # Tìm pivot
        pivot_row = None
        for i in range(current_row, n):
            if abs(matrix[i, col]) > epsilon:
                pivot_row = i
                break
        if pivot_row is None:
            continue  # Bỏ qua cột nếu không có pivot
        
        # In thông tin pivot
        pivot = matrix[pivot_row, col]
        print(f"\nPivot tại ({pivot_row+1}, {col+1}) có giá trị {pivot}")
        print("\n")
        
        # Hoán đổi dòng pivot lên current_row
        if pivot_row != current_row:
            matrix[[current_row, pivot_row]] = matrix[[pivot_row, current_row]]
            print(f"Hoán đổi dòng {current_row} và dòng {pivot_row}")
            print(matrix)
        
        # Khử các phần tử dưới pivot
        for i in range(current_row + 1, n):
            if abs(matrix[i, col]) > epsilon:
                factor = matrix[i, col] / matrix[current_row, col]
                matrix[i] -= factor * matrix[current_row]
                print(f"H{i+1} - ({factor:.2f}) x H{current_row+1}")
                print(matrix)
        
        current_row += 1
        if current_row >= n:
            break
    
    print("\nMa trận dạng bậc thang cuối cùng:")
    # Sử dụng array2string để chèn dấu phẩy giữa các phần tử
    formatted_matrix = np.array2string(matrix, separator=', ')
    print(f"\n{formatted_matrix}\n")
    
# Ví dụ sử dụng

input_A = """
3  -2  5  -7  4 
2   9 14 -30  0 
5  -4 18 -26 14 
-4  2  3  -5  2 
1   3  2  -6 -2 
"""

input_B = """
3   5
-5  10
7   11
-2  -4
-2   3
"""

try:
    # Đọc dữ liệu tự động bỏ qua khoảng trắng thừa
    A = np.loadtxt(StringIO(input_A.strip()))
    B = np.loadtxt(StringIO(input_B.strip()))
    gauss_elimination(A, B)
    print("\n")

except Exception as e:
    print(f"Lỗi nhập liệu: {e}")