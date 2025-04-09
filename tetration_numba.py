import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float64
from matplotlib.colors import LinearSegmentedColormap
import math

# 초기 파라미터
x0, y0 = 0.0, 0.0   # x, y
eps = 5.0   # 표시 범위
zoom_factor = 100.0  # 확대/축소 비율
n = 1920    # 해상도
nx, ny = n, int(n * (9 / 16))

# 계산 파라미터
max_iter = 500
escape_radius = 1e10

# 컬러맵 정의
cmap = LinearSegmentedColormap.from_list("custom_cmap", ["black", "white"])

@cuda.jit
def compute_tetration_gpu(xr, yr, output, max_iter, escape_radius):
    i, j = cuda.grid(2)
    nx, ny = xr.shape[0], yr.shape[0]
    
    if i < nx and j < ny:
        c_re = xr[i]
        c_im = yr[j]
        z_re = c_re
        z_im = c_im

        for _ in range(max_iter):
            r = (z_re ** 2 + z_im ** 2) ** 0.5
            if r == 0:
                break

            base_r = (c_re ** 2 + c_im ** 2) ** 0.5
            base_theta = math.atan2(c_im, c_re)

            logc_re = math.log(base_r)
            logc_im = base_theta

            mult_re = z_re * logc_re - z_im * logc_im
            mult_im = z_re * logc_im + z_im * logc_re

            exp_r = math.exp(mult_re)
            z_re = exp_r * math.cos(mult_im)
            z_im = exp_r * math.sin(mult_im)

            if z_re * z_re + z_im * z_im > escape_radius:
                output[i, j] = 1
                return
        output[i, j] = 0

def render():
    global x0, y0, eps

    eps_y = eps * (9 / 16)
    x = np.linspace(x0 - eps, x0 + eps, nx, dtype=np.float64)
    y = np.linspace(y0 - eps_y, y0 + eps_y, ny, dtype=np.float64)

    d_output = np.zeros((nx, ny), dtype=np.uint8)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(nx / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(ny / threadsperblock[1]))

    compute_tetration_gpu[(blockspergrid_x, blockspergrid_y), threadsperblock](
        x, y, d_output, max_iter, escape_radius
    )

    plt.clf()
    plt.imshow(
        d_output.T,
        extent=[x0 - eps, x0 + eps, y0 - eps_y, y0 + eps_y],
        origin="lower",
        cmap=cmap,
    )
    plt.title(f"x0={x0:.2e}, y0={y0:.2e}, eps={eps:.2e}")
    plt.axis("off")
    plt.draw()

def on_click(event):
    global x0, y0, eps
    if event.xdata is None or event.ydata is None:
        return

    if event.button == 1:  # 왼쪽 클릭 - 확대
        x0, y0 = event.xdata, event.ydata
        eps /= zoom_factor
        print(f"Zoom In -> x0: {x0:.2e}, y0: {y0:.2e}, eps: {eps:.2e}")
    elif event.button == 3:  # 오른쪽 클릭 - 축소
        x0, y0 = event.xdata, event.ydata
        eps *= zoom_factor
        print(f"Zoom Out -> x0: {x0:.2e}, y0: {y0:.2e}, eps: {eps:.2e}")
    render()

# 첫 렌더링 + 상호작용 설정
plt.figure(figsize=(12, 6))
render()
plt.connect('button_press_event', on_click)
plt.show()