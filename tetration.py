import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 기본 파라미터
x0, y0 = 0.0, 0.0   # x, y
eps = 5.0   # 표시 범위
zoom_factor = 100.0  # 확대/축소 비율
n = 300 # 해상도
aspect_ratio = 9 / 16
eps_y = eps * aspect_ratio
nx, ny = n, int(n * aspect_ratio)

max_iter = 500
escape_radius = 1e+10

# 테트레이션 발산 여부 계산
def compute_tetration_divergence(x0, y0, eps, nx, ny, max_iter, escape_radius):
    eps_y = eps * (ny / nx)
    x = np.linspace(x0 - eps, x0 + eps, nx)
    y = np.linspace(y0 - eps_y, y0 + eps_y, ny)
    c = x[:, np.newaxis] + 1j * y[np.newaxis, :]

    divergence_map = np.zeros_like(c, dtype=bool)

    for i in range(nx):
        for j in range(ny):
            c_val = c[i, j]
            z = c_val
            for _ in range(max_iter):
                z = c_val ** z
                if np.abs(z) > escape_radius:
                    divergence_map[i, j] = True
                    break
    return divergence_map, x, y

# 초기 그래프 렌더링
fig, ax = plt.subplots()
cmap = LinearSegmentedColormap.from_list("custom_cmap", ["black", "white"])

def render():
    global img, x0, y0, eps
    divergence_map, x_vals, y_vals = compute_tetration_divergence(x0, y0, eps, nx, ny, max_iter, escape_radius)
    ax.clear()
    extent = [x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]]
    img = ax.imshow(divergence_map.T, extent=extent, origin='lower', cmap=cmap)
    ax.set_title(f"x0={x0:.2e}, y0={y0:.2e}, eps={eps:.2e}")
    ax.axis('off')
    fig.canvas.draw()

# 마우스 클릭 이벤트 핸들러
def on_click(event):
    global x0, y0, eps
    if event.inaxes is None:
        return

    x0, y0 = event.xdata, event.ydata

    if event.button == 1:  # 좌클릭 -> 확대
        eps /= zoom_factor
    elif event.button == 3:  # 우클릭 -> 축소
        eps *= zoom_factor

    render()

# 이벤트 연결
fig.canvas.mpl_connect('button_press_event', on_click)

# 초기 렌더링
render()
plt.show()