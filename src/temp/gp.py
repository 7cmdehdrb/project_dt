import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from skopt import gp_minimize
from skopt.benchmarks import branin
from skopt.space import Real

# 1. 2차원 탐색 공간 정의
space = [Real(-5.0, 10.0, name='x1'),
         Real(0.0, 15.0, name='x2')]

# 2. gp_minimize 함수 실행 (목표 함수로 branin 사용)
# 2차원은 탐색할 공간이 넓으므로 n_calls를 늘려주는 것이 좋습니다.
result = gp_minimize(
    func=branin,
    dimensions=space,
    n_calls=25,
    random_state=13
)

# --- 시각화 ---

# 3. 컨투어 플롯을 그리기 위한 그리드 데이터 생성
x1_grid = np.linspace(-5, 10, 100)
x2_grid = np.linspace(0, 15, 100)
x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
z_mesh = branin((x1_mesh, x2_mesh))

# 4. 맷플롯립(matplotlib)으로 3D 그래프 그리기
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 배경 서피스 플롯
ax.plot_surface(x1_mesh, x2_mesh, z_mesh, cmap='viridis_r', alpha=0.6)

# 5. 탐색 지점 및 최적해 오버레이
# result.x_iters는 [[x1_1, x2_1], [x1_2, x2_2], ...] 형태입니다.
x_iters_T = np.array(result.x_iters).T
z_iters = result.func_vals
ax.scatter(x_iters_T[0], x_iters_T[1], z_iters, c='red', s=50, label='Evaluated Points (x_iters)')

# 최종 최적해 (검은색 별표)
ax.scatter(result.x[0], result.x[1], result.fun, c='black', s=200, marker='*',
           label=f"Optimal Point: ({result.x[0]:.2f}, {result.x[1]:.2f})")


# 6. 그래프 스타일링
ax.set_title('Visualization of 3D Gaussian Process Optimization')
ax.set_xlabel('Parameter (x1)')
ax.set_ylabel('Parameter (x2)')
ax.set_zlabel('Function Value')
ax.legend()

# 7. 그래프 파일로 저장
plt.savefig('optimization_visualization_3d.png')

print("3차원 시각화 그래프를 'optimization_visualization_3d.png' 파일로 저장했습니다.")