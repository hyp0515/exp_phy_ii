import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# 固定隨機數種子
np.random.seed(0)

# 設定自由度範圍
dofs = range(2, 11)

# 建立subplots 3*3
fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
x_vals = np.linspace(0, 30, 1000)

# 遍歷每個自由度與子圖
for idx, dof in enumerate(dofs):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
 
    # 模擬卡方資料
    sum_chisqr = np.zeros(500)
    for _ in range(dof):
        data = np.random.normal(0, 1, 500)
        sum_chisqr += data**2

    # 畫直方圖
    ax.hist(sum_chisqr, bins=50, density=True, alpha=0.6, label='Simulated Data')

    # 畫理論曲線
    ax.plot(x_vals, chi2.pdf(x_vals, df=dof), color='red', label='Chi-square PDF')

    ax.set_xlim(0, 30)
    ax.set_title(f'df = {dof}')
    ax.legend()
    ax.set_xlabel('Chi-square value')
    ax.set_ylabel('Density')

plt.suptitle('Chi-square Distributions (df = 2 to 10)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 調整 layout 以容納 suptitle
plt.show()




