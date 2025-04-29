#practice 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

#固定隨機數種子
np.random.seed(0)

#產生一組500個標準常態分布資料
data = np.random.normal(0, 1, 500)

#計算每個點的卡方值(自由度=1)
chisqr = (data - 0)**2 / 1  # μ=0, σ=1

#畫直方圖
plt.hist(chisqr, bins=50, density=True, alpha=0.6, label='Simulated Data')

#畫理論自由度1的卡方分布
x_vals = np.linspace(0, 10, 500)
plt.plot(x_vals, chi2.pdf(x_vals, df=1), label='Chi-square Distribution (df=1)', color='red')

plt.xlabel('Chi-square value')
plt.ylabel('Probability Density')
plt.title('Chi-square Distribution (df=1)')
plt.legend()
plt.grid(True)
plt.show()

#多組資料情況(自由度從2到10)
for dof in range(2, 11):
    sum_chisqr = np.zeros(500)
    for _ in range(dof):
        data = np.random.normal(0, 1, 500)
        sum_chisqr += (data - 0)**2 / 1

    #畫直方圖
    plt.hist(sum_chisqr, bins=50, density=True, alpha=0.6, label=f'Simulated Data (dof={dof})')

    #畫理論卡方分布
    plt.plot(x_vals, chi2.pdf(x_vals, df=dof), label=f'Chi-square Distribution (df={dof})', color='red')

    plt.xlabel('Chi-square value')
    plt.ylabel('Probability Density')
    plt.title(f'Chi-square Distribution (df={dof})')
    plt.legend()
    plt.grid(True)
    plt.show()
