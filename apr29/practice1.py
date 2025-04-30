import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#固定隨機數種子，讓每次執行結果一致
np.random.seed(0)

#參數設定
a_true = 2.0
b_true = 1.0
sigma_normal = 3
sigma_large = 10
x = np.linspace(1, 10, 10)

#生成20組資料
data_all = []
for _ in range(20):
    noise = np.random.normal(0, sigma_normal, size=x.shape)
    # 將x=4,5,6,7對應的noise調大
    for idx, val in enumerate(x):
        if 4 <= val <= 7:
            noise[idx] = np.random.normal(0, sigma_large)
    y = a_true * x + b_true + noise
    data_all.append(y)

data_all = np.array(data_all)  #(20,10)array

#計算每個x的平均值與標準差
y_mean = np.mean(data_all, axis=0)
y_std = np.std(data_all, axis=0)

#定義擬合函數
def linear_func(x, a, b):
    return a * x + b

#Least Squares
popt_ls, pcov_ls = curve_fit(linear_func, x, y_mean)

#Chi-square Fitting
popt_chi, pcov_chi = curve_fit(linear_func, x, y_mean, sigma=y_std, absolute_sigma=True)

#預測值
y_fit_ls = linear_func(x, *popt_ls)
y_fit_chi = linear_func(x, *popt_chi)

#計算Chi-square value
chi2_ls = np.sum(((y_mean - y_fit_ls) / y_std)**2)
chi2_chi = np.sum(((y_mean - y_fit_chi) / y_std)**2)

#印出來
print("Least Squares Fit: a = {:.3f}, b = {:.3f}, Chi2 = {:.2f}".format(popt_ls[0], popt_ls[1], chi2_ls))
print("Chi-square Fit: a = {:.3f}, b = {:.3f}, Chi2 = {:.2f}".format(popt_chi[0], popt_chi[1], chi2_chi))

#繪圖
plt.figure(figsize=(10, 6))
plt.errorbar(x, y_mean, yerr=y_std, fmt='o', label='Data (mean ± std)')
plt.plot(x, y_fit_ls, label='Least Squares Fit', linestyle='--')
plt.plot(x, y_fit_chi, label='Chi-square Fit', linestyle='-')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Comparison of Least Squares and Chi-square Fitting')
# plt.grid(True)
# plt.show()
plt.savefig('comparison.pdf', transparent=True)