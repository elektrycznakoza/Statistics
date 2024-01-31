import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Definiowanie zakresu wieku
lower_bound = 20
upper_bound = 80

# Nowe dane o wieku z rozkładu jednostajnego od 20 do 80 lat
data_100 = np.random.uniform(low=lower_bound, high=upper_bound, size=100)
data_10000 = np.random.uniform(low=lower_bound, high=upper_bound, size=10000)

# Histogram z Seaborn dla data_100
plt.figure(figsize=(10, 6))
sns.histplot(data_100, bins=20, kde=True, color='skyblue')
plt.title('Histogram dla danych "data_100"')
plt.xlabel('Wiek')
plt.ylabel('Liczba pasażerów')
plt.show()

# Średnia
mean_age_100 = np.round(np.mean(data_100))
print(f'Średni wiek dla "data_100": {mean_age_100}')

# Mediana
median_age_100 = np.median(data_100)
print(f'Mediana wieku dla "data_100": {median_age_100}')

# Moda
mode_age_100 = stats.mode(data_100)
print(f'Moda wieku dla "data_100": {mode_age_100.mode}')

# Kwartyle
q1_100 = np.percentile(data_100, 25)
q3_100 = np.percentile(data_100, 75)
iqr_100 = q3_100 - q1_100
print(f'Q1 dla "data_100": {q1_100}, Q3: {q3_100}, IQR: {iqr_100}')

# Wykres pudełkowy (BoxPlot)
plt.boxplot(data_100)
plt.title('Wykres pudełkowy dla danych "data_100"')
plt.ylabel('Wiek')
plt.show()

# Wariancja
variance_age_100 = np.var(data_100, ddof=1)
print(f'Wariancja wieku dla "data_100": {variance_age_100}')

# Odchylenie standardowe
std_dev_age_100 = np.std(data_100, ddof=1)
print(f'Odchylenie standardowe wieku dla "data_100": {std_dev_age_100}')

# Skalowanie zmiennej (StandardScaler)
scaler_standardized_100 = StandardScaler()
standardized_data_100 = scaler_standardized_100.fit_transform(data_100.reshape(-1, 1))

# Histogram po skalowaniu dla data_100
plt.figure(figsize=(10, 6))
sns.histplot(standardized_data_100, bins=20, color='skyblue', kde=True)
plt.title('Histogram po skalowaniu danych "data_100"')
plt.xlabel('Znormalizowany wiek')
plt.ylabel('Liczba pasażerów')
plt.show()

# Korelacja (zmienne 'data_100' i 'SibSp')
sibsp_values_100 = np.random.uniform(low=0, high=8, size=100)  # Symulacja danych 'SibSp'
correlation_coefficient_100, p_value_100 = stats.pearsonr(data_100, sibsp_values_100)
print(f'Współczynnik korelacji dla "data_100" i losowych "SibSp": {correlation_coefficient_100}, p-value: {p_value_100}')

# Wykres punktowy dla zmiennych 'data_100' i 'SibSp'
plt.scatter(x=data_100, y=sibsp_values_100, alpha=0.5)
plt.title('Wykres punktowy dla zmiennych "data_100" i losowych "SibSp"')
plt.xlabel('Wiek')
plt.ylabel('SibSp')
plt.show()

# Inne analizy dla "data_100"

# Analogicznie dla danych data_10000
# Histogram, metryki, skalowanie itd.

# Histogram z Seaborn dla data_10000
plt.figure(figsize=(10, 6))
sns.histplot(data_10000, bins=20, kde=True, color='orange')
plt.title('Histogram dla danych "data_10000"')
plt.xlabel('Wiek')
plt.ylabel('Liczba pasażerów')
plt.show()

# Średnia
mean_age_10000 = np.round(np.mean(data_10000))
print(f'Średni wiek dla "data_10000": {mean_age_10000}')

# Mediana
median_age_10000 = np.median(data_10000)
print(f'Mediana wieku dla "data_10000": {median_age_10000}')

# Moda
mode_age_10000 = stats.mode(data_10000)
print(f'Moda wieku dla "data_10000": {mode_age_10000.mode}')

# Kwartyle
q1_10000 = np.percentile(data_10000, 25)
q3_10000 = np.percentile(data_10000, 75)
iqr_10000 = q3_10000 - q1_10000
print(f'Q1 dla "data_10000": {q1_10000}, Q3: {q3_10000}, IQR: {iqr_10000}')

# Wykres pudełkowy (BoxPlot)
plt.boxplot(data_10000)
plt.title('Wykres pudełkowy dla danych "data_10000"')
plt.ylabel('Wiek')
plt.show()

# Wariancja
variance_age_10000 = np.var(data_10000, ddof=1)
print(f'Wariancja wieku dla "data_10000": {variance_age_10000}')

# Odchylenie standardowe
std_dev_age_10000 = np.std(data_10000, ddof=1)
print(f'Odchylenie standardowe wieku dla "data_10000": {std_dev_age_10000}')

# Skalowanie zmiennej (StandardScaler)
scaler_standardized_10000 = StandardScaler()
standardized_data_10000 = scaler_standardized_10000.fit_transform(data_10000.reshape(-1, 1))

# Histogram po skalowaniu dla data_10000
plt.figure(figsize=(10, 6))
sns.histplot(standardized_data_10000, bins=20, color='orange', kde=True)
plt.title('Histogram po skalowaniu danych "data_10000"')
plt.xlabel('Znormalizowany wiek')
plt.ylabel('Liczba pasażerów')
plt.show()

# Korelacja (zmienne 'data_10000' i 'SibSp')
sibsp_values_10000 = np.random.uniform(low=0, high=8, size=10000)  # Symulacja danych 'SibSp'
correlation_coefficient_10000, p_value_10000 = stats.pearsonr(data_10000, sibsp_values_10000)
print(f'Współczynnik korelacji dla "data_10000" i losowych "SibSp": {correlation_coefficient_10000}, p-value: {p_value_10000}')

# Wykres punktowy dla zmiennych 'data_10000' i 'SibSp'
plt.scatter(x=data_10000, y=sibsp_values_10000, alpha=0.5)
plt.title('Wykres punktowy dla zmiennych "data_10000" i losowych "SibSp"')
plt.xlabel('Wiek')
plt.ylabel('SibSp')
plt.show()

# Inne analizy dla "data_10000"
