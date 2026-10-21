import pandas as pd
import os
import matplotlib.pyplot as plt


from pathlib import Path

BASE = Path(__file__).resolve().parent
FIGURAS = BASE / 'figuras'
FIGURAS.mkdir(exist_ok=True)

data= pd.read_csv(BASE / 'viajes_taxi.csv')

print('Analisis exploratorio descriptivo')

data.info()

print(data.dtypes)
print(data.isna().sum())
print(data.duplicated().sum())

print(data.describe())


plt.figure()
plt.hist(data['tarifa'], bins=50)
plt.title('Distribucion de tarifa')
plt.xlabel('USD')
plt.ylabel('# VIAJES')
plt.savefig(str(FIGURAS / 'hist_tarifa.png'))
plt.close()


plt.figure()
plt.boxplot(data['tarifa'])
plt.title('Boxplot de tarifa')  
plt.savefig(str(FIGURAS / 'boxplot_tarifa.png'))
plt.close()


# Extraccion de valores atipicos

q1= data['tarifa'].quantile(0.25)
q3= data['tarifa'].quantile(0.75)

iqr= q3-q1

limite_superior= q3 + 1.5*iqr
limite_inferior= q1 - 1.5*iqr

atipicos= data.query('tarifa < @limite_inferior or tarifa > @limite_superior')
sin_atipicos= data.query('tarifa >= @limite_inferior and tarifa <= @limite_superior')   

print(atipicos)

plt.figure()
plt.scatter(data['distancia_km'], data['tarifa'])
plt.title('Relacion entre distancia y tarifa')
plt.xlabel('Distancia (km)')
plt.ylabel('Tarifa (USD)')
plt.savefig(str(FIGURAS / 'scatter_distancia_tarifa.png'))
plt.close() 

print('Correlaciones numericas (datos originales):')
print(data.corr(numeric_only=True).round(3))


plt.figure()
plt.scatter(sin_atipicos['distancia_km'], sin_atipicos['tarifa'])
plt.title('Relacion entre distancia y tarifa sin atipicos')
plt.xlabel('Distancia (km)')        
plt.ylabel('Tarifa (USD)')
plt.savefig(str(FIGURAS / 'scatter_distancia_tarifa_sin_atipicos.png'))
plt.close()

print('Correlaciones numericas (sin atipicos IQR):')
print(sin_atipicos.corr(numeric_only=True).round(3))