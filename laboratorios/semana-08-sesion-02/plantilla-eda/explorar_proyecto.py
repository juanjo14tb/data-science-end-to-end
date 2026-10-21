from pathlib import Path
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = Path(__file__).resolve().parent / 'figuras'
FIGURAS.mkdir(exist_ok=True)
RUTA = Path(os.getenv('PROJECT_CSV', str(RAIZ / 'datos-proyecto' / 'players.csv.gz')))
if not RUTA.is_file():
    raise SystemExit(f'No se encontro el CSV del proyecto: {RUTA}')

datos = pd.read_csv(RUTA, low_memory=False)
OBJETIVO = 'market_value_in_eur'
print('EJERCICIO 3: EDA TRANSFERMARKT')
print('Archivo:', RUTA.name)
print('Filas y columnas:', datos.shape)
print('Duplicados por player_id:', datos.duplicated('player_id').sum())
print('Valores nulos relevantes:\n', datos[['market_value_in_eur', 'date_of_birth', 'height_in_cm', 'position']].isna().sum().to_string())
print('Valores de mercado (EUR):\n', datos[OBJETIVO].describe().round(2).to_string())

# Paso 1: definicion del objetivo y limpieza basica.
datos['date_of_birth'] = pd.to_datetime(datos['date_of_birth'], errors='coerce')
datos['last_season'] = pd.to_numeric(datos['last_season'], errors='coerce')
# Edad aproximada en la temporada que figura en el archivo, NO edad actual.
referencia = pd.to_datetime(datos['last_season'].astype('Int64').astype(str) + '-07-01', errors='coerce')
datos['edad_temporada'] = (referencia - datos['date_of_birth']).dt.days / 365.25
validos = datos.loc[datos[OBJETIVO].notna() & (datos[OBJETIVO] > 0)].copy()
print('\nRegistros con valor de mercado positivo:', len(validos))
print('Objetivo nulo:', datos[OBJETIVO].isna().sum(), '| objetivo <= 0:', (datos[OBJETIVO].notna() & (datos[OBJETIVO] <= 0)).sum())

# Paso 2: distribucion y atipicos estadisticos.
plt.figure(figsize=(9, 5))
plt.hist(validos[OBJETIVO] / 1e6, bins=60)
plt.yscale('log')
plt.title('Distribucion del valor de mercado de jugadores')
plt.xlabel('Valor de mercado (millones de euros)')
plt.ylabel('Numero de jugadores (escala log)')
plt.tight_layout()
plt.savefig(FIGURAS / 'distribucion_objetivo.png', dpi=140)
plt.close()
q1, q3 = validos[OBJETIVO].quantile([.25, .75])
iqr = q3 - q1
limite = q3 + 1.5 * iqr
atipicos = validos[validos[OBJETIVO] > limite]
print('Limite superior IQR (EUR):', round(limite, 2))
print('Atipicos IQR:', len(atipicos), f'({len(atipicos) / len(validos) * 100:.2f}%)')
print('Nota: jugadores de valor alto pueden ser reales, no errores.')

# Paso 3: relacion edad-valor (descartar edades inverosimiles).
relacion = validos[validos['edad_temporada'].between(15, 45)].copy()
plt.figure(figsize=(9, 5))
plt.scatter(relacion['edad_temporada'], relacion[OBJETIVO] / 1e6, alpha=.14, s=9)
plt.title('Edad aproximada en la ultima temporada vs valor de mercado')
plt.xlabel('Edad aproximada (anos)')
plt.ylabel('Valor de mercado (millones EUR)')
plt.tight_layout()
plt.savefig(FIGURAS / 'relacion_objetivo.png', dpi=140)
plt.close()
print('\nRegistros con edad aproximada valida (15 a 45):', len(relacion))
print('Correlacion de Pearson edad-valor:', round(relacion[['edad_temporada', OBJETIVO]].corr().iloc[0, 1], 3))

# Paso 4: segmentos y faltantes para decisiones de modelado.
resumen_posicion = validos.groupby('position')[OBJETIVO].agg(['count', 'median', 'mean']).sort_values('median', ascending=False)
print('\nValor de mercado segun posicion (EUR):\n', resumen_posicion.round(2).to_string())
