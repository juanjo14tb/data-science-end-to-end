from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
FIGURAS = BASE / "figuras"
FIGURAS.mkdir(exist_ok=True)
viajes = pd.read_csv(BASE.parent / "starter" / "viajes_taxi.csv")

# 1. Retirar SOLO los errores confirmados en el ejercicio 1.
limpios = viajes.loc[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)].copy()
print(f"Viajes originales: {len(viajes)} | Viajes limpios: {len(limpios)} | Eliminados: {len(viajes) - len(limpios)}")

# 2. Distancia vs. tarifa, incluidos los valores altos validos.
plt.figure(figsize=(8, 5))
plt.scatter(limpios["distancia_km"], limpios["tarifa"], alpha=0.25, s=10)
plt.title("Relacion entre distancia y tarifa")
plt.xlabel("Distancia (km)")
plt.ylabel("Tarifa (USD)")
plt.tight_layout()
plt.savefig(FIGURAS / "distancia_vs_tarifa.png", dpi=140)
plt.close()
# Identificar la banda horizontal de tarifa fija, que no sigue una tarifa proporcional a distancia.
tarifa_fija = limpios.loc[limpios["tarifa"] == 70]
print(f"Viajes con tarifa exacta de $70: {len(tarifa_fija)}")
print("Destinos de esos viajes:")
print(tarifa_fija["zona_destino"].value_counts().to_string())

# 3. Propinas por metodo de pago, conservando los metodos faltantes.
propinas = limpios.groupby("metodo_pago", dropna=False).agg(
    viajes=("propina", "size"),
    propina_promedio=("propina", "mean"),
    viajes_propina_cero=("propina", lambda x: (x == 0).sum()),
)
print("\nPropina por metodo de pago:")
print(propinas.round(2).to_string())

# 4. Solo correlaciones de columnas relevantes numericas (sin textos ni ID).
columnas = ["distancia_km", "duracion_min", "pasajeros", "propina", "tarifa"]
print("\nCorrelaciones con la tarifa:")
print(limpios[columnas].corr(numeric_only=True)["tarifa"].round(3).to_string())

# 5. Pasajeros faltantes y registrados como cero.
print(f"\nPasajeros vacios: {limpios['pasajeros'].isna().sum()}")
print(f"Pasajeros igual a 0: {(limpios['pasajeros'] == 0).sum()}")
