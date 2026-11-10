"""Prueba una intención fuera del entrenamiento, sin cambiar el chatbot de reglas."""
import argparse
from pathlib import Path
import joblib
# Import explícito para disponer del tokenizador al recuperar el modelo con joblib.
import preprocesamiento  # noqa: F401

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predecir intención con TF-IDF + SVM')
    parser.add_argument('mensaje', help='Texto a clasificar entre comillas')
    parser.add_argument('--modelo', default=str(ROOT / 'resultados' / 'modelo_intenciones_svm.joblib'))
    args = parser.parse_args()
    # Cargar exclusivamente archivos propios/confiables: joblib usa pickle.
    modelo = joblib.load(args.modelo)
    print('Intención estimada:', modelo.predict([args.mensaje])[0])
    print('Nota: la predicción NO ejecuta acciones; el diálogo requiere contexto y validaciones.')
