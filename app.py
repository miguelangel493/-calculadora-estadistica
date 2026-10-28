import importlib.util
import io
import math
import os
import re
import shutil

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Calculadora estadística", layout="wide")
st.title("Calculadora estadística")


# ---------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------
def fmt(x):
    """Formatea un valor con hasta 4 decimales (o '—' si no existe)."""
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(x)


def leer_manual(texto, tipo):
    """Convierte el texto escrito a mano en una serie de datos."""
    partes = [p.strip() for p in texto.replace("\n", ",").replace(";", ",").split(",")]
    partes = [p for p in partes if p]
    if tipo == "Cuantitativa":
        try:
            return pd.Series([float(p) for p in partes], dtype=float)
        except ValueError:
            st.error(
                "Revisa los datos: solo números separados por comas "
                "(usa punto para los decimales)."
            )
            return None
    return pd.Series(partes)


def leer_archivo(archivo):
    """Lee un archivo CSV o Excel (.xlsx) y devuelve un DataFrame."""
    if archivo.name.lower().endswith(".csv"):
        try:
            return pd.read_csv(archivo, sep=None, engine="python")
        except UnicodeDecodeError:
            archivo.seek(0)
            return pd.read_csv(archivo, sep=None, engine="python", encoding="latin-1")
    libro = pd.ExcelFile(archivo)
    hojas = libro.sheet_names
    hoja = st.selectbox("Hoja del libro", hojas) if len(hojas) > 1 else hojas[0]
    return pd.read_excel(libro, sheet_name=hoja)


def tabla_frecuencias(etiquetas, fi, n):
    """Arma la tabla de frecuencias a partir de etiquetas y conteos."""
    tabla = pd.DataFrame({"Categoría": etiquetas, "fi": fi})
    tabla["hi"] = tabla["fi"] / n
    tabla["%"] = tabla["hi"] * 100
    tabla["Fi"] = tabla["fi"].cumsum()
    tabla["Hi"] = tabla["hi"].cumsum()
    tabla[["hi", "Hi"]] = tabla[["hi", "Hi"]].round(4)
    tabla["%"] = tabla["%"].round(2)
    return tabla


# ---------------------------------------------------------------
# Lectura de datos desde una imagen (OCR con Tesseract)
# ---------------------------------------------------------------
def buscar_tesseract():
    """Busca el programa Tesseract en el sistema (servidor Linux o Windows)."""
    ruta = shutil.which("tesseract")
    if ruta:
        return ruta
    for candidata in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.exists(candidata):
            return candidata
    return None


TESSERACT = buscar_tesseract()


@st.cache_data(show_spinner=False)
def leer_imagen(contenido, solo_numeros, disposicion):
    """Devuelve las líneas de texto que Tesseract encuentra en la imagen."""
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = TESSERACT

    imagen = Image.open(io.BytesIO(contenido))
    imagen = ImageOps.exif_transpose(imagen).convert("L")  # orientación y escala de grises
    lado = max(imagen.size)
    if lado < 1500:  # amplía imágenes pequeñas para leer mejor
        factor = 1500 / lado
        imagen = imagen.resize(
            (int(imagen.width * factor), int(imagen.height * factor)),
            Image.Resampling.LANCZOS,
        )
    elif lado > 2400:  # reduce fotos muy grandes
        imagen.thumbnail((2400, 2400))
    imagen = ImageOps.autocontrast(imagen)

    psm = 6 if disposicion == "Lista o tabla" else 11
    config = f"--psm {psm}"
    if solo_numeros:
        config += " -c tessedit_char_whitelist=0123456789.,-"
        idioma = "eng"
    else:
        idioma = "spa+eng"
    texto = pytesseract.image_to_string(imagen, lang=idioma, config=config)
    return [linea.strip() for linea in texto.splitlines() if linea.strip()]


def extraer_numeros(lineas):
    """Saca los números de las líneas leídas (acepta coma o punto decimal)."""
    numeros = []
    for linea in lineas:
        for encontrado in re.findall(r"-?\d+(?:[.,]\d+)?", linea):
            numeros.append(encontrado.replace(",", "."))
    return numeros


# ---------------------------------------------------------------
# Análisis de variables cuantitativas
# ---------------------------------------------------------------
def analisis_cuantitativo(datos):
    n = len(datos)
    if n < 2:
        st.info("Ingresa al menos 2 datos para calcular las medidas.")
        return

    base = st.radio("Los datos son una...", ["Muestra", "Población"], horizontal=True)
    ddof = 1 if base == "Muestra" else 0
    sufijo = "muestral" if ddof == 1 else "poblacional"

    media = datos.mean()
    mediana = datos.median()
    if datos.value_counts().max() == 1:
        moda = "Sin moda"
    else:
        moda = ", ".join(fmt(m) for m in datos.mode())

    desv = datos.std(ddof=ddof)
    q1 = datos.quantile(0.25)
    q3 = datos.quantile(0.75)
    cv = desv / abs(media) * 100 if media != 0 else float("nan")

    central = pd.DataFrame(
        {
            "Medida": ["Datos (n)", "Media", "Mediana", "Moda"],
            "Valor": [str(n), fmt(media), fmt(mediana), moda],
        }
    )
    forma = pd.DataFrame(
        {
            "Medida": ["Asimetría", "Curtosis (exceso)"],
            "Valor": [fmt(datos.skew()), fmt(datos.kurt())],
        }
    )
    dispersion = pd.DataFrame(
        {
            "Medida": [
                "Mínimo",
                "Máximo",
                "Rango",
                f"Varianza ({sufijo})",
                f"Desviación estándar ({sufijo})",
                "Coeficiente de variación (%)",
                "Q1",
                "Q3",
                "Rango intercuartílico",
            ],
            "Valor": [
                fmt(datos.min()),
                fmt(datos.max()),
                fmt(datos.max() - datos.min()),
                fmt(datos.var(ddof=ddof)),
                fmt(desv),
                fmt(cv),
                fmt(q1),
                fmt(q3),
                fmt(q3 - q1),
            ],
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tendencia central")
        st.table(central.set_index("Medida"))
        st.subheader("Forma de la distribución")
        st.table(forma.set_index("Medida"))
    with col2:
        st.subheader("Dispersión")
        st.table(dispersion.set_index("Medida"))
    st.caption(
        "Los cuartiles usan interpolación lineal (igual que QUARTIL.INC en Excel). "
        "La asimetría y la curtosis son las de Fisher."
    )

    st.subheader("Gráficos")
    k = max(1, math.ceil(1 + math.log2(n)))  # regla de Sturges
    tab1, tab2, tab3 = st.tabs(["Histograma", "Diagrama de caja", "Tabla de frecuencias"])

    with tab1:
        fig = px.histogram(x=datos, nbins=k, title="Histograma")
        fig.update_layout(xaxis_title="Valor", yaxis_title="Frecuencia", bargap=0.05)
        st.plotly_chart(fig)

    with tab2:
        fig = px.box(x=datos, title="Diagrama de caja", points="outliers")
        fig.update_layout(xaxis_title="Valor")
        st.plotly_chart(fig)

    with tab3:
        if datos.nunique() > 1:
            cortes = pd.cut(datos, bins=k, include_lowest=True)
            fi = cortes.value_counts(sort=False)
            tabla = tabla_frecuencias(fi.index.astype(str), fi.values, n)
            tabla = tabla.rename(columns={"Categoría": "Intervalo"})
            st.dataframe(tabla, hide_index=True)
            st.caption(f"{k} intervalos calculados con la regla de Sturges.")
        else:
            st.info("Todos los datos son iguales; no hay intervalos que mostrar.")


# ---------------------------------------------------------------
# Análisis de variables cualitativas
# ---------------------------------------------------------------
def analisis_cualitativo(datos):
    datos = datos.astype(str).str.strip()
    datos = datos[datos != ""]
    n = len(datos)
    if n < 1:
        st.info("Ingresa al menos 1 dato.")
        return

    ordenar = st.checkbox("Ordenar categorías de mayor a menor frecuencia", value=True)
    conteo = datos.value_counts()
    if not ordenar:
        conteo = conteo.sort_index()

    tabla = tabla_frecuencias(conteo.index, conteo.values, n)
    maximo = tabla["fi"].max()
    modas = tabla.loc[tabla["fi"] == maximo, "Categoría"].tolist()

    c1, c2, c3 = st.columns(3)
    c1.metric("Datos (n)", n)
    c2.metric("Categorías", len(tabla))
    c3.metric("Moda", ", ".join(modas))
    if len(modas) > 1:
        st.caption("Hay empate: la variable es multimodal.")

    st.subheader("Tabla de frecuencias")
    st.dataframe(tabla, hide_index=True)
    st.caption(
        "fi: frecuencia absoluta · hi: frecuencia relativa · % : porcentaje · "
        "Fi y Hi: acumuladas (tienen sentido si las categorías tienen un orden)."
    )

    st.subheader("Gráficos")
    tab1, tab2 = st.tabs(["Barras", "Circular"])

    with tab1:
        fig = px.bar(tabla, x="Categoría", y="fi", text="fi", title="Gráfico de barras")
        fig.update_xaxes(type="category")
        fig.update_layout(yaxis_title="Frecuencia")
        if maximo <= 10:
            fig.update_yaxes(dtick=1)
        st.plotly_chart(fig)

    with tab2:
        fig = px.pie(tabla, names="Categoría", values="fi", title="Gráfico circular")
        st.plotly_chart(fig)


# ---------------------------------------------------------------
# Entrada de datos
# ---------------------------------------------------------------
# La opción de imagen solo aparece si Tesseract está instalado
# (en el servidor se instala con packages.txt; en tu PC, solo si lo instalaste).
OCR_DISPONIBLE = TESSERACT is not None and importlib.util.find_spec("pytesseract") is not None
opciones_fuente = ["Ingreso manual", "Archivo Excel / CSV"]
if OCR_DISPONIBLE:
    opciones_fuente.append("Imagen (foto)")

fuente = st.radio(
    "¿De dónde vienen los datos?",
    opciones_fuente,
    horizontal=True,
)

datos = None
tipo = None

if fuente == "Ingreso manual":
    tipo = st.radio("Tipo de variable", ["Cuantitativa", "Cualitativa"], horizontal=True)
    if tipo == "Cuantitativa":
        ejemplo = "12, 15, 15, 18, 20, 22, 22, 22, 30"
    else:
        ejemplo = "Rojo, Azul, Azul, Verde, Rojo, Rojo, Verde"
    texto = st.text_area(
        "Ingresa los datos separados por comas o saltos de línea",
        ejemplo,
        key=f"texto_{tipo}",
    )
    datos = leer_manual(texto, tipo)

elif fuente == "Imagen (foto)":
    tipo = st.radio("Tipo de variable", ["Cuantitativa", "Cualitativa"], horizontal=True)
    disposicion = st.radio(
        "¿Cómo están los datos en la foto?",
        ["Lista o tabla", "Datos sueltos"],
        horizontal=True,
    )
    imagen = st.file_uploader(
        "Sube una foto o captura con los datos",
        type=["png", "jpg", "jpeg"],
    )
    if st.checkbox("Tomar la foto con la cámara"):
        foto = st.camera_input("Toma la foto")
        if foto is not None:
            imagen = foto
    if imagen is None:
        st.info("Sube o toma una foto donde se vean claramente los datos.")
        st.stop()

    st.image(imagen, caption="Imagen cargada")
    try:
        with st.spinner("Leyendo la imagen..."):
            lineas = leer_imagen(imagen.getvalue(), tipo == "Cuantitativa", disposicion)
    except ImportError:
        st.error("El lector de imágenes no está disponible en este equipo.")
        st.stop()
    except Exception as e:
        st.error(f"No se pudo leer la imagen: {e}")
        st.stop()

    if tipo == "Cuantitativa":
        sugerido = ", ".join(extraer_numeros(lineas))
    else:
        sugerido = ", ".join(l.strip() for l in lineas if l.strip())
    if not sugerido:
        st.warning(
            "No se encontraron datos en la imagen. Prueba con una foto más nítida "
            "o escribe los datos a mano en el cuadro de abajo."
        )
    st.caption(
        "Revisa y corrige lo que se leyó antes de calcular: el OCR puede confundir "
        "caracteres (por ejemplo 0 con O, o 1 con 7)."
    )
    texto = st.text_area(
        "Datos leídos (puedes editarlos)",
        sugerido,
        key=f"ocr_{tipo}_{imagen.name}_{imagen.size}",
    )
    datos = leer_manual(texto, tipo)

else:
    archivo = st.file_uploader("Sube tu archivo", type=["xlsx", "csv"])
    if archivo is None:
        st.info(
            "Sube un archivo .xlsx o .csv con los datos en columnas. "
            "La primera fila debe tener el nombre de cada columna."
        )
        st.stop()

    try:
        df = leer_archivo(archivo)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()

    if df.empty:
        st.warning("El archivo no tiene datos.")
        st.stop()

    with st.expander("Vista previa de los datos"):
        st.dataframe(df.head(20))

    columna = st.selectbox("Columna a analizar", df.columns)
    serie = df[columna].dropna()
    es_numerica = pd.api.types.is_numeric_dtype(serie)

    tipo = st.radio(
        "Tipo de variable",
        ["Cuantitativa", "Cualitativa"],
        index=0 if es_numerica else 1,
        horizontal=True,
        key=f"tipo_{columna}",
    )

    if tipo == "Cuantitativa":
        convertida = pd.to_numeric(serie, errors="coerce")
        descartados = int(convertida.isna().sum())
        if descartados:
            st.warning(f"Se ignoraron {descartados} valores que no son numéricos.")
        datos = convertida.dropna().astype(float)
    else:
        datos = serie


# ---------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------
if datos is None:
    st.stop()
if len(datos) == 0:
    st.info("No hay datos para analizar.")
    st.stop()

st.divider()
if tipo == "Cuantitativa":
    analisis_cuantitativo(datos)
else:
    analisis_cualitativo(datos)
