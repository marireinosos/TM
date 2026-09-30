"""
🤍 Visión de Mari — Reconocimiento de imágenes con Teachable Machine
La app usa un modelo entrenado en Teachable Machine (keras_model.h5)
para adivinar qué hay en la foto que tomas o subes.
"""

import os
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Visión de Mari",
    page_icon="🤍",
    layout="wide",
)


def html(codigo):
    """Muestra HTML/CSS. Usa st.html (Streamlit nuevo) y si no existe, st.markdown."""
    if hasattr(st, "html"):
        st.html(codigo)
    else:
        st.markdown(codigo, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CLASES DEL MODELO
# Si tienes el archivo labels.txt de Teachable Machine, la app lo lee sola.
# Si no, usa esta lista (en el mismo orden en que las entrenaste).
# ─────────────────────────────────────────────
CLASES_POR_DEFECTO = ["Izquierda", "Arriba", "Derecha"]

EMOJIS = {
    "izquierda": "👈",
    "arriba": "👆",
    "derecha": "👉",
    "abajo": "👇",
}

UMBRAL = 0.5   # confianza mínima para decir que "sí es"


# ─────────────────────────────────────────────
# ESTILOS BEIGE & GIRLY 🤍
# ─────────────────────────────────────────────
html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&family=Poppins:wght@300;400;500;600&display=swap');

    .stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp li {
        font-family: 'Poppins', sans-serif;
    }
    .stApp h1, .stApp h2, .stApp h3 {
        font-family: 'Cormorant Garamond', serif !important;
        color: #6b4f3a !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, #fbeee6 0%, transparent 40%),
            radial-gradient(circle at 90% 30%, #f5e6d8 0%, transparent 45%),
            #f8f2eb;
    }
    p, li, label { color: #7a6250; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #fffaf4 !important;
        border-right: 1px solid #eadbc8;
    }
    [data-testid="stSidebar"] h2 {
        font-family: 'Cormorant Garamond', serif !important;
        font-style: italic;
        font-size: 2rem !important;
        color: #8b6a50 !important;
    }
    [data-testid="stSidebar"] h3 {
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #b08d6e !important;
    }

    /* Cajas */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #eadbc8 !important;
        border-radius: 26px !important;
        background: #fffdf9;
        box-shadow: 0 10px 30px rgba(139,106,80,0.08);
    }

    /* Botones */
    .stButton > button {
        background: #d9b99b !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        background: #c9a57f !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(185,145,105,0.35);
    }

    /* Encabezado */
    .hero {
        background: linear-gradient(135deg, #fffaf4 0%, #f3e3d3 55%, #ecd3c5 100%);
        border: 1px solid #eadbc8;
        border-radius: 34px;
        padding: 44px 40px 38px;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 14px 40px rgba(139,106,80,0.12);
        font-family: 'Poppins', sans-serif;
    }
    .hero .mini {
        font-size: 0.72rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: #b08d6e;
    }
    .hero h1 {
        font-family: 'Cormorant Garamond', serif;
        font-style: italic;
        font-weight: 600;
        font-size: 3.4rem;
        color: #6b4f3a;
        margin: 6px 0 4px;
    }
    .hero p {
        color: #8b6a50;
        font-size: 1rem;
        margin: 0;
    }
    .brillo {
        position: absolute;
        font-size: 1.3rem;
        opacity: 0.7;
        animation: flotar 5s ease-in-out infinite;
    }
    @keyframes flotar {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50%      { transform: translateY(-8px) rotate(10deg); }
    }

    /* Pasos */
    .pasos {
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        margin-top: 18px;
        font-family: 'Poppins', sans-serif;
    }
    .paso {
        flex: 1;
        min-width: 150px;
        background: #fffdf9;
        border: 1px solid #eadbc8;
        border-radius: 22px;
        padding: 16px;
        text-align: center;
        color: #7a6250;
        font-size: 0.88rem;
        transition: transform 0.25s;
    }
    .paso:hover { transform: translateY(-4px); }
    .paso b {
        display: block;
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.9rem;
        color: #c9a57f;
        font-weight: 600;
    }

    /* Resultado ganador */
    .ganador {
        background: linear-gradient(135deg, #f3e3d3, #ecd3c5);
        border-radius: 28px;
        padding: 30px;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        animation: aparecer 0.6s ease;
    }
    .ganador .emoji { font-size: 3.4rem; }
    .ganador .nombre {
        font-family: 'Cormorant Garamond', serif;
        font-style: italic;
        font-size: 2.6rem;
        font-weight: 600;
        color: #6b4f3a;
        line-height: 1.1;
    }
    .ganador .conf {
        margin-top: 6px;
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #a07e62;
    }
    .ganador.dudoso { background: #f5eee6; }
    @keyframes aparecer {
        from { opacity: 0; transform: scale(0.95); }
        to   { opacity: 1; transform: scale(1); }
    }

    /* Barras de probabilidad */
    .barra-fila {
        margin: 12px 0;
        font-family: 'Poppins', sans-serif;
    }
    .barra-top {
        display: flex;
        justify-content: space-between;
        color: #6b4f3a;
        font-size: 0.92rem;
        margin-bottom: 5px;
    }
    .barra-fondo {
        background: #f3e9df;
        border-radius: 999px;
        height: 12px;
        overflow: hidden;
    }
    .barra-llena {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #e8cfc0, #c9a57f);
        transition: width 0.8s ease;
    }
    .barra-llena.top { background: linear-gradient(90deg, #d9a89c, #b8866b); }

    .footer {
        text-align: center;
        color: #b08d6e;
        font-family: 'Cormorant Garamond', serif;
        font-style: italic;
        font-size: 1.1rem;
        padding: 26px 0 6px;
    }
</style>
""")


# ─────────────────────────────────────────────
# CARGAR MODELO (solo una vez, para que sea rápida)
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return load_model("keras_model.h5", compile=False)


def cargar_clases():
    if os.path.exists("labels.txt"):
        with open("labels.txt", encoding="utf-8") as f:
            clases = []
            for linea in f:
                linea = linea.strip()
                if linea:
                    # Teachable Machine escribe "0 Izquierda": quitamos el número
                    partes = linea.split(" ", 1)
                    clases.append(partes[1] if len(partes) == 2 and partes[0].isdigit() else linea)
            if clases:
                return clases
    return CLASES_POR_DEFECTO


def preparar_imagen(img):
    """Deja la imagen igual a como la espera Teachable Machine (224x224, valores -1 a 1)."""
    img = img.convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    arreglo = np.asarray(img).astype(np.float32)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = (arreglo / 127.5) - 1
    return data


def emoji_de(clase):
    return EMOJIS.get(clase.lower().strip(), "✨")


try:
    modelo = cargar_modelo()
except Exception as e:
    st.error("No pude cargar el modelo 😢 Revisa que el archivo **keras_model.h5** esté en el repo.")
    st.caption(f"Detalle: {e}")
    st.stop()

clases = cargar_clases()


# ─────────────────────────────────────────────
# PANEL LATERAL
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Visión de Mari")
    st.caption("Una IA que aprendió a reconocer imágenes 🤍")
    st.divider()

    st.markdown("### Imagen")
    fuente = st.radio("fuente", ["📸 Tomar foto", "🖼️ Subir imagen"],
                      label_visibility="collapsed")

    st.divider()
    st.markdown("### Lo que sabe reconocer")
    for c in clases:
        st.write(f"{emoji_de(c)} {c}")

    st.divider()
    st.markdown("### ¿Cómo funciona?")
    st.caption(
        "Este modelo se entrenó en **Teachable Machine** de Google con fotos de ejemplo "
        "de cada clase. Cuando le muestras una foto nueva, calcula qué tan parecida es "
        "a cada una y te dice la más probable."
    )


# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────
html("""
<div class="hero">
    <span class="brillo" style="top:22px; left:34px;">🤍</span>
    <span class="brillo" style="top:40px; right:48px; animation-delay:1.2s;">✨</span>
    <span class="brillo" style="bottom:20px; left:20%; animation-delay:2.2s;">🎀</span>
    <span class="brillo" style="bottom:24px; right:22%; animation-delay:1.6s;">☁️</span>
    <div class="mini">Reconocimiento de imágenes · IA</div>
    <h1>Visión de Mari</h1>
    <p>Muéstrale una foto y mi modelo adivinará lo que ve</p>
</div>
<div class="pasos">
    <div class="paso"><b>01</b>Toma o sube una foto</div>
    <div class="paso"><b>02</b>El modelo la analiza</div>
    <div class="paso"><b>03</b>Te dice qué ve y qué tan segura está</div>
</div>
""")

st.write("")


# ─────────────────────────────────────────────
# ENTRADA DE IMAGEN
# ─────────────────────────────────────────────
col_foto, col_resultado = st.columns([1, 1], gap="large")

with col_foto:
    with st.container(border=True):
        st.markdown("### Tu foto")
        if fuente == "📸 Tomar foto":
            archivo = st.camera_input("Toma una foto", label_visibility="collapsed")
        else:
            archivo = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg", "webp"],
                                       label_visibility="collapsed")
            if archivo is not None:
                st.image(archivo, use_container_width=True)

        # Imagen decorativa del proyecto (si existe en el repo)
        if archivo is None and os.path.exists("OIG5.jpg"):
            st.image("OIG5.jpg", use_container_width=True)


# ─────────────────────────────────────────────
# RESULTADO
# ─────────────────────────────────────────────
with col_resultado:
    with st.container(border=True):
        st.markdown("### Lo que veo")

        if archivo is None:
            st.caption("Aquí aparecerá el resultado cuando tomes o subas una foto 🤍")
            html("""
            <div class="ganador dudoso">
                <div class="emoji">☁️</div>
                <div class="nombre">Esperando tu foto...</div>
            </div>
            """)
        else:
            with st.spinner("Mirando tu foto con atención ✨"):
                datos = preparar_imagen(Image.open(archivo))
                prediccion = modelo.predict(datos, verbose=0)[0]

            # por si el modelo tiene más o menos clases que la lista
            nombres = [clases[i] if i < len(clases) else f"Clase {i + 1}"
                       for i in range(len(prediccion))]

            i_top = int(np.argmax(prediccion))
            conf_top = float(prediccion[i_top])

            if conf_top >= UMBRAL:
                html(f"""
                <div class="ganador">
                    <div class="emoji">{emoji_de(nombres[i_top])}</div>
                    <div class="nombre">{nombres[i_top]}</div>
                    <div class="conf">{conf_top * 100:.1f}% de seguridad</div>
                </div>
                """)
            else:
                html(f"""
                <div class="ganador dudoso">
                    <div class="emoji">🤔</div>
                    <div class="nombre">No estoy tan segura...</div>
                    <div class="conf">Lo más parecido: {nombres[i_top]} ({conf_top * 100:.1f}%)</div>
                </div>
                """)

            st.write("")
            st.markdown("#### Probabilidad de cada clase")
            filas = ""
            orden = np.argsort(prediccion)[::-1]
            for i in orden:
                p = float(prediccion[i]) * 100
                clase_top = "top" if i == i_top else ""
                filas += f"""
                <div class="barra-fila">
                    <div class="barra-top">
                        <span>{emoji_de(nombres[i])} {nombres[i]}</span>
                        <span>{p:.1f}%</span>
                    </div>
                    <div class="barra-fondo">
                        <div class="barra-llena {clase_top}" style="width:{max(p, 1):.1f}%;"></div>
                    </div>
                </div>
                """
            html(filas)


html('<div class="footer">hecho con cariño por Mari 🤍</div>')
