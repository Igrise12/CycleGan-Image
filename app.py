from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from gan_art.inference import load_generators, translate_image


CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints/cyclegan_v2_fast/latest.pt"
DIRECTIONS = {
    "Foto → Monet": ("photo_to_monet", "monet"),
    "Monet → Foto": ("monet_to_photo", "photo"),
}

st.set_page_config(page_title="GAN Art", page_icon="🎨", layout="wide")
st.title("GAN Art")
st.write("Ubah foto menjadi lukisan bergaya Monet, atau terjemahkan lukisan Monet menjadi foto.")

direction = st.radio("Arah transformasi", DIRECTIONS, horizontal=True)
uploaded = st.file_uploader("Pilih gambar", type=("jpg", "jpeg", "png", "webp"))


@st.cache_resource
def get_generators(checkpoint_path: str, checkpoint_version: tuple[int, int]):
    return load_generators(Path(checkpoint_path))


if uploaded is not None:
    try:
        source = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("File ini tidak bisa dibaca sebagai gambar.")
        st.stop()

    model_name, output_style = DIRECTIONS[direction]
    input_column, output_column = st.columns(2)
    input_column.image(source, caption="Gambar input", use_container_width=True)

    if not CHECKPOINT_PATH.is_file():
        output_column.error(f"Checkpoint tidak ditemukan: {CHECKPOINT_PATH}")
    else:
        try:
            with st.spinner("Memuat model dan membuat gambar…"):
                checkpoint_stat = CHECKPOINT_PATH.stat()
                checkpoint_version = (checkpoint_stat.st_mtime_ns, checkpoint_stat.st_size)
                generators, device = get_generators(str(CHECKPOINT_PATH), checkpoint_version)
                result = translate_image(source, generators[model_name], device)
        except Exception as error:
            st.error(f"Gagal memproses gambar: {error}")
        else:
            output_column.image(result, caption=direction, use_container_width=True)
            image_bytes = BytesIO()
            result.save(image_bytes, format="PNG")
            output_column.download_button(
                "Unduh hasil PNG",
                data=image_bytes.getvalue(),
                file_name=f"{Path(uploaded.name).stem}_{output_style}.png",
                mime="image/png",
                use_container_width=True,
            )
            st.caption(f"Diproses pada {device}. Hasil mengikuti input model 256 × 256 piksel.")
