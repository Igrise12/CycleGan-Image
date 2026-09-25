# GAN Art

App Streamlit untuk mengubah foto menjadi gaya Monet dengan checkpoint CycleGAN yang sudah ada.

```bash
uv sync
uv run streamlit run app.py
```

App memakai `checkpoints/cyclegan_v2/latest.pt` dan mendukung terjemahan Foto → Monet serta Monet → Foto. Model mengikuti preprocessing saat training, jadi hasilnya berukuran 256 × 256 piksel.
