# ML-49

# 🧠 Custom Feedforward Neural Network (FFNN)

Proyek ini adalah implementasi dari _Feedforward Neural Network (FFNN)_ yang dibangun dari nol menggunakan PyTorch. Framework ini mendukung berbagai fungsi aktivasi, metode inisialisasi bobot, fungsi loss, serta opsional penggunaan RMS Normalization. Selain itu, disediakan juga fitur visualisasi struktur jaringan dan distribusi bobot/gradien.

## 📁 Struktur File

- `ffnn.py` — Class utama untuk FFNN: training, forward, backward, visualisasi.
- `layer.py` — Implementasi satu layer neural network dengan RMSNorm opsional.
- `initializers.py` — Fungsi-fungsi inisialisasi bobot dan bias.
- `activations.py` — Berbagai fungsi aktivasi dan turunannya.
- `loss_functions.py` — Tiga jenis loss: MSE, BCE, Categorical Cross Entropy.
- `rms_norm.py` — Implementasi RMS Normalization.
- `main.ipynb` — Notebook eksplorasi dan contoh penggunaan.

## ⚙️ Cara Setup & Jalankan

### 1. Clone repository

```bash
git clone https://github.com/mybajwk/ML-49.git
cd ML-49
```

### 2. Buat Virtual Environment & Install Dependency

Disarankan menggunakan virtual environment agar dependency terisolasi.

#### a. Buat dan aktifkan virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# MacOS/Linux
source venv/bin/activate
```

#### b. Install dependency dari `requirement.txt`:

```bash
pip install -r requirement.txt
```

#### c. (Opsional) Tambahkan kernel environment ke Jupyter Notebook:

```bash
pip install ipykernel
python -m ipykernel install --user --name=venv-ml49 --display-name "Python (ML-49)"
```

Pastikan kernel `Python (ML-49)` dipilih saat membuka `main.ipynb` di VS Code.

### 3. Jalankan Notebook

Buka `main.ipynb` di VS Code dan tekan tombol ▶️ (Run All) untuk mengeksekusi semua kode.
Pastikan kernel Python yang sesuai sudah dipilih.

Alternatif lain, jalankan dari terminal dengan:

```bash
jupyter notebook main.ipynb
```

---

## 🚀 Inisialisasi Kelas FFNN

Berikut adalah cara untuk menginisialisasi model FFNN:

```python
from ffnn import FFNN

model = FFNN(
    layer_sizes=[4, 8, 6, 3],                          # Ukuran tiap layer: input, hidden1, hidden2, output
    activations_list=["relu", "tanh", "softmax"],      # Fungsi aktivasi per layer
    loss_function="mse",                               # 'mse', 'bce', atau 'cce'
    weight_inits=["he", "xavier", "random_normal"],     # Inisialisasi bobot berbeda tiap layer
    init_params_list=[                                  # Param tambahan untuk setiap metode
        {"seed": 42},                                   # Untuk He
        {"seed": 42},                                   # Untuk Xavier
        {"mean": 0.0, "variance": 0.01, "seed": 42}     # Untuk Normal
    ],
    regularization="L2",                                # 'none', 'L1', atau 'L2'
    req_lambda=0.01,                                    # Nilai lambda untuk regulasi
    use_rmsnorm=True                                    # Gunakan RMSNorm di setiap layer
)
```

### 📌 Detail Parameter `init_params`

Parameter `init_params` adalah dictionary opsional untuk mengatur parameter tambahan sesuai metode inisialisasi bobot yang digunakan:

| Metode Inisialisasi | Parameter `init_params` yang Didukung                        | Default         |
| ------------------- | ------------------------------------------------------------ | --------------- |
| `random_uniform`    | `lower`, `upper`, `seed`                                     | -0.5, 0.5, None |
| `random_normal`     | `mean`, `variance`, `seed`                                   | 0.0, 1.0, None  |
| `zero`              | `seed` (optional, jarang digunakan)                          | None            |
| `he_xavier`         | `seed` (otomatis memilih `he` atau `xavier` sesuai aktivasi) | None            |

Contoh penggunaan:

```python
init_params = {
    "seed": 42,
    "lower": -1.0,
    "upper": 1.0
}
```

---

## 🏋️ Cara Melatih (Train) Model FFNN

Setelah model `FFNN` diinisialisasi, kamu bisa melatih model dengan memanggil fungsi `.train()`:

```python
history = model.train(
    X_train,          # Input data training (Tensor)
    y_train,          # Target data training (Tensor)
    X_val=None,       # (Opsional) Input data validasi
    y_val=None,       # (Opsional) Target data validasi
    epochs=100,       # Jumlah epoch pelatihan
    batch_size=32,    # Ukuran batch
    learning_rate=0.01, # Learning rate (α)
    verbose=1,        # 0 = tidak cetak apa-apa, 1 = tampilkan progress
    tol=1e-4,         # Toleransi minimum perbedaan loss antar epoch (jika early stopping aktif)
    patience=10,      # Jumlah epoch tanpa peningkatan sebelum berhenti (jika early stopping aktif)
    stop_in_convergence=False  # Aktifkan early stopping jika True
)
```

### 📌 Penjelasan Parameter `.train()`

| Parameter             | Deskripsi                                                               |
| --------------------- | ----------------------------------------------------------------------- |
| `X_train`, `y_train`  | Data training (dalam bentuk Tensor)                                     |
| `X_val`, `y_val`      | Data validasi opsional                                                  |
| `epochs`              | Jumlah iterasi penuh terhadap dataset                                   |
| `batch_size`          | Jumlah sampel per batch                                                 |
| `learning_rate`       | Ukuran langkah update bobot                                             |
| `verbose`             | Tampilkan log training jika 1, tidak tampilkan jika 0                   |
| `tol`                 | Perbedaan minimum untuk mendeteksi konvergensi (untuk early stopping)   |
| `patience`            | Jumlah epoch berturut-turut tanpa peningkatan sebelum berhenti training |
| `stop_in_convergence` | Jika `True`, aktifkan early stopping berdasarkan `tol` dan `patience`   |

---

## 🔍 Fitur Utama

- ✅ Multi-layer FFNN dengan konfigurasi bebas
- ✅ RMS Normalization
- ✅ Pilihan inisialisasi bobot: He, Xavier, Uniform, Normal, Zero, He/Xavier otomatis
- ✅ Fungsi aktivasi: ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, Softsign
- ✅ Fungsi loss: MSE, BCE, CCE
- ✅ Visualisasi struktur jaringan, bobot, dan gradien antar layer

---

## 👥 Pembagian Tugas

| Nama Anggota       | NIM      | Tugas         |
| ------------------ | -------- | ------------- |
| Mesach Harmasendro | 13522117 | Code, Laporan |
| Enrique Yanuar     | 13522077 | Code, Laporan |
| Wilson Yusda       | 13522019 | Code, Laporan |
