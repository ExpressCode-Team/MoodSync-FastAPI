# MoodSync-FastAPI

## Prasyarat
1. [python](https://www.python.org/downloads/) (3.12.5)
2. [cmake](https://cmake.org/download/) (latest)
3. pip (latest)
4. [git lfs](https://git-lfs.com/) (latest)

## Untuk Menjalankan
1. Clone repository

```
git clone https://github.com/ExpressCode-Team/MoodSync-FastAPI.git
```

2. install semua keperluan library yang ada pada file `requirement.txt`

```
pip install -r requirement.txt
```

+ dlib terkadang akan mengalami error apabila menggunakan `pip install dlib` untuk memperbaiki hal ini pastikan cmake telah terinstall dan path cmake sudah ada pada environtment variable serta download tar.gz dlib pada [link ini](https://pypi.org/project/dlib/#files) lalu lakukan `pip install path/to/namafile.tar.gz` pada terminal.

3. jalankan kode ini

```
fastapi run
```

atau 

```
fastapi dev
```

## Catatan
1. Apabila file dengan bentuk .pkl bisa dibuka dan berisi bentuk link dari file tersebut **(bukan file asli)** maka dapat dinyatakan bahwa git lfs tidak bekerja dengan baik.

Solusi: hubungi pengurus fastAPI ini untuk mendapat file .pkl di luar kode ini.


Apabila masih terdapat pertanyaan bisa langsung bertanya ke pengurus fastAPI ini.

dokumentasi fastAPI bisa dipelajari lebih lanjut melalui [link ini](https://fastapi.tiangolo.com/)