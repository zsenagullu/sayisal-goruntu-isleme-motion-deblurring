# Richardson-Lucy ile Hareket Bulanıklığı Giderme

Bu proje, net fotoğraflardan kontrollü hareket bulanıklığı üretir, Richardson-Lucy algoritmasıyla görüntüyü geri kazanmaya çalışır ve PSNR, SSIM ile işlem süresini kaydeder. İşlemler CPU üzerinde çalışır; GPU gerekmez.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## İlk demo

Önce daha hafif bir bulanıklık koşuluyla, tam 512 px uzun kenar çözünürlüğünde dene:

```bash
python main.py --demo --blur-lengths 9 17 --angles 45 --iterations 50 --max-side 512
```

Daha zor koşullara geçmek için:

```bash
python main.py --demo --blur-lengths 9 17 --angles 0 45 --iterations 30 --max-side 512
```

## Kendi veri setin

Net JPG/PNG fotoğraflarını `data/raw/` klasörüne koy ve `--demo` seçeneği olmadan çalıştır:

```bash
python main.py --blur-lengths 9 17 25 --angles 0 45 90 --iterations 30 --max-side 512
```

## Çıktılar

- `outputs/comparisons/`: Orijinal, bulanık ve geri kazanılmış görüntünün tam çözünürlüklü yan yana karşılaştırması.
- `outputs/restored/`: Tek başına geri kazanılmış görüntüler.
- `results/metrics_iterN.csv`: İterasyon sayısı N olan çalışmanın metrikleri.
- `results/metrics.csv`: En son çalışmanın metrikleri.

## Deney tasarımı hakkında

Sentetik deneylerde bulanıklık çekirdeği (PSF) bilindiği için algoritmaya aynı çekirdek verilir. Bu, kontrollü bir başlangıç testidir. Gerçek hayatta çekim sırasında oluşan bulanıklığın PSF'si genellikle bilinmez; gerçek bulanık fotoğrafları iyileştirmeye geçerken PSF tahminine ilişkin ayrı bir adım gerekecektir.

Richardson-Lucy iterasyon sayısını artırmak her zaman daha iyi sonuç vermez. Çok yüksek iterasyon sayıları gürültüyü ve yapay kenarları artırabilir; sonuçlar hem görsel olarak hem PSNR/SSIM ölçütleriyle incelenmelidir.
