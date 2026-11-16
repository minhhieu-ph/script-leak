# scriptleak v4

Mã nguồn cho nghiên cứu "rò ngôn ngữ thiểu số trong ASR song ngữ nhỏ gọn".
Bản v4 đi hết một vòng thí nghiệm cho đầu CTC: chuẩn bị dữ liệu, huấn luyện,
lưu xác suất theo khung, phân tích nguyên nhân, giải mã và so sánh phương pháp.

## Đã kiểm thử đến đâu

| Phần | Mô-đun | Trạng thái |
|---|---|---|
| Lọc dạng nói, trộn, chia tập | `audit`, `mix`, `split` | Có kiểm thử tự động |
| Chỉ số rò, CER, WER | `metrics`, `evaluate` | Có kiểm thử tự động |
| Giải mã và baseline | `decoding`, `sweep` | Có kiểm thử trên xác suất tổng hợp |
| Phân tích theo khung | `analysis` | Có kiểm thử trên xác suất tổng hợp |
| Thêm nhiễu | `noise` | Phần trộn tín hiệu có kiểm thử; phần đọc ghi tệp chưa chạy |
| Chuyển định dạng | `prepare` | LibriSpeech, AISHELL-1, TSV có kiểm thử cấu trúc thư mục; Hugging Face chưa chạy |
| Mọi thứ dùng NeMo | `nemo_tools/*` | **Chưa chạy lần nào**, vì nơi viết mã không có NeMo và GPU |

`python -m pytest -q` chạy 71 bài kiểm thử, chỉ cần numpy và scipy.

Vì phần NeMo chưa được chạy, bước đầu tiên trên máy huấn luyện là:

```bash
python -m scriptleak.nemo_tools.check_env
```

Lệnh này kiểm tra từng điểm của NeMo mà mã dựa vào. Nếu có dòng `FAIL`, gửi
lại toàn bộ kết quả để sửa trước khi huấn luyện.

## Cài đặt môi trường

```bash
bash scripts/setup_env.sh
```

Script tạo môi trường conda `scriptleak` (Python 3.12), cài PyTorch, NeMo và các
gói còn lại, tải kho NeMo về `third_party/NeMo` để lấy tệp cấu hình, rồi chạy
bộ kiểm thử và `check_env`. Script chưa được chạy thử vì nơi viết mã không có
mạng.

| Biến | Mặc định | Khi nào đổi |
|---|---|---|
| `TORCH_INDEX` | `https://download.pytorch.org/whl/cu130` | Khi PyTorch không thấy GPU. Chọn bản khớp driver theo trang cài đặt của PyTorch |
| `SKIP_TORCH=1` | tắt | Khi môi trường đã có PyTorch chạy được với GPU |
| `SKIP_CONDA=1` | tắt | Khi tự quản lý môi trường Python |
| `ENV_NAME` | `scriptleak` | Tên môi trường conda |

NeMo hiện là bản 3.0 (kho đổi tên thành NeMo Speech), yêu cầu Python 3.12 và
PyTorch 2.7 trở lên. Mã trong `nemo_tools` viết theo giao diện mình biết từ các
bản trước, nên kết quả `check_env` ở cuối script là thứ cần xem đầu tiên. Bước 4
của script in ra các tệp cấu hình FastConformer tìm thấy: dùng một tệp trong đó
làm `--base-config`.

## Tải dữ liệu

```bash
bash scripts/download_data.sh /data/scriptleak
```

| Bộ | Dung lượng tải | Cách tải |
|---|---|---|
| LibriSpeech train-clean-100, dev-clean, test-clean | khoảng 7 GB | Script, từ OpenSLR |
| MUSAN | khoảng 11 GB | Script, từ OpenSLR |
| Thai Dialect Corpus, 4 bộ trên Hugging Face | giọng trung tâm khoảng 7,5 GB | Lệnh `prepare hf` ở bước 1 |
| AISHELL-1 | khoảng 15 GB | `WITH_AISHELL=1`, cho cặp Quan Thoại–Anh |
| GigaSpeech 2, tập test tiếng Thái | chưa rõ | Phải đăng nhập Hugging Face và chấp nhận điều khoản; script in hướng dẫn |

Âm thanh tiếng Thái được ghi lại thành wav 16 kHz, khoảng 115 MB mỗi giờ, tức
khoảng 80 GB cho 700 giờ.

## Chạy nhanh bằng script theo giai đoạn

`scripts/killtest.sh` gói các bước bên dưới thành chín giai đoạn. Chạy lần lượt:

```bash
export D=/data/scriptleak
TEXT_COL=<tên cột> bash scripts/killtest.sh prepare
bash scripts/killtest.sh audit        # dừng lại đọc số giờ tiếng Thái sạch
bash scripts/killtest.sh split
bash scripts/killtest.sh mix
bash scripts/killtest.sh tokenizer
BASE_CONFIG=<tệp yaml> bash scripts/killtest.sh train05
BASE_CONFIG=<tệp yaml> bash scripts/killtest.sh train20
bash scripts/killtest.sh dump
bash scripts/killtest.sh results
```

Mặc định dùng 8 GPU, batch 32 mỗi GPU, 10 epoch, 300 giờ tiếng Thái. Đổi bằng
các biến `GPUS`, `BATCH`, `EPOCHS`, `TH_HOURS`. `INIT=none` để không khởi tạo từ
checkpoint tiếng Anh. Script này chưa được chạy thử.

## Phép thử hai tuần, từng bước

Ký hiệu: `$D` là thư mục dữ liệu, `$NEMO` là `third_party/NeMo`.

### 1. Chuyển về manifest thô

```bash
python -m scriptleak.prepare librispeech --root $D/LibriSpeech \
    --subsets train-clean-100 --out en_train.raw.jsonl
python -m scriptleak.prepare librispeech --root $D/LibriSpeech \
    --subsets dev-clean --out en_dev.raw.jsonl
python -m scriptleak.prepare librispeech --root $D/LibriSpeech \
    --subsets test-clean --out en_test.raw.jsonl

# Xem tên cột trước, rồi thay TEXT_COL bằng cột chứa câu đúng như được nói
python -m scriptleak.prepare hf-columns --dataset CMKL/Porjai-Thai-voice-dataset-korat
for name in central khummuang korat pattani; do
  python -m scriptleak.prepare hf --dataset CMKL/Porjai-Thai-voice-dataset-$name \
      --split train --text-col TEXT_COL \
      --audio-out $D/porjai/$name/wav --out th_$name.raw.jsonl
done
```

Ba bộ phương ngữ có hai cột văn bản: câu phương ngữ và bản dịch sang tiếng Thái
chuẩn. Người đọc nói câu phương ngữ, nên phải lấy cột đó. Mỗi bộ chỉ có một
split tên `train`.

### 2. Kiểm tra và lọc về dạng nói

```bash
python -m scriptleak.audit --manifest th_central.raw.jsonl --lang th \
    --out-manifest th_central.clean.jsonl --report th_central.audit.json
python -m scriptleak.audit --manifest en_train.raw.jsonl --lang en --out-manifest en_train.clean.jsonl
python -m scriptleak.audit --manifest en_dev.raw.jsonl   --lang en --out-manifest en_dev.clean.jsonl
python -m scriptleak.audit --manifest en_test.raw.jsonl  --lang en --out-manifest en_test.clean.jsonl
for name in khummuang korat pattani; do
  python -m scriptleak.audit --manifest th_$name.raw.jsonl --lang th \
      --out-manifest th_$name.clean.jsonl --set stratum=$name
done
```

Với tiếng Thái, khoảng trắng giữa các từ bị xóa. Lý do: bộ dữ liệu tách từ
không nhất quán, cùng một câu có chỗ viết liền có chỗ viết rời.

**Điểm dừng:** xem số giờ sạch của tiếng Thái trong báo cáo. Cần ít nhất 320
giờ (300 huấn luyện, 20 cho dev và test).

### 3. Chia tập và tạo tầng độ khó

```bash
python -m scriptleak.split --manifest th_central.clean.jsonl --dev-hours 3 --test-hours 10 \
    --out-prefix th_central
python -m scriptleak.noise --manifest th_central.test.jsonl --noise-dir $D/musan/noise \
    --snrs 10 5 0 --out-dir $D/th_test_noisy --out-manifest th_test.strata.jsonl
cat th_korat.clean.jsonl th_khummuang.clean.jsonl th_pattani.clean.jsonl >> th_test.strata.jsonl
```

Nếu `hf-columns` cho thấy có cột mã người nói, thêm cột đó vào `--copy-cols` ở
bước 1 và dùng `--speaker-field` ở đây để train và test không chung người nói.

### 4. Trộn dữ liệu huấn luyện

```bash
python -m scriptleak.mix --majority th_central.train.jsonl --minority en_train.clean.jsonl \
    --majority-hours 300 --minority-fraction 0.05 --out train_th300_en05.jsonl
python -m scriptleak.mix --majority th_central.train.jsonl --minority en_train.clean.jsonl \
    --majority-hours 300 --minority-fraction 0.20 --out train_th300_en20.jsonl
```

### 5. Bộ từ vựng và huấn luyện

```bash
python -m scriptleak.nemo_tools.build_tokenizer --manifest th_central.train.jsonl \
    --out-dir tokenizers/th_bpe1024 --vocab-size 1024
python -m scriptleak.nemo_tools.build_tokenizer --manifest en_train.clean.jsonl \
    --out-dir tokenizers/en_bpe256 --vocab-size 256

python -m scriptleak.nemo_tools.train \
    --base-config $NEMO/examples/asr/conf/fastconformer/hybrid_cache_aware_streaming/fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml \
    --train-manifest train_th300_en05.jsonl \
    --val-manifest th_central.dev.jsonl en_dev.clean.jsonl \
    --tokenizer th=tokenizers/th_bpe1024 en=tokenizers/en_bpe256 \
    --init-pretrained stt_en_fastconformer_hybrid_large_streaming_multi \
    --exp-dir exp/th300_en05 --max-epochs 10 --batch-size 64 --devices 1
```

Lặp lại với `train_th300_en20.jsonl` và `--exp-dir exp/th300_en20`.

Ghi chú:
- Đường dẫn tệp cấu hình và tên checkpoint khởi tạo lấy theo trí nhớ về kho
  NeMo trước bản 3.0. Dùng danh sách tệp mà `setup_env.sh` in ra để chọn lại.
- Khởi tạo encoder từ tiếng Anh có thể tự gây thiên lệch về tiếng Anh. Khi làm
  thí nghiệm chính cần thêm một cấu hình đối chứng không khởi tạo từ tiếng Anh.
- `--batch-size 64` là mức thận trọng. Tăng dần cho tới khi gần đầy bộ nhớ GPU.

### 6. Lưu xác suất theo khung

```bash
python -m scriptleak.nemo_tools.dump_logprobs --model exp/th300_en05/final.nemo \
    --manifest th_test.strata.jsonl --out dumps/th300_en05/th_test --transducer
python -m scriptleak.nemo_tools.dump_logprobs --model exp/th300_en05/final.nemo \
    --manifest en_test.clean.jsonl --out dumps/th300_en05/en_test --transducer
```

### 7. Đọc kết quả

```bash
# Có rò không, và có tăng theo độ khó không
python -m scriptleak.sweep --dump dumps/th300_en05/th_test --majority th --minority en \
    --spoken th --group-by stratum --methods greedy --out kill_greedy.json

# Vì sao rò: kiểm định giả thuyết chia nhỏ xác suất
python -m scriptleak.analysis --dump dumps/th300_en05/th_test --majority th --minority en \
    --group-by stratum --out kill_frames.json

# Các phương pháp chặn rò và cái giá với tiếng Anh
python -m scriptleak.sweep --dump dumps/th300_en05/th_test --majority th --minority en \
    --spoken th --group-by stratum --out sweep_th.json
python -m scriptleak.sweep --dump dumps/th300_en05/en_test --majority th --minority en \
    --spoken en --out sweep_en.json
```

Đầu transducer: chấm bằng `evaluate` với `--hyp-field pred_transducer` trên
tệp `utts.jsonl` trong thư mục dump.

### Ngưỡng quyết định

| Kết quả | Quyết định |
|---|---|
| SLR-u từ 0,5% trở lên ở tầng khó nhất và gấp ít nhất 2 lần tầng `clean` | Đi tiếp |
| Thêm vào đó, dòng H1 trong `analysis` báo từ 50% trở lên | Giải mã gộp theo hệ chữ là phương pháp chính |
| Có rò nhưng dòng H1 dưới 50% | Chuyển trọng tâm sang đầu phụ dự đoán hệ chữ và khâu dữ liệu |
| `lock` cho SLR-u gần 0 và `lock:k=6` cũng vậy | Cách đơn giản đã đủ, đóng góp phương pháp mỏng |
| SLR-u dưới 0,1% ở mọi tầng | Thử tăng tiếng Anh lên 50%, giảm tiếng Thái còn 100 giờ, rồi mới kết luận |

## Các phương pháp giải mã

| Tên trong `--methods` | Ý nghĩa |
|---|---|
| `greedy` | Mô hình nguyên trạng |
| `mask` | Cấm hẳn ngôn ngữ thiểu số |
| `bias:b=4` | Trừ một hằng số vào điểm của token thiểu số |
| `lock` | Quyết định hệ chữ một lần cho cả câu (không streaming) |
| `lock:k=25` | Quyết định từ 25 khung đầu, phần trước đó không khóa |
| `marg` | Đề xuất, dạng không nhớ: hệ chữ có tổng xác suất lớn hơn thắng ở mỗi khung |
| `marg:tau=1,rho=0.9` | Đề xuất, có quán tính: tích lũy bằng chứng, chỉ đổi hệ chữ khi vượt ngưỡng |

## Chưa có trong v4

- Giải mã gộp theo hệ chữ cho đầu TDT/transducer (hiện chỉ có cho CTC).
- Đầu phụ dự đoán hệ chữ lúc huấn luyện.
- Baseline nhận diện ngôn ngữ bằng mô hình ngoài và trọng số loss theo hệ chữ.
- Tầng "câu rất ngắn".
- Cặp Quan Thoại–Anh: bộ chuyển AISHELL-1 đã có, chưa có hướng dẫn chạy.
