# UIT_Champion - DSC 2025 Track B: Phân loại Hallucination

Đây là source code của đội **UIT_Champion** tham dự cuộc thi **DSC
2025 - Track B - Phân loại Hallucination**.

------------------------------------------------------------------------

## Cấu hình huấn luyện

-   **Seed**: `22520465` (được cố định mặc định trong mọi file).\
-   **Hyperparameters**: định nghĩa trong file `train.sh`.\
-   **Checkpoint**: lưu tại thư mục `results/`.\
-   **Predictions cho stack ensemble (XGBoost)**: lưu tại thư mục
    `output/` hoặc `output_true/`.

------------------------------------------------------------------------

## Hướng dẫn reproduce kết quả

### 1. Chạy stack ensemble (XGBoost)

Để reproduce **100% kết quả cuối cùng** (giả sử đã có sẵn predictions từ các base models):

``` bash
python stack_ensemble.py
```

### 2. Reproduce toàn bộ quá trình inference 
- Ở đây đã có sẵn checkpoint, nhưng nhóm gặp vấn đề với 1 model dangvantuan, 5 models còn lại reproduce kết quả đã giống nhưng ở model cuối (dangvantuan) thì vẫn chưa thể reproduce lại được y hệt -> làm quá trình kết quả submit.csv lệch 10/2000 câu so với python stack_ensemble.py. 
BTC có thể xóa output của dvt và thay bằng output ban đầu trong output_true để có thể reproduce chính xác. 
Hoặc đây là notebook nhóm đã finetune lúc thi private cho model dangvantuan: https://www.kaggle.com/code/lvanhoang/optimize-dangvantuan/notebook?scriptVersionId=265454489. BTC có thể download output và bỏ vào output\dangvantuan_vietnamese-document-embedding để có thể reproduce chính xác. Lúc thi private thì để cho nhanh nên nhóm đã tránh save model để tránh tràn disk trên kaggle.  

Cho nên để chạy inference **từ checkpoint (không reproduce dvt, không reproduce translate)** 

``` bash
chmod +x ./inference.sh
TRANSLATE=false RUN_DVT=false ./inference.sh
```

### 3. Chạy inference **từ checkpoint (không reproduce dvt,reproduce translate)** 

``` bash
chmod +x ./inference.sh
RUN_DVT=false ./inference.sh
```

### 4. Reproduce cho tất cả:

``` bash
chmod +x ./inference.sh
./inference.sh
```

Kết quả sẽ được trả về submit.csv

------------------------------------------------------------------------
