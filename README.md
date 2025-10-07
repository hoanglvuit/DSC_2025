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

Chạy inference mà **không reproduce quá trình translate**:

``` bash
chmod +x ./inference.sh
TRANSLATE=false ./inference.sh
```

### 3. Reproduce cả **translate**:

``` bash
chmod +x ./inference.sh
TRANSLATE=true ./inference.sh
```

Kết quả sẽ được trả về submit.csv

------------------------------------------------------------------------
