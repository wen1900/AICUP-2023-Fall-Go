# AICUP-2023-Fall-Go

## 環境建置
### 作業系統及 CUDA 配置
* 作業系統為 win10
* CUDA 11.8：https://developer.nvidia.com/cuda-11-8-0-download-archive
* cuDNN 請選擇支援 CUDA 11.x：https://developer.nvidia.com/rdp/cudnn-archive
* python 版本：3.9.18
### 使用 Anaconda 建置環境
先創建一個 python3.9 的 anaconda環境
```
conda create --name AICUP python=3.9
conda acitvate AICUP
```
接著 install 資料夾中的 requirements.txt
```
pip install -r requirements.txt
```
這樣環境就安裝完成了

## 資料
### 訓練資料
直接使用官方提供的資料集
* **CSVs/dan_train.csv**: 段位棋手的訓練資料
* **CSVs/kyu_train.csv**: 級位棋手的訓練資料
* **CSVs/play_style_train.csv**: 棋風辨識的訓練資料
### 驗證資料
驗證資料則是在程式碼中分割訓練資料集  

Dan_Training.py
```
games_train, games_val = train_test_split(games, test_size=0.1, random_state=seed)
```
Kyu_Training.py
```
games_train, games_val = train_test_split(games, test_size=0.1, random_state=seed)
```
PlayStyle_Training.py
```
x_train, x_val, y_train, y_val = train_test_split(x, y_hot.numpy(), test_size=0.30)
```
### 測試資料
#### Public 資料集
*  **CSVs/dan_test_public.csv**: 段位棋手的 public 測試資料
* **CSVs/kyu_test_public.csv**: 級位棋手的 public 測試資料
* **CSVs/play_style_test_public.csv**: 棋風辨識的 public 測試資料
#### Private 資料集
* **CSVs/dan_test_public.csv**: 段位棋手的 private 測試資料
* **CSVs/kyu_test_public.csv**: 級位棋手的 private 測試資料
* **CSVs/play_style_test_public.csv**: 棋風辨識的 private 測試資料

## 訓練
本次訓練分為棋力模仿(段位與級位棋手)及棋風辨識
### 棋力模仿
* **段位訓練**：`Dan_Training.py`  
* **級位訓練**：`Kyu_Training.py`
### 棋風辨識
*  **棋風辨識**：`PlayStyle_Training.py`  

訓練後會將權重儲存在`/weight`中

## 預測
最後分別將3個權重進行預測並輸出成csv的上傳格式  
* **預測程式**：`Create_Upload_CSV.py`  

最終結果儲存在`/submit`中

## 函式介紹
### 資料處理
棋盤的訓練需要自行產生feature訓練，feature的好壞會直接影響訓練結果，以下為資料處理的函式。

|  函式 |功能     |備註|
|:-----:|--------|-----|
|`prepare_input` |將訓練資料集的每場棋盤處理後，產生feature已進行訓練|
|`prepare_label` |將訓練資料集的每場棋盤處理後，產生答案|此函式只用於棋力模仿|

### Feature map
分別介紹棋力模仿及棋風辨識的feature map
#### 棋力模仿
* 將黑子設為1其餘為0
* 將白子設為1其餘為0
* 將空位設為1其餘為0
* 將最後一手設為1其餘為0
* 將倒數第二手設為1其餘為0
* 將倒數第三手設為1其餘為0
* 將倒數第四手設為1其餘為0  

總共7張feature map

#### 棋風辨識
* 將盤面所有子設為1其餘為0
* 將最後一手設為1其餘為0
* 將倒數第二手設為1其餘為0
* 將倒數第三手設為1其餘為0
* 將倒數第四手設為1其餘為0  
* 將倒數第四手以前的每個子設為1其餘為0
* 將最後四手設為1其餘為0
* 將與最後一手同顏色的子設為1其餘為0

總共8張feature map


### Data Generator
由於棋力模仿的資料較多，一次性輸入會造成記憶體不足，因此利用 Data Generator 將資料輸入模型。

先用此行程式碼切出訓練與驗證資料集
```
games_train, games_val = train_test_split(games, test_size=0.1, random_state=seed)
```
接著再用 data_generator 分別將兩個資料集輸入
|  函式 |功能     |備註|
|:-----:|--------|-----|
|`data_generator` |將訓練資料分批輸入|此函式只用於棋力模仿|

### 模型
在棋力模仿及棋風辨識分別使用不同模型訓練  

#### 棋力模仿
|  函式 |結構     |
|:-----:|--------|
|`go_res` |為resnet架構，共使用6層residual_block組成|
#### 棋風辨識
|  函式 |結構     |
|:-----:|--------|
|`create_model` |為DCNN架構|

### 預測
|  函式 |結構     |備註|
|:-----:|--------|-----|
|`prepare_input_for_dan_kyu_models` |產生棋力模仿的feature以進行預測|
|`prepare_input_for_playstyle_model` |產生棋風辨識的feature以進行預測|
|`top_5_preds_with_chars` |根據預測值由大到小產生top 5的下一手預測|此函式只用於棋力模仿|
|`number_to_char` |將`top_5_preds_with_chars`產生的預測轉換為棋盤位置|此函式只用於棋力模仿|

預測時會同時預測 public 與 private的結果，並寫入`/submit/pri_sub_.csv`，由於 csv 是用 append 的方式寫入，如需寫入新的結果須更改`pri_sub_.csv`檔名，避免重複寫入。

## 最終結果
本次訓練在段位及級位棋力模仿中各訓練7次，棋風辨識則是訓練100次，在`/final`中分別附上棋力模仿訓練6次及訓練7次的權重，以及棋風辨識的最終權重，並附上最後成績。

|  epochs |public score|private score|
|:-----:|:--------:|:-----:|
|6 |0.645986|0.642947|
|7 |0.646671|0.644601|

以及最後一次的public細項成績
|Ten_Kyu_1|Ten_Kyu_5|One_Dan_1|One_Dan_5|PSA|
|:-----:|:--------:|:-----:|:-----:|:-----:|
|0.534744|0.820988|0.497182|0.8|0.755302|
