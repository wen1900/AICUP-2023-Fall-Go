# AICUP-2023-Fall-Go

## 環境建置
### 作業系統及 CUDA 配置
* 作業系統為win10
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
#### 直接使用官方提供的資料集
* **CSVs/dan_train.csv**: 段位棋手的訓練資料
* **CSVs/kyu_train.csv**: 級位棋手的訓練資料
* **CSVs/play_style_train.csv**: 棋風辨識的訓練資料
### 驗證資料
#### 驗證資料則是在程式碼中分割訓練資料集
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
* **CSVs/dan_test_public.csv**: 段位棋手的 public 測試資料
* **CSVs/kyu_test_public.csv**: 級位棋手的 public 測試資料
* **CSVs/play_style_test_public.csv**: 棋風辨識的 public 測試資料
#### Private 資料集
* **CSVs/dan_test_public.csv**: 段位棋手的 private 測試資料
* **CSVs/kyu_test_public.csv**: 級位棋手的 private 測試資料
* **CSVs/play_style_test_public.csv**: 棋風辨識的 private 測試資料

## 訓練
#### 本次訓練分為棋力模仿(段位與級位棋手)及棋風辨識
### 棋力模仿
* **段位訓練**：`Dan_Training.py`  
* **級位訓練**：`Kyu_Training.py`
### 棋風辨識
* **棋風辨識**：`PlayStyle_Training.py`  
#### 訓練後會將權重儲存在`/weight`中

## 預測
#### 最後分別將3個權重進行預測並輸出成csv格式的結果  
* **預測程式**：`Create_Upload_CSV.py`  

#### 最終結果儲存在`/submit`中

## 函式介紹
