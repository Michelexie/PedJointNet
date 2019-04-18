# PedJointNet行人頭肩偵測系統
2019放視大賞競賽參賽作品
本作品的運行環境為：
硬體環境
處理器：Intel(R) Core(TM) i7-8700 CPU @ 3.20 GHz 3.19 GHz；
顯示卡：NVDIA GeForce GTX 1080Ti;RAM: 16.0 GB

開發環境：JetBrains PyCharm Community Edition 2018.2.4 x64，Python3.6，Tensorflow1.6

主要功能：利用我們設計的PedJointNet行人頭肩偵測網絡，預先對CHUK-SYSU, TownCentre, CityPersons等行人數據集訓練得到权重，可以對輸入影像中的行人頭肩與全身進行偵測，輸出預測影像。

操作步驟:
1. run main.py
會出現如下主界面
![image](https://github.com/Michelexie/PedJointNet/blob/master/main_1.png)

2. click【輸入影像】按鍵，從本地選擇一張待偵測行人圖片，該圖片會顯示在界面左下方
![image](https://github.com/Michelexie/PedJointNet/blob/master/main_2.png)

3. click【一鍵偵測】按鍵，將進行行人頭肩與全身偵測，并將偵測結果顯示在界面右下方
![image](https://github.com/Michelexie/PedJointNet/blob/master/main_3.png)

圖中綠色框代表偵測出的頭肩部位，藍色框代表偵測出的全身部位，每個框上的數字代表偵測的分數。
