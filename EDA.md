# EDA

## stock_id 毎の統計量
### stock_id 毎のボラティリティの合計のヒストグラム
![stock0_time5_target_sum](https://user-images.githubusercontent.com/79825066/129476861-a67c2b04-1e04-4dde-9aa8-9e1eb219d0a2.png)

##### stock_id 毎にボラティリティの合計を見ると，10~14.99にあるstock_idにあるstock_id が最も多い．<br>この期間のtotalボラティリティは，平均14.8，最大だと30を超える銘柄も存在する．

### stock_id 毎のボラティリティの平均値のヒストグラム
![stock0_time5_target_mean](https://user-images.githubusercontent.com/79825066/129477766-a941f42b-5f65-4c2b-888d-bf3c035edf26.png)
##### sotck_id 毎にボラティリティの平均値を見ると，0.003~0.00399にあるstock_id が最も多い．<br>平均値はほぼ0に近い


## 価格変動
#### stock_idが0，time_idが5のオーダーブックに関して
##### time_id(10分間)内での，ask_price1，ask_price2，bid_price1，bid_price2の変動
![stock0_time5_price](https://user-images.githubusercontent.com/79825066/129478908-dcba3000-f38a-47cc-896e-1abce65fe15e.png)

## 実際の取引価格
![stock0_time5_trade](https://user-images.githubusercontent.com/79825066/129478943-86bb4e20-0906-4794-a185-1a0d7948fde7.png)
##### オーダーブックのbidとaskの間を上下している．

##### 価格変動が大きい場合，ボラティリティの値も大きくなる？

## ボラティリティと価格変動の関係性
#### stock_id = 0 で最もボラティリティが低かった time_id = 24253 と最も高かった time_id = 19725 の時の価格変動を比較する．
![price_diff_max_min](https://user-images.githubusercontent.com/79825066/129479055-f91c4c60-5164-4480-b324-83b46a85ce13.png)

##### ボラティリティが高い場合，10分間で大きく価格が変動している．