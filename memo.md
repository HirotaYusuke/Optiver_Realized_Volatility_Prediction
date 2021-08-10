# Dataについて

## book_[train/test].parquet:
- 株の銘柄毎に最も競争力のある上位2つの売買注文のデータがはいっている。
- Data parquetファイルの読み込み方法: parquet file  

## stock_id 
- 株の銘柄 (どの株か) 銘柄は0から126まである。
- 読み込んだ後、カテゴリカルデータになってしまうので、int型に変えたほうがいいかも

## time_id 
- 取引の時間 (submissionファイルのtime_idと連動している)
- time_idは、連続値ではないが、全ての銘柄で統一されている。

## Seconds_in_bucket
- time_idの中で，バケットが始まって0からスタートして何秒後か．おそらく予測するのは10分後なのSecconds_in_bucketは最大600．

## bid_price[1/2] 
- 株の買値の希望値の1番目と2番目．1番目と2番目に競争力のある買値．
- Normalized prices of the most/second most competitive buy level.

## ask_price[1/2] 
- 株の売値の希望値の1番目と2番目．1番目と2番目に競争力のある売値．
- Normalized prices of the most/second most competitive sell level.

## bid_size[1/2] 
- 買うのを希望している側の1番目と2番目に競争力のある株式の数
- The number of shares on the most/second most competitive buy level.

## ask_size[1/2] 
- 売るのを希望している側の1番目と2番目に競争力のある株式の数
- The number of shares on the most/second most competitive sell level.

## trade_[train/test].parquet: 
- 実際に行われた売買のデータが入っている。parquet fileの読み込み方法： parquet file 

## stock_id 
- 同上．
- Same as above.

## time_id 
- 同上．
- Same as above.

## seconds_in_bucket 
- time_idの中で，バケットが始まって0からスタートして何後か．
- Same as above. Note that since trade and book data are taken from the same time window and trade data is more sparse in general, this field is not necessarily starting from 0.

## price 
- 1秒間で行われた取引の平均価格
- 価格は正規化され、価格の平均はそれぞれのトランザクションで取引された株式の数で重み付される。
- The average price of executed transactions happening in one second. Prices have been normalized and the average has been weighted by the number of shares traded in each transaction.
　
## size 
- 取引された株式の合計
- The sum number of shares traded.

## order_count 
- 固有の取引の数
- The number of unique trade orders taking place.

# train.csv The ground truth values for the training set.

## stock_id 
- 株の銘柄 (どの株か) 銘柄は0から126まである。

## time_id 
- 取引の時間 (submissionファイルのtime_idと連動している)
- time_idは、連続値ではないが、全ての銘柄で統一されている。

## target 
- 株式の変動性
- 同じstock_id・time_idの株に対して、10分単位で計算される。
- tutorial notebook.
- The realized volatility computed over the 10 minute window following the feature data under the same stock/time_id. There is no overlap between feature and target data. You can find more info in our 

# test.csv
### テストデータは3つ
<br>
Provides the mapping between the other data files and the submission file. As with other test files, most of the data is only available to your notebook upon submission with just the first few rows available for download.

## stock_id 
- Same as above.

## time_id
- Same as above.

## row_id 
- Submission.csvでの行番号。

## stock_id 
- 例: stock_id: 0, time_id:4 ⇒ 0-4

# sample_submission.csv 
### A sample submission file in the correct format.

## row_id 
- Submission.csvでの行番号。

## stock_id
- 例: stock_id: 0, time_id:4 ⇒ 0-4
- Same as in test.csv.

## target 
- 株の変動性
- Same definition as in train.csv. The benchmark is using the median target value from train.csv.

#### book_[train/test].parquet, trade_[train/test].parquet, train.csvを用いてモデルを作成，学習を行う．最終的にtrain.csvの予測結果を提出する．

## tutorial notebookのmemo
- 取引は、売りての言い値(ask price)と買い手の掛け値(bid price)が同じときでないと発生しない．
- 株式市場では、流動性が大事

# Order bookの統計量
#### 基本的な市場予測アルゴリズム ・ 一般的な統計値

## bid/ask spreadの計算方法
#### 取引の最も高い買値（最良買い気配値）と最も安い売値（最良売り気配値）の差をいう.この差が小さいほど、売買にかかる取引費用が小さい．
<br>

#### また、売買にかかる取引費用が小さい市場ほど、流動性が高いといえる．
<br>

#### 市場では様々な銘柄が異なるレベルで取引させれているため，bid/ask spread を計算するには，Best Offer と Best bid の価格の比率を取る．
<br>

![image](https://user-images.githubusercontent.com/79825066/128835343-aec665b1-8423-4e15-bb30-f7a5e566132f.png)

- Best offer, best bidの求め方
- 同じstock_id,time_id, seconds_in_bucketである場合のbid_price1 と ask_price1

## Weighted average priceの計算方法 (VWAPと同じ)
#### 今回のコンペティションでは，WAP(加重平均価格)を用いて瞬間的な株価平均とRealized Volatility(実現ボラティリティ)を算出する．
<br>

#### 売買代金を出来高で割ったものでその日の平均売買価格を表す．
<br>

![image](https://user-images.githubusercontent.com/79825066/128835244-b0708926-da3f-4d2b-93d4-ef519fe77f2d.png)

#### ある銘柄を売買したすべての投資家の平均購入値段に当たるため，株価がWAPの上にあるか，下にあるかでその日の買った人の損益がプラスかマイナスかを確認できる

## Log returns
#### その日とその前日の株価の比較を行う方法．その日の株価を前日の株価で割る．
<br>

#### リターンは金融の分野で広く使われるが，数学的なモデリングが必要な場合，Log returns が好まれる．
![image](https://user-images.githubusercontent.com/79825066/128835103-502fd301-cda9-4b0a-94fb-2d59fc7b8e84.png)

- Stはとある瞬間 t の株価を表している．
- 上の定義はt₁とt₂間の株価の差

## Realized volatility
#### オプション取引を行う際，株式のログリターンの標準偏差がモデルに入れる重要な説明変数となる．通常は1年単位で標準化し，年単位の標準偏差をボラティリティという．
<br>

#### 以下が予測するボラティリティを求める式．今回は10分間のブックデータが与えられ，その後10分後のボラティリティを求める．

![image](https://user-images.githubusercontent.com/79825066/128838854-759cb753-e909-4a1b-888d-da10dec1641b.png)

- 今回，全ての連続したブックのログリターンを計算し，それを実現ボラティリティ σ と定義する．
- σ：ログリターンの二乗の総和の平方根

#### ここでは，WAPを株式の価格とし，ログリターンを計算している．