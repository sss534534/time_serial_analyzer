# 时序数据分析模块

## 功能简介

该模块提供了一个全面的时序数据分析工具 `TimeSeriesAnalyzer`，支持以下功能：

- 数据加载和预处理
- 异常值检测
- 统计分析
- 平稳性检验
- 时序分解
- 可视化（趋势图、分解图、自相关图等）
- 特征提取（移动平均、差分、滞后特征等）
- 预测模型（ARIMA、LSTM、移动平均）
- 模型评估和比较
- 结果保存和加载

## 安装依赖

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

如果需要使用LSTM模型，还需要安装TensorFlow：

```bash
pip install tensorflow
```

## 快速开始

### 1. 基本使用流程

```python
from time_series_analyzer import TimeSeriesAnalyzer

# 1. 初始化分析器
analyzer = TimeSeriesAnalyzer()

# 2. 加载数据
df = analyzer.load_data('sample_data.csv', date_column='date', value_column='value')

# 3. 预处理数据
df = analyzer.preprocess_data(df)

# 4. 异常值检测
outliers = analyzer.detect_outliers(df, method='zscore')
print(f'检测到 {len(outliers)} 个异常值')

# 5. 统计分析
stats = analyzer.descriptive_statistics(df)
print(stats)

# 6. 平稳性检验
stationarity = analyzer.test_stationarity(df)
print(f'平稳性检验结果: {stationarity}')

# 7. 时序分解
decomposition = analyzer.decompose_time_series(df)
```

### 2. 可视化

```python
# 绘制时序趋势图
analyzer.plot_time_series(df)

# 绘制分解图
analyzer.plot_decomposition(df)

# 绘制自相关和偏自相关图
analyzer.plot_acf_pacf(df)

# 绘制箱线图
analyzer.plot_boxplot(df)

# 绘制热力图
analyzer.plot_heatmap(df)
```

### 3. 特征提取

```python
# 添加移动平均特征
df = analyzer.add_moving_average(df, windows=[7, 30])

# 添加差分特征
df = analyzer.add_differencing(df, lags=[1, 2])

# 添加滞后特征
df = analyzer.add_lag_features(df, lags=[1, 3, 7])

# 添加滚动统计特征
df = analyzer.add_rolling_stats(df, window=30, stats=['mean', 'std', 'min', 'max'])

# 添加时间特征
df = analyzer.add_time_features(df)

# 添加季节性特征
df = analyzer.add_seasonal_features(df)
```

### 4. 预测模型

```python
# 划分训练测试集
train, test = analyzer.train_test_split(df, test_size=0.2)

# 移动平均预测
ma_preds, ma_metrics = analyzer.moving_average_forecast(train, test, window=7)
print('移动平均预测指标:', ma_metrics)

# ARIMA预测
arima_preds, arima_metrics = analyzer.arima_forecast(train, test, order=(1,1,1))
print('ARIMA预测指标:', arima_metrics)

# LSTM预测（需要安装TensorFlow）
try:
    lstm_preds, lstm_metrics = analyzer.lstm_forecast(train, test, epochs=50, batch_size=32)
    print('LSTM预测指标:', lstm_metrics)
except ImportError:
    print('需要安装TensorFlow才能使用LSTM模型')
```

### 5. 模型评估和比较

```python
# 比较模型性能
analyzer.compare_models()

# 保存评估结果
analyzer.save_evaluation_results('evaluation_results.json')
```

### 6. 保存和加载

```python
# 保存数据
analyzer.save_data(df, 'processed_data.csv')

# 保存模型
analyzer.save_model('arima', 'arima_model.pkl')

# 加载模型
loaded_model = analyzer.load_model('arima_model.pkl')
```

## API参考

### TimeSeriesAnalyzer类

#### 初始化
```python
TimeSeriesAnalyzer(data=None, date_column=None, value_column=None)
```

- `data`: 可选的输入数据DataFrame
- `date_column`: 日期列名称
- `value_column`: 值列名称

#### 数据处理方法

- `load_data(file_path, date_column, value_column)`: 从CSV文件加载数据
- `preprocess_data(df, fill_method='ffill', freq=None)`: 预处理数据（处理缺失值、设置日期索引）
- `detect_outliers(df, method='zscore', threshold=3)`: 检测异常值（支持zscore和iqr方法）
- `descriptive_statistics(df)`: 计算描述性统计
- `test_stationarity(df)`: 进行ADF平稳性检验
- `decompose_time_series(df, model='additive')`: 分解时序数据（趋势、季节性、残差）

#### 可视化方法

- `plot_time_series(df, title='Time Series Plot')`: 绘制时序趋势图
- `plot_decomposition(df, model='additive', title='Time Series Decomposition')`: 绘制分解图
- `plot_acf_pacf(df, lags=30, title='ACF and PACF Plots')`: 绘制自相关和偏自相关图
- `plot_boxplot(df, by=None, title='Box Plot')`: 绘制箱线图
- `plot_heatmap(df, lag=14, title='Correlation Heatmap')`: 绘制热力图

#### 特征提取方法

- `add_moving_average(df, windows=[7, 30])`: 添加移动平均特征
- `add_differencing(df, lags=[1])`: 添加差分特征
- `add_lag_features(df, lags=[1, 3, 7])`: 添加滞后特征
- `add_rolling_stats(df, window=30, stats=['mean', 'std', 'min', 'max'])`: 添加滚动统计特征
- `add_time_features(df)`: 添加时间特征（年、月、日、星期等）
- `add_seasonal_features(df)`: 添加季节性特征（正弦和余弦变换）

#### 预测模型方法

- `train_test_split(df, test_size=0.2)`: 划分训练测试集
- `moving_average_forecast(train, test, window=7)`: 移动平均预测
- `arima_forecast(train, test, order=(1,1,1))`: ARIMA预测
- `lstm_forecast(train, test, epochs=50, batch_size=32)`: LSTM预测

#### 评估和保存方法

- `calculate_metrics(y_true, y_pred)`: 计算评估指标
- `save_data(df, file_path)`: 保存数据到CSV文件
- `save_model(model_name, file_path)`: 保存模型
- `load_model(file_path)`: 加载模型
- `save_evaluation_results(file_path)`: 保存评估结果
- `compare_models()`: 比较所有模型的性能

## 示例数据

模块包含 `sample_data.csv` 文件，包含2020年全年的时序数据，可用于测试各种功能。

## 注意事项

1. 使用LSTM模型需要安装TensorFlow
2. 大数据集可能需要较长的处理时间
3. 可视化功能在非图形界面环境下可能需要调整
4. 模型参数可能需要根据具体数据进行调优

## 扩展建议

1. 添加更多的预测模型（如Prophet、XGBoost等）
2. 实现自动化特征选择
3. 添加超参数优化功能
4. 支持多变量时序分析
5. 实现实时数据处理功能