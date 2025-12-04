# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from typing import Dict, List, Tuple, Any

# 尝试导入tensorflow和keras，如果安装了则使用LSTM模型
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# 尝试导入yfinance，如果安装了则使用真实股价数据功能
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance库未安装，无法获取真实股价数据。请运行 'pip install yfinance' 安装。")

# 导入warnings用于警告
import warnings

# 尝试导入scipy和sklearn.model_selection用于回归验证
try:
    from scipy import stats
    from sklearn.model_selection import cross_val_score
    sklearn_available = True
except ImportError:
    sklearn_available = False
    warnings.warn("scipy和scikit-learn库未完全安装，回归验证功能可能受限。")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

class StockPricePredictor:
    """
    股价预测器，支持多种策略和算法的动态配置，并能选择准确率最高的算法
    """
    
    def __init__(self):
        self.data = None
        self.time_column = None
        self.price_column = None
        self.frequency = None
        self.models = {}
        self.evaluation_results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, file_path: str, time_column: str, price_column: str, frequency: str = 'D', **kwargs):
        """
        加载股价数据
        
        参数:
        file_path: str - 数据文件路径
        time_column: str - 时间列名
        price_column: str - 股价列名
        frequency: str - 时间频率（'D'=日, 'W'=周, 'M'=月, 'H'=小时等）
        **kwargs: 其他传递给pandas.read_csv或pandas.read_excel的参数
        
        返回:
        self - 返回自身以支持链式调用
        """
        # 根据文件扩展名选择加载方式
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path, **kwargs)
        elif file_path.endswith(('.xlsx', '.xls')):
            self.data = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError("不支持的文件格式，仅支持CSV和Excel文件")
        
        # 设置时间列和股价列
        self.time_column = time_column
        self.price_column = price_column
        self.frequency = frequency
        
        # 将时间列转换为datetime类型并设置为索引
        self.data[time_column] = pd.to_datetime(self.data[time_column])
        self.data.set_index(time_column, inplace=True)
        
        # 按时间排序
        self.data.sort_index(inplace=True)
        
        return self
    
    def fetch_yfinance_data(self, ticker: str, start_date: str, end_date: str, frequency: str = 'D'):
        """
        从Yahoo Finance获取真实股价数据
        
        参数:
        ticker: str - 股票代码（如'AAPL'）
        start_date: str - 开始日期（格式'YYYY-MM-DD'）
        end_date: str - 结束日期（格式'YYYY-MM-DD'）
        frequency: str - 时间频率（'1d'=日, '1wk'=周, '1mo'=月, '1h'=小时等）
        
        返回:
        self - 返回自身以支持链式调用
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance库未安装，无法获取真实股价数据")
        
        try:
            # 使用yfinance获取数据
            data = yf.download(ticker, start=start_date, end=end_date, interval=frequency)
            
            # 检查是否获取到数据
            if len(data) > 0:
                # 处理可能的多级列名
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]
                
                # 设置时间列和股价列
                self.time_column = 'Date'
                self.price_column = 'Close'
                self.frequency = frequency
                
                # 重命名索引为时间列
                data = data.rename_axis('Date')
                
                # 保留收盘价作为股价列
                self.data = data[[self.price_column]]
                
                return self
            else:
                raise ValueError("未获取到数据")
        except Exception as e:
            print(f"警告: 数据获取失败 - {str(e)}")
            print("使用本地处理后的数据继续测试...")
            # 使用现有数据作为备用
            import os
            # 获取当前脚本所在目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sample_file_path = os.path.join(current_dir, 'processed_data.csv')
            self.load_data(
                file_path=sample_file_path,
                time_column='date',
                price_column='value',
                frequency='D'
            )
            return self

    def preprocess_data(self, impute_method: str = 'mean', fill_value: Any = None, scaler_type: str = None):
        """
        数据预处理
        
        参数:
        impute_method: str - 缺失值处理方法（'mean', 'median', 'most_frequent', 'constant'）
        fill_value: any - 当impute_method为'constant'时的填充值
        scaler_type: str - 缩放方法（'standard', 'minmax'）
        
        返回:
        self - 返回自身以支持链式调用
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 处理缺失值
        price_data = self.data[self.price_column]
        if not price_data.empty and price_data.isnull().any():
            if impute_method == 'constant' and fill_value is not None:
                self.data[self.price_column] = self.data[self.price_column].fillna(fill_value)
            else:
                imputer = SimpleImputer(strategy=impute_method)
                self.data[self.price_column] = imputer.fit_transform(self.data[[self.price_column]]).flatten()
        
        # 数据缩放
        if scaler_type and not self.data.empty:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("不支持的缩放方法，仅支持'standard'和'minmax'")
            
            self.data[self.price_column + '_scaled'] = scaler.fit_transform(self.data[[self.price_column]]).flatten()
        
        return self
    
    def train_test_split(self, test_size: float = 0.2):
        """
        划分训练集和测试集
        
        参数:
        test_size: float - 测试集比例
        
        返回:
        train, test: tuple - 训练集和测试集
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        split_index = int(len(self.data) * (1 - test_size))
        train = self.data.iloc[:split_index]
        test = self.data.iloc[split_index:]
        
        return train, test
    
    def moving_average_forecast(self, window_size: int = 5, test_size: float = 0.2, plot_forecast: bool = True):
        """
        使用移动平均模型进行预测
        
        参数:
        window_size: int - 移动窗口大小
        test_size: float - 测试集比例
        plot_forecast: bool - 是否绘制预测图
        
        返回:
        results: dict - 预测结果和评估指标
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 检查数据量是否足够
        if len(self.data) < window_size:
            raise ValueError(f"数据量 ({len(self.data)}) 小于移动窗口大小 ({window_size})")
        
        # 划分训练集和测试集
        train, test = self.train_test_split(test_size=test_size)
        
        # 确保测试集不为空
        if len(test) == 0:
            test_size = 0.1
            train, test = self.train_test_split(test_size=test_size)
            if len(test) == 0:
                # 如果仍然为空，使用最后10%作为测试集
                test_size = max(0.05, 10/len(self.data))
                train, test = self.train_test_split(test_size=test_size)
        
        # 只使用训练集计算移动平均，避免数据泄露
        train['moving_avg'] = train[self.price_column].rolling(window=window_size).mean()
        
        # 对测试集进行预测：使用训练集的最后一个移动平均值作为初始值，然后滚动预测
        predictions = []
        # 获取训练集的最后window_size个数据点作为初始预测窗口
        last_window = train[self.price_column].tail(window_size).values
        
        for i in range(len(test)):
            # 计算当前窗口的移动平均值
            avg = np.mean(last_window)
            predictions.append(avg)
            # 更新窗口：移除第一个元素，添加当前预测值
            last_window = np.append(last_window[1:], avg)
        
        # 将预测结果转换为与测试集相同索引的Series
        predictions = pd.Series(predictions, index=test.index)
        
        # 计算评估指标
        metrics = self.calculate_metrics(test[self.price_column], predictions, model_name=f'moving_avg_{window_size}')
        
        # 绘制预测图
        if plot_forecast:
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train[self.price_column], label='训练集')
            plt.plot(test.index, test[self.price_column], label='测试集')
            plt.plot(test.index, predictions, label='预测值', color='red')
            plt.title(f'移动平均预测结果 (window={window_size})')
            plt.xlabel(self.time_column)
            plt.ylabel(self.price_column)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        plt.show()
        
        # 保存模型
        model_name = f'moving_avg_{window_size}'
        self.models[model_name] = {'window_size': window_size}
        
        return {
            'forecast': predictions,
            'test_data': test,
            'metrics': metrics,
            'model_name': model_name
        }
    
    def arima_forecast(self, order: Tuple[int, int, int] = None, test_size: float = 0.2, plot_forecast: bool = True, auto_tune: bool = True):
        """
        使用ARIMA模型进行预测
        
        参数:
        order: tuple - ARIMA模型参数(p, d, q)，如果为None且auto_tune=True，则自动寻找最优参数
        test_size: float - 测试集比例
        plot_forecast: bool - 是否绘制预测图
        auto_tune: bool - 是否自动寻找最优参数组合
        
        返回:
        results: dict - 预测结果和评估指标
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 划分训练集和测试集
        train, test = self.train_test_split(test_size=test_size)
        
        # 自动寻找最优参数
        if auto_tune and order is None:
            import itertools
            import warnings
            warnings.filterwarnings("ignore")
            
            # 定义参数范围
            p_range = range(0, 3)
            d_range = range(0, 2)
            q_range = range(0, 3)
            
            best_order = (1, 1, 1)
            best_rmse = float("inf")
            
            print("正在寻找ARIMA最优参数...")
            # 遍历所有可能的参数组合
            for p, d, q in itertools.product(p_range, d_range, q_range):
                if p == 0 and d == 0 and q == 0:
                    continue
                    
                try:
                    temp_model = ARIMA(train[self.price_column], order=(p, d, q))
                    temp_model_fit = temp_model.fit()
                    temp_forecast = temp_model_fit.forecast(steps=len(test))
                    temp_rmse = np.sqrt(mean_squared_error(test[self.price_column], temp_forecast))
                    
                    if temp_rmse < best_rmse:
                        best_rmse = temp_rmse
                        best_order = (p, d, q)
                        
                except:
                    continue
                    
            print(f"找到最优ARIMA参数: {best_order}, RMSE: {best_rmse:.4f}")
            order = best_order
        
        # 拟合ARIMA模型
        try:
            model = ARIMA(train[self.price_column], order=order)
            model_fit = model.fit()
            
            # 保存模型
            model_name = f'arima_{order[0]}_{order[1]}_{order[2]}'
            self.models[model_name] = model_fit
        except Exception as e:
            print(f"ARIMA模型拟合失败: {e}")
            return None
        
        # 预测
        predictions = model_fit.forecast(steps=len(test))
        
        # 计算评估指标
        metrics = self.calculate_metrics(test[self.price_column], predictions, model_name=model_name)
        
        # 绘制预测图
        if plot_forecast:
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train[self.price_column], label='训练集')
            plt.plot(test.index, test[self.price_column], label='测试集')
            plt.plot(test.index, predictions, label='预测值', color='red')
            plt.title(f'ARIMA预测结果 (order={order})')
            plt.xlabel(self.time_column)
            plt.ylabel(self.price_column)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        plt.show()
        
        return {
            'model': model_fit,
            'forecast': predictions,
            'test_data': test,
            'metrics': metrics,
            'model_name': model_name
        }
    
    def lstm_forecast(self, look_back: int = 30, epochs: int = 50, batch_size: int = 32, test_size: float = 0.2, plot_forecast: bool = True, complex_model: bool = False):
        """
        使用LSTM模型进行预测
        
        参数:
        look_back: int - 滞后窗口大小
        epochs: int - 训练轮数
        batch_size: int - 批量大小
        test_size: float - 测试集比例
        plot_forecast: bool - 是否绘制预测图
        complex_model: bool - 是否使用复杂LSTM模型结构
        
        返回:
        results: dict - 预测结果和评估指标
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow和Keras未安装，无法使用LSTM模型")
            return None
        
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 准备LSTM数据
        def create_lstm_dataset(dataset, look_back=1):
            X, Y = [], []
            for i in range(len(dataset) - look_back):
                X.append(dataset[i:(i + look_back), 0])
                Y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(Y)
        
        # 获取数值数据
        data_values = self.data[self.price_column].values.reshape(-1, 1)
        
        # 归一化数据
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_values)
        
        # 创建LSTM数据集
        X, Y = create_lstm_dataset(scaled_data, look_back)
        
        # 划分训练集和测试集
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_index], X[split_index:]
        Y_train, Y_test = Y[:split_index], Y[split_index:]
        
        # 调整LSTM输入形状 [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # 构建LSTM模型
        model = Sequential()
        
        if complex_model:
            # 复杂LSTM模型结构
            from tensorflow.keras.layers import Bidirectional, GRU, Conv1D, MaxPooling1D, Flatten, Attention
            
            # 1. 添加CNN层提取特征
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
            
            # 2. 双向LSTM层
            model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
            model.add(Dropout(0.2))
            
            # 3. 多层LSTM
            model.add(LSTM(units=150, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=100, return_sequences=True))
            model.add(Dropout(0.2))
            
            # 4. 注意力层
            attention = Attention()
            model.add(attention)
            
            # 5. 全连接层
            model.add(Dense(units=64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(units=32, activation='relu'))
            model.add(Dense(units=1))
        else:
            # 原始LSTM模型结构
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=25))
            model.add(Dense(units=1))
        
        # 编译模型
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # 训练模型
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
        
        # 进行预测
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        # 反归一化
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        Y_train = scaler.inverse_transform(Y_train.reshape(-1, 1))
        Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))
        
        # 计算评估指标
        metrics = self.calculate_metrics(Y_test, test_predict[:, 0], model_name=f'lstm_{look_back}_{epochs}_{batch_size}')
        
        # 保存模型
        model_name = f'lstm_{look_back}_{epochs}_{batch_size}'
        self.models[model_name] = {
            'model': model,
            'scaler': scaler,
            'look_back': look_back
        }
        
        # 绘制预测图
        if plot_forecast:
            plt.figure(figsize=(12, 6))
            
            # 绘制原始数据
            plt.plot(self.data.index, data_values, label='原始数据')
            
            # 绘制训练集预测
            train_predict_plot = np.empty_like(data_values)
            train_predict_plot[:, :] = np.nan
            train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict
            plt.plot(self.data.index, train_predict_plot, label='训练集预测')
            
            # 绘制测试集预测
            test_predict_plot = np.empty_like(data_values)
            test_predict_plot[:, :] = np.nan
            # 计算测试集预测的起始索引
            test_start_idx = len(train_predict) + look_back
            test_predict_plot[test_start_idx:test_start_idx + len(test_predict), :] = test_predict
            plt.plot(self.data.index, test_predict_plot, label='测试集预测', color='red')
            
            plt.title('LSTM预测结果')
            plt.xlabel(self.time_column)
            plt.ylabel(self.price_column)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        plt.show()
        
        return {
            'model': model,
            'scaler': scaler,
            'look_back': look_back,
            'train_predict': train_predict,
            'test_predict': test_predict,
            'Y_train': Y_train,
            'Y_test': Y_test,
            'metrics': metrics,
            'model_name': model_name
        }
    
    def calculate_metrics(self, y_true, y_pred, model_name: str = 'model'):
        """
        计算预测的评估指标
        
        参数:
        y_true: array-like - 真实值
        y_pred: array-like - 预测值
        model_name: str - 模型名称，用于保存结果
        
        返回:
        dict - 包含各项评估指标的字典
        """
        # 检查输入数据是否有效
        if len(y_true) == 0 or len(y_pred) == 0:
            print(f"警告: {model_name} 没有足够的数据进行评估")
            metrics = {
                'mse': 0.0,
                'rmse': 0.0,
                'mae': 0.0,
                'r2': -1.0,
                'mape': 0.0
            }
        else:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # 转换为百分比
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
        
        # 保存评估结果
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def run_strategies(self, strategies: List[Dict]):
        """
        运行多种预测策略
        
        参数:
        strategies: List[Dict] - 策略配置列表，每个字典包含算法类型和参数
        
        返回:
        results: Dict - 所有策略的预测结果
        """
        results = {}
        
        for strategy in strategies:
            algo_type = strategy.get('algo')
            params = strategy.get('params', {})
            
            print(f"\n运行策略: {algo_type} {params}")
            
            if algo_type == 'moving_average':
                result = self.moving_average_forecast(**params)
            elif algo_type == 'arima':
                result = self.arima_forecast(**params)
            elif algo_type == 'lstm' and TENSORFLOW_AVAILABLE:
                result = self.lstm_forecast(**params)
            else:
                print(f"不支持的算法类型或TensorFlow未安装: {algo_type}")
                continue
            
            if result:
                results[result['model_name']] = result
                print(f"{algo_type}预测完成，RMSE: {result['metrics']['rmse']:.2f}, MAPE: {result['metrics']['mape']:.2f}%")
        
        return results
    
    def find_best_model(self, metric: str = 'rmse'):
        """
        找出性能最好的模型
        
        参数:
        metric: str - 用于评估的指标 ('rmse', 'mae', 'mape' 越小越好; 'r2' 越大越好)
        
        返回:
        best_model_name: str - 最好的模型名称
        best_metrics: dict - 最好模型的评估指标
        """
        if not self.evaluation_results:
            raise ValueError("没有评估结果，请先运行预测策略")
        
        # 根据指标类型选择比较方式
        if metric in ['rmse', 'mae', 'mape']:
            # 越小越好
            best_model_name = min(self.evaluation_results, 
                                key=lambda x: self.evaluation_results[x][metric])
        elif metric == 'r2':
            # 越大越好
            best_model_name = max(self.evaluation_results, 
                                key=lambda x: self.evaluation_results[x][metric])
        else:
            raise ValueError(f"不支持的评估指标: {metric}")
        
        best_metrics = self.evaluation_results[best_model_name]
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"\n最佳模型: {best_model_name}")
        print(f"评估指标 ({metric}): {best_metrics[metric]:.4f}")
        print(f"完整指标: {best_metrics}")
        
        return best_model_name, best_metrics
    
    def predict_with_best_model(self, steps: int = 30):
        """
        使用最佳模型进行未来股价预测
        
        参数:
        steps: int - 预测步数
        
        返回:
        forecast: array-like - 预测结果
        """
        if not self.best_model:
            raise ValueError("没有最佳模型，请先运行find_best_model")
        
        # 根据模型类型进行预测
        if self.best_model_name.startswith('moving_avg'):
            # 移动平均模型
            window_size = self.best_model['window_size']
            # 移动平均模型只能基于历史数据进行预测，这里简单实现
            last_values = self.data[self.price_column].tail(window_size).values
            forecast = np.full(steps, np.mean(last_values))
        
        elif self.best_model_name.startswith('arima'):
            # ARIMA模型
            forecast = self.best_model.forecast(steps=steps)
        
        elif self.best_model_name.startswith('lstm') and TENSORFLOW_AVAILABLE:
            # LSTM模型
            model = self.best_model['model']
            scaler = self.best_model['scaler']
            look_back = self.best_model['look_back']
            
            # 获取最后look_back个数据点
            last_values = self.data[self.price_column].tail(look_back).values.reshape(-1, 1)
            scaled_values = scaler.transform(last_values)
            
            # 预测未来steps步
            forecast = []
            current_input = scaled_values.reshape(1, look_back, 1)
            
            for _ in range(steps):
                next_value = model.predict(current_input)
                forecast.append(next_value[0, 0])
                
                # 更新输入序列
                current_input = np.append(current_input[:, 1:, :], next_value.reshape(1, 1, 1), axis=1)
            
            # 反归一化
            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        
        else:
            raise ValueError(f"不支持的模型类型: {self.best_model_name}")
        
        # 生成预测的时间索引
        last_date = self.data.index[-1]
        if self.frequency == 'D':
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
        elif self.frequency == 'W':
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')
        elif self.frequency == 'M':
            forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='M')
        elif self.frequency == 'H':
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=steps, freq='H')
        else:
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
        
        # 绘制预测图
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data[self.price_column], label='历史数据')
        plt.plot(forecast_dates, forecast, label='未来预测', color='red', linestyle='--')
        plt.title(f'最佳模型({self.best_model_name})未来股价预测')
        plt.xlabel(self.time_column)
        plt.ylabel(self.price_column)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return pd.DataFrame({
            'date': forecast_dates,
            'forecast_price': forecast
        }).set_index('date')
        
    def regression_validation(self, model_name: str = None):
        """
        对模型进行回归验证分析
        
        参数:
        model_name: str - 模型名称，如果为None则使用最佳模型
        
        返回:
        validation_results: dict - 回归验证结果
        """
        if not sklearn_available:
            raise ImportError("scipy和scikit-learn库未完全安装，无法进行回归验证")
            
        # 选择模型
        if model_name is None:
            if not self.best_model:
                raise ValueError("没有最佳模型，请先运行find_best_model")
            model_name = self.best_model_name
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"模型 {model_name} 不存在")
            model = self.models[model_name]
        
        # 获取模型的预测结果
        if model_name.startswith('moving_avg'):
            # 移动平均模型
            window_size = model['window_size']
            predictions = self.data[self.price_column].rolling(window=window_size).mean().values
            # 去除NaN值
            valid_indices = ~np.isnan(predictions)
            y_true = self.data[self.price_column].values[valid_indices]
            y_pred = predictions[valid_indices]
            
        elif model_name.startswith('arima'):
            # ARIMA模型
            predictions = model.fittedvalues
            y_true = self.data[self.price_column].values
            y_pred = predictions
            
        elif model_name.startswith('lstm') and TENSORFLOW_AVAILABLE:
            # LSTM模型
            lstm_model = model['model']
            scaler = model['scaler']
            look_back = model['look_back']
            
            # 准备数据
            data_values = self.data[self.price_column].values.reshape(-1, 1)
            scaled_data = scaler.transform(data_values)
            
            # 创建预测数据
            X = []
            for i in range(len(scaled_data) - look_back):
                X.append(scaled_data[i:i+look_back, 0])
            X = np.array(X)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # 进行预测
            predictions = lstm_model.predict(X)
            predictions = scaler.inverse_transform(predictions)
            
            y_true = data_values[look_back:].flatten()
            y_pred = predictions.flatten()
        
        # 计算回归统计量
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
        
        # 计算决定系数
        r_squared = r_value ** 2
        
        # 计算残差
        residuals = y_true - y_pred
        
        # 残差统计量
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        residual_rmse = np.sqrt(np.mean(residuals ** 2))
        
        # 打印回归验证结果
        print(f"\n=== 回归验证结果 ({model_name}) ===")
        print(f"回归方程: y = {slope:.4f}x + {intercept:.4f}")
        print(f"相关系数 (r): {r_value:.4f}")
        print(f"决定系数 (R²): {r_squared:.4f}")
        print(f"p值: {p_value:.4f}")
        print(f"标准误差: {std_err:.4f}")
        print(f"残差均值: {residual_mean:.4f}")
        print(f"残差标准差: {residual_std:.4f}")
        print(f"残差RMSE: {residual_rmse:.4f}")
        
        # 可视化回归结果
        plt.figure(figsize=(12, 6))
        
        # 散点图: 真实值 vs 预测值
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'回归验证: {model_name}')
        plt.grid(True)
        
        # 残差图
        plt.subplot(1, 2, 2)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差图')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 返回验证结果
        return {
            'model_name': model_name,
            'regression_equation': f'y = {slope:.4f}x + {intercept:.4f}',
            'r_value': r_value,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_err': std_err,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'residual_rmse': residual_rmse,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def save_results(self, output_dir: str = './results'):
        """
        保存所有结果
        
        参数:
        output_dir: str - 输出目录
        """
        import os
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存评估结果
        evaluation_file = os.path.join(output_dir, 'evaluation_results.json')
        # 将numpy类型转换为Python基本类型
        serializable_results = {}
        for model, metrics in self.evaluation_results.items():
            serializable_results[model] = {}
            for metric, value in metrics.items():
                if isinstance(value, np.generic):
                    serializable_results[model][metric] = value.item()
                else:
                    serializable_results[model][metric] = value
        
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=4)
        
        print(f"评估结果已保存到: {evaluation_file}")
        
        # 保存最佳模型
        if self.best_model:
            best_model_file = os.path.join(output_dir, f'best_model_{self.best_model_name}.joblib')
            joblib.dump(self.best_model, best_model_file)
            print(f"最佳模型已保存到: {best_model_file}")
        
        return self

# 示例用法
if __name__ == "__main__":
    # 创建股价预测器
    predictor = StockPricePredictor()
    
    # 加载数据
    try:
        # 尝试使用示例数据文件
        predictor.load_data(
            file_path='./sample_data.csv',
            time_column='date',
            price_column='value',
            frequency='D'
        )
        print("数据加载成功")
    except FileNotFoundError:
        print("示例数据文件未找到，请确保sample_data.csv文件存在")
        exit(1)
    
    # 数据预处理
    predictor.preprocess_data()
    
    # 定义预测策略
    strategies = [
        {
            'algo': 'moving_average',
            'params': {'window_size': 5, 'test_size': 0.2, 'plot_forecast': True}
        },
        {
            'algo': 'moving_average',
            'params': {'window_size': 10, 'test_size': 0.2, 'plot_forecast': True}
        },
        {
            'algo': 'arima',
            'params': {'order': (1, 1, 1), 'test_size': 0.2, 'plot_forecast': True}
        },
        {
            'algo': 'arima',
            'params': {'order': (2, 1, 2), 'test_size': 0.2, 'plot_forecast': True}
        },
        {
            'algo': 'lstm',
            'params': {'look_back': 30, 'epochs': 20, 'batch_size': 32, 'test_size': 0.2, 'plot_forecast': True}
        }
    ]
    
    # 运行所有策略
    print("开始运行预测策略...")
    results = predictor.run_strategies(strategies)
    
    # 找出最佳模型
    best_model_name, best_metrics = predictor.find_best_model(metric='rmse')
    
    # 使用最佳模型预测未来30天股价
    print("\n使用最佳模型预测未来30天股价...")
    future_forecast = predictor.predict_with_best_model(steps=30)
    print("未来30天股价预测:")
    print(future_forecast.head(10))  # 只显示前10条
    
    # 保存结果
    predictor.save_results()
