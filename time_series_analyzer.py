# coding: utf-8 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# 尝试导入tensorflow和keras，如果安装了则使用LSTM模型
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


# 方法一：临时设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

class TimeSeriesAnalyzer:
    """
    时序数据分析器，提供数据加载、预处理、可视化、特征提取和预测功能
    """
    
    def __init__(self):
        self.data = None
        self.time_column = None
        self.value_column = None
        self.frequency = None
        self.models = {}
        self.evaluation_results = {}
        self.decomposition_result = None
    
    def load_data(self, file_path, time_column, value_column, frequency='D', **kwargs):
        """
        加载时序数据
        
        参数:
        file_path: str - 数据文件路径
        time_column: str - 时间列名
        value_column: str - 数值列名
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
        
        # 设置时间列和数值列
        self.time_column = time_column
        self.value_column = value_column
        self.frequency = frequency
        
        # 将时间列转换为datetime类型并设置为索引
        self.data[time_column] = pd.to_datetime(self.data[time_column])
        self.data.set_index(time_column, inplace=True)
        
        # 按时间排序
        self.data.sort_index(inplace=True)
        
        return self
    
    def preprocess_data(self, impute_method='mean', fill_value=None, scaler_type=None):
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
        if self.data[self.value_column].isnull().any():
            if impute_method == 'constant' and fill_value is not None:
                self.data[self.value_column] = self.data[self.value_column].fillna(fill_value)
            else:
                imputer = SimpleImputer(strategy=impute_method)
                self.data[self.value_column] = imputer.fit_transform(self.data[[self.value_column]]).flatten()
        
        # 数据缩放
        if scaler_type:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("不支持的缩放方法，仅支持'standard'和'minmax'")
            
            self.data[self.value_column + '_scaled'] = scaler.fit_transform(self.data[[self.value_column]]).flatten()
        
        return self
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """
        异常值检测
        
        参数:
        method: str - 检测方法（'iqr'=四分位距, 'zscore'=Z分数）
        threshold: float - 异常值阈值
        
        返回:
        outliers: pd.DataFrame - 异常值数据
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        if method == 'iqr':
            Q1 = self.data[self.value_column].quantile(0.25)
            Q3 = self.data[self.value_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = self.data[(self.data[self.value_column] < lower_bound) | (self.data[self.value_column] > upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs((self.data[self.value_column] - self.data[self.value_column].mean()) / self.data[self.value_column].std())
            outliers = self.data[z_scores > threshold]
        
        else:
            raise ValueError("不支持的异常值检测方法，仅支持'iqr'和'zscore'")
        
        return outliers
    
    def get_summary_statistics(self):
        """
        获取数据的基本统计信息
        
        返回:
        summary: pd.DataFrame - 统计信息
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        return self.data[self.value_column].describe()
    
    def check_stationarity(self):
        """
        检查时序数据的平稳性（ADF检验）
        
        返回:
        result: dict - 检验结果
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        result = adfuller(self.data[self.value_column])
        return {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4],
            'Stationary': result[1] < 0.05
        }
    
    def decompose(self, model='additive'):
        """
        时序数据分解（趋势+季节性+残差）
        
        参数:
        model: str - 分解模型（'additive'=加法模型, 'multiplicative'=乘法模型）
        
        返回:
        decomposition: statsmodels.tsa.seasonal.DecomposeResult - 分解结果
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        return seasonal_decompose(self.data[self.value_column], model=model, period=self._get_period())
    
    def _get_period(self):
        """
        根据时间频率获取分解周期
        """
        if self.frequency == 'D':
            return 7  # 周周期
        elif self.frequency == 'W':
            return 52  # 年周期
        elif self.frequency == 'M':
            return 12  # 年周期
        elif self.frequency == 'H':
            return 24  # 日周期
        else:
            return 12  # 默认年周期
    
    def plot_time_series(self, title=None, figsize=(12, 6), save_path=None):
        """
        绘制时序趋势图
        
        参数:
        title: str - 图表标题
        figsize: tuple - 图表大小
        save_path: str - 保存路径（如果提供）
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        plt.figure(figsize=figsize)
        plt.plot(self.data.index, self.data[self.value_column], color='blue', linewidth=1.5)
        plt.title(title or f"时序趋势图 - {self.value_column}")
        plt.xlabel(self.time_column)
        plt.ylabel(self.value_column)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
    
    def plot_decomposition(self, model='additive', title=None, figsize=(12, 8), save_path=None, show_interpretation=True):
        """
        绘制时序分解图（趋势+季节性+残差）
        
        参数:
        model: str - 分解模型（'additive'=加法模型, 'multiplicative'=乘法模型）
        title: str - 图表标题
        figsize: tuple - 图表大小
        save_path: str - 保存路径（如果提供）
        show_interpretation: bool - 是否显示解读说明
        """
        decomposition = self.decompose(model=model)
        
        plt.figure(figsize=figsize)
        
        plt.subplot(411)
        plt.plot(decomposition.observed, color='blue')
        plt.title(title or f"时序分解图 - {self.value_column}")
        plt.ylabel('观测值')
        
        plt.subplot(412)
        plt.plot(decomposition.trend, color='red')
        plt.ylabel('趋势')
        
        plt.subplot(413)
        plt.plot(decomposition.seasonal, color='green')
        plt.ylabel('季节性')
        
        plt.subplot(414)
        plt.plot(decomposition.resid, color='purple')
        plt.ylabel('残差')
        plt.xlabel(self.time_column)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        
        # 显示解读说明
        if show_interpretation:
            # 分析趋势
            trend_values = decomposition.trend.dropna()
            trend_increasing = trend_values.iloc[-1] > trend_values.iloc[0]
            trend_decreasing = trend_values.iloc[-1] < trend_values.iloc[0]
            
            # 分析季节性
            seasonal_strong = np.std(decomposition.seasonal) > 0.2 * np.std(self.data[self.value_column])
            
            # 分析残差
            residuals_small = np.std(decomposition.resid.dropna()) < 0.3 * np.std(self.data[self.value_column])
            
            interpretation = self.interpret_chart('decomposition', {
                'trend_increasing': trend_increasing,
                'trend_decreasing': trend_decreasing,
                'seasonality_strong': seasonal_strong,
                'residuals_small': residuals_small
            })
            print("\n" + "="*50)
            print(interpretation)
            print("="*50 + "\n")
    
    def plot_acf_pacf(self, lags=30, figsize=(12, 6), save_path=None, show_interpretation=True):
        """
        绘制自相关和偏自相关图
        
        参数:
        lags: int - 滞后阶数
        figsize: tuple - 图表大小
        save_path: str - 保存路径（如果提供）
        show_interpretation: bool - 是否显示解读说明
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        plt.figure(figsize=figsize)
        
        plt.subplot(121)
        plot_acf(self.data[self.value_column], lags=lags, ax=plt.gca())
        plt.title('自相关图 (ACF)')
        
        plt.subplot(122)
        plot_pacf(self.data[self.value_column], lags=lags, ax=plt.gca())
        plt.title('偏自相关图 (PACF)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        
        # 显示解读说明
        if show_interpretation:
            from statsmodels.tsa.stattools import acf, pacf
            acf_result = acf(self.data[self.value_column], nlags=lags)
            pacf_result = pacf(self.data[self.value_column], nlags=lags)
            
            interpretation = self.interpret_chart('acf_pacf', {
                'acf_lags': acf_result,
                'pacf_lags': pacf_result
            })
            print("\n" + "="*50)
            print(interpretation)
            print("="*50 + "\n")
    
    def plot_boxplot(self, by='month', title=None, figsize=(12, 6), save_path=None, show_interpretation=True):
        """
        绘制箱线图（按时间周期）
        
        参数:
        by: str - 分组方式（'year', 'month', 'quarter', 'dayofweek'）
        title: str - 图表标题
        figsize: tuple - 图表大小
        save_path: str - 保存路径（如果提供）
        show_interpretation: bool - 是否显示解读说明
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        data_copy = self.data.copy()
        
        if by == 'year':
            data_copy['group'] = data_copy.index.year
        elif by == 'month':
            data_copy['group'] = data_copy.index.month
        elif by == 'quarter':
            data_copy['group'] = data_copy.index.quarter
        elif by == 'dayofweek':
            data_copy['group'] = data_copy.index.dayofweek
        else:
            raise ValueError("不支持的分组方式")
        
        plt.figure(figsize=figsize)
        sns.boxplot(x='group', y=self.value_column, data=data_copy)
        plt.title(title or f"箱线图 - 按{by}分组")
        plt.xlabel(by)
        plt.ylabel(self.value_column)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        
        # 显示解读说明
        if show_interpretation:
            # 分析箱线图
            outliers = any(data_copy.groupby('group').apply(lambda x: len(x) - len(sns.boxplot(x=x[self.value_column]).artists)) > 0)
            
            # 计算不同分组的方差
            group_variances = data_copy.groupby('group')[self.value_column].var()
            variance_high = group_variances.max() / group_variances.min() > 2
            
            # 分析中位数趋势
            medians = data_copy.groupby('group')[self.value_column].median()
            median_trend = 'stable'
            if len(medians) > 1:
                if medians.iloc[-1] > medians.iloc[0] * 1.1:
                    median_trend = 'increasing'
                elif medians.iloc[-1] < medians.iloc[0] * 0.9:
                    median_trend = 'decreasing'
            
            interpretation = self.interpret_chart('boxplot', {
                'by': by,
                'outliers': outliers,
                'variance_high': variance_high,
                'median_trend': median_trend
            })
            print("\n" + "="*50)
            print(interpretation)
            print("="*50 + "\n")
    
    def interpret_chart(self, chart_type, data=None, model_results=None):
        """
        为不同类型的图表生成解读说明
        
        参数:
        chart_type: str - 图表类型
        data: dict - 图表数据
        model_results: dict - 模型结果（如果有）
        
        返回:
        str - 图表解读说明
        """
        interpretation = []
        
        if chart_type == 'acf_pacf':
            interpretation.append("**ACF/PACF图解读**：")
            interpretation.append("- ACF（自相关函数）展示了时间序列与其自身滞后版本的相关性")
            interpretation.append("- PACF（偏自相关函数）展示了在控制中间滞后项影响后，时间序列与其特定滞后版本的相关性")
            
            acf_lags = data.get('acf_lags', [])
            pacf_lags = data.get('pacf_lags', [])
            
            # 分析ACF
            significant_acf = [lag for lag, value in enumerate(acf_lags) if abs(value) > 1.96/len(acf_lags)**0.5]
            if significant_acf:
                interpretation.append(f"- ACF在滞后阶数 {significant_acf[:3]} 处显示显著相关性，表明存在一定的自相关性")
            
            # 分析PACF
            significant_pacf = [lag for lag, value in enumerate(pacf_lags) if abs(value) > 1.96/len(pacf_lags)**0.5]
            if significant_pacf:
                interpretation.append(f"- PACF在滞后阶数 {significant_pacf[:3]} 处显示显著相关性，可能暗示AR模型的阶数")
            
            interpretation.append("- 这些图表可用于确定ARIMA模型的p和q参数")
            
        elif chart_type == 'decomposition':
            interpretation.append("**时间序列分解图解读**：")
            interpretation.append("- 趋势分量：展示数据的长期变化趋势")
            interpretation.append("- 季节性分量：展示数据的周期性变化模式")
            interpretation.append("- 残差分量：展示去除趋势和季节性后的随机波动")
            
            # 分析趋势
            if data.get('trend_increasing', False):
                interpretation.append("- 趋势分量呈现上升趋势，表明整体数据在观察期内持续增长")
            elif data.get('trend_decreasing', False):
                interpretation.append("- 趋势分量呈现下降趋势，表明整体数据在观察期内持续减少")
            else:
                interpretation.append("- 趋势分量相对平稳，没有明显的上升或下降趋势")
            
            # 分析季节性
            if data.get('seasonality_strong', False):
                interpretation.append("- 季节性分量非常明显，说明数据存在强烈的周期性变化")
            else:
                interpretation.append("- 季节性分量相对较弱，说明数据的周期性变化不明显")
            
            # 分析残差
            if data.get('residuals_small', False):
                interpretation.append("- 残差分量相对较小，说明趋势和季节性模型能很好地解释数据")
            else:
                interpretation.append("- 残差分量较大，可能存在未被捕获的模式或异常值")
        
        elif chart_type == 'boxplot':
            interpretation.append("**箱线图解读**：")
            interpretation.append(f"- 图表展示了{data.get('by', '月份')}维度上{self.value_column}的分布情况")
            
            if data.get('outliers', False):
                interpretation.append("- 图表中存在异常值（箱线外的点），这些可能是数据中的特殊情况或测量误差")
            
            if data.get('variance_high', False):
                interpretation.append("- 不同分组之间的分布差异较大，说明{data.get('by', '月份')}对{self.value_column}有显著影响")
            
            if data.get('median_trend', 'stable') == 'increasing':
                interpretation.append("- 中位数呈现上升趋势，表明整体水平随时间递增")
            elif data.get('median_trend', 'stable') == 'decreasing':
                interpretation.append("- 中位数呈现下降趋势，表明整体水平随时间递减")
            else:
                interpretation.append("- 中位数相对稳定，表明整体水平没有明显变化")
        
        elif chart_type == 'heatmap':
            interpretation.append("**热力图解读**：")
            interpretation.append(f"- 图表展示了{self.value_column}在不同时间维度上的分布模式")
            
            if data.get('seasonal_pattern', False):
                interpretation.append("- 可以观察到明显的季节性模式，某些时间段的值明显高于或低于其他时间段")
            
            if data.get('trend_visible', False):
                interpretation.append("- 可以观察到长期趋势，整体数值随时间呈现上升或下降趋势")
            
            peak_periods = data.get('peak_periods', [])
            if peak_periods:
                interpretation.append(f"- 峰值主要出现在{peak_periods[:3]}等时间段，这些可能是重要的时间节点")
        
        elif chart_type == 'forecast':
            interpretation.append("**预测结果分析**：")
            interpretation.append(f"- 使用{model_results.get('model_name', '模型')}对{self.value_column}进行了预测")
            
            # 分析预测准确性
            rmse = model_results.get('metrics', {}).get('rmse', 0)
            mae = model_results.get('metrics', {}).get('mae', 0)
            r2 = model_results.get('metrics', {}).get('r2', 0)
            
            if r2 > 0.8:
                interpretation.append(f"- 模型拟合效果优秀（R²={r2:.2f}），预测值与实际值高度一致")
            elif r2 > 0.6:
                interpretation.append(f"- 模型拟合效果良好（R²={r2:.2f}），预测值与实际值较为一致")
            else:
                interpretation.append(f"- 模型拟合效果一般（R²={r2:.2f}），可能需要进一步优化模型或考虑更多因素")
            
            interpretation.append(f"- 预测误差：RMSE={rmse:.2f}，MAE={mae:.2f}")
            
            # 分析预测趋势
            forecast_values = model_results.get('predictions', model_results.get('forecast', []))
            if len(forecast_values) > 1:
                if forecast_values[-1] > forecast_values[0]:
                    interpretation.append("- 预测结果显示未来趋势将继续上升")
                elif forecast_values[-1] < forecast_values[0]:
                    interpretation.append("- 预测结果显示未来趋势将继续下降")
                else:
                    interpretation.append("- 预测结果显示未来趋势相对平稳")
        
        return '\n'.join(interpretation)
    
    def plot_heatmap(self, pivot_table=True, title=None, figsize=(12, 8), save_path=None, show_interpretation=True):
        """
        绘制热力图（按时间周期）
        
        参数:
        pivot_table: bool - 是否生成透视表
        title: str - 图表标题
        figsize: tuple - 图表大小
        save_path: str - 保存路径（如果提供）
        show_interpretation: bool - 是否显示解读说明
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        data_copy = self.data.copy()
        
        if self.frequency == 'D':
            # 按年和月绘制热力图
            data_copy['year'] = data_copy.index.year
            data_copy['month'] = data_copy.index.month
            data_copy['day'] = data_copy.index.day
            pivot = data_copy.pivot_table(values=self.value_column, index=['year', 'month'], columns='day', aggfunc='mean')
        elif self.frequency == 'H':
            # 按天和小时绘制热力图
            data_copy['date'] = data_copy.index.date
            data_copy['hour'] = data_copy.index.hour
            pivot = data_copy.pivot_table(values=self.value_column, index='date', columns='hour', aggfunc='mean')
        else:
            # 默认按年和月绘制
            data_copy['year'] = data_copy.index.year
            data_copy['month'] = data_copy.index.month
            pivot = data_copy.pivot_table(values=self.value_column, index='year', columns='month', aggfunc='mean')
        
        plt.figure(figsize=figsize)
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.1f', cbar_kws={'label': self.value_column})
        plt.title(title or f"热力图 - {self.value_column}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        
        # 显示解读说明
        if show_interpretation:
            # 分析热力图
            seasonal_pattern = False
            trend_visible = False
            peak_periods = []
            
            if self.frequency == 'D':
                # 检查季节性模式（月度）
                monthly_means = data_copy.groupby('month')[self.value_column].mean()
                seasonal_pattern = monthly_means.max() / monthly_means.min() > 1.3
                
                # 检查年度趋势
                yearly_means = data_copy.groupby('year')[self.value_column].mean()
                if len(yearly_means) > 1:
                    trend_visible = abs(yearly_means.iloc[-1] - yearly_means.iloc[0]) / yearly_means.iloc[0] > 0.1
                
                # 找出峰值月份
                peak_months = monthly_means.nlargest(3).index.tolist()
                peak_periods = [f"第{month}个月" for month in peak_months]
            
            interpretation = self.interpret_chart('heatmap', {
                'seasonal_pattern': seasonal_pattern,
                'trend_visible': trend_visible,
                'peak_periods': peak_periods
            })
            print("\n" + "="*50)
            print(interpretation)
            print("="*50 + "\n")
    
    def add_moving_average(self, window_sizes=[7, 30, 90], types=['sma']):
        """
        添加移动平均特征
        
        参数:
        window_sizes: list - 窗口大小列表
        types: list - 移动平均类型（'sma'=简单移动平均, 'wma'=加权移动平均, 'ema'=指数移动平均）
        
        返回:
        self - 返回自身以支持链式调用
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        for window in window_sizes:
            for ma_type in types:
                if ma_type == 'sma':
                    # 简单移动平均
                    self.data[f'{self.value_column}_sma_{window}'] = self.data[self.value_column].rolling(window=window).mean()
                elif ma_type == 'wma':
                    # 加权移动平均
                    weights = np.arange(1, window + 1)
                    self.data[f'{self.value_column}_wma_{window}'] = self.data[self.value_column].rolling(window=window).apply(
                        lambda x: np.sum(x * weights) / np.sum(weights)
                    )
                elif ma_type == 'ema':
                    # 指数移动平均
                    self.data[f'{self.value_column}_ema_{window}'] = self.data[self.value_column].ewm(span=window, adjust=False).mean()
                else:
                    raise ValueError(f"不支持的移动平均类型: {ma_type}")
        
        return self
    
    def add_differencing(self, lags=[1, 2]):
        """
        添加差分特征
        
        参数:
        lags: list - 差分阶数列表
        
        返回:
        self - 返回自身以支持链式调用
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        for lag in lags:
            self.data[f'{self.value_column}_diff_{lag}'] = self.data[self.value_column].diff(periods=lag)
        
        return self
    
    def add_lag_features(self, lags=[1, 2, 7, 14, 30]):
        """
        添加滞后特征
        
        参数:
        lags: list - 滞后阶数列表
        
        返回:
        self - 返回自身以支持链式调用
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        for lag in lags:
            self.data[f'{self.value_column}_lag_{lag}'] = self.data[self.value_column].shift(periods=lag)
        
        return self
    
    def add_rolling_stats(self, window_sizes=[7, 30], stats=['min', 'max', 'mean', 'std']):
        """
        添加滚动统计特征
        
        参数:
        window_sizes: list - 窗口大小列表
        stats: list - 统计指标列表（'min', 'max', 'mean', 'std', 'median', 'sum'）
        
        返回:
        self - 返回自身以支持链式调用
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        for window in window_sizes:
            for stat in stats:
                if stat == 'min':
                    self.data[f'{self.value_column}_roll_{stat}_{window}'] = self.data[self.value_column].rolling(window=window).min()
                elif stat == 'max':
                    self.data[f'{self.value_column}_roll_{stat}_{window}'] = self.data[self.value_column].rolling(window=window).max()
                elif stat == 'mean':
                    self.data[f'{self.value_column}_roll_{stat}_{window}'] = self.data[self.value_column].rolling(window=window).mean()
                elif stat == 'std':
                    self.data[f'{self.value_column}_roll_{stat}_{window}'] = self.data[self.value_column].rolling(window=window).std()
                elif stat == 'median':
                    self.data[f'{self.value_column}_roll_{stat}_{window}'] = self.data[self.value_column].rolling(window=window).median()
                elif stat == 'sum':
                    self.data[f'{self.value_column}_roll_{stat}_{window}'] = self.data[self.value_column].rolling(window=window).sum()
                else:
                    raise ValueError(f"不支持的统计指标: {stat}")
        
        return self
    
    def add_time_features(self, features=['year', 'month', 'day', 'dayofweek', 'quarter']):
        """
        添加时间相关特征
        
        参数:
        features: list - 时间特征列表（'year', 'month', 'day', 'dayofweek', 'quarter', 'hour', 'minute', 'second'）
        
        返回:
        self - 返回自身以支持链式调用
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        for feature in features:
            if feature == 'year':
                self.data['year'] = self.data.index.year
            elif feature == 'month':
                self.data['month'] = self.data.index.month
            elif feature == 'day':
                self.data['day'] = self.data.index.day
            elif feature == 'dayofweek':
                self.data['dayofweek'] = self.data.index.dayofweek
            elif feature == 'quarter':
                self.data['quarter'] = self.data.index.quarter
            elif feature == 'hour':
                self.data['hour'] = self.data.index.hour
            elif feature == 'minute':
                self.data['minute'] = self.data.index.minute
            elif feature == 'second':
                self.data['second'] = self.data.index.second
            else:
                raise ValueError(f"不支持的时间特征: {feature}")
        
        return self
    
    def add_seasonal_features(self):
        """
        添加季节性特征（正弦和余弦变换）
        
        返回:
        self - 返回自身以支持链式调用
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 年度季节性
        self.data['sin_year'] = np.sin(2 * np.pi * self.data.index.dayofyear / 365.25)
        self.data['cos_year'] = np.cos(2 * np.pi * self.data.index.dayofyear / 365.25)
        
        # 月度季节性
        self.data['sin_month'] = np.sin(2 * np.pi * self.data.index.month / 12)
        self.data['cos_month'] = np.cos(2 * np.pi * self.data.index.month / 12)
        
        # 周季节性
        self.data['sin_week'] = np.sin(2 * np.pi * self.data.index.dayofweek / 7)
        self.data['cos_week'] = np.cos(2 * np.pi * self.data.index.dayofweek / 7)
        
        # 日季节性（如果是小时级数据）
        if self.frequency == 'H':
            self.data['sin_day'] = np.sin(2 * np.pi * self.data.index.hour / 24)
            self.data['cos_day'] = np.cos(2 * np.pi * self.data.index.hour / 24)
        
        return self
    
    def train_test_split(self, test_size=0.2):
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
    
    def arima_forecast(self, order=(1, 1, 1), test_size=0.2, plot_forecast=True):
        """
        使用ARIMA模型进行预测
        
        参数:
        order: tuple - ARIMA模型参数(p, d, q)
        test_size: float - 测试集比例
        plot_forecast: bool - 是否绘制预测图
        
        返回:
        results: dict - 预测结果和评估指标
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 划分训练集和测试集
        train, test = self.train_test_split(test_size=test_size)
        
        # 拟合ARIMA模型
        try:
            model = ARIMA(train[self.value_column], order=order)
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
        metrics = self.calculate_metrics(test[self.value_column], predictions, model_name=f'arima_{order[0]}_{order[1]}_{order[2]}')
        
        # 绘制预测图
        if plot_forecast:
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train[self.value_column], label='训练集')
            plt.plot(test.index, test[self.value_column], label='测试集')
            plt.plot(test.index, predictions, label='预测值', color='red')
            plt.title(f'ARIMA预测结果 (order={order})')
            plt.xlabel(self.time_column)
            plt.ylabel(self.value_column)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        plt.show()
        
        # 显示解读说明
        print("\n" + "="*50)
        interpretation = self.interpret_chart('forecast', model_results={
            'model_name': f'ARIMA({order[0]},{order[1]},{order[2]})',
            'metrics': metrics,
            'forecast': predictions
        })
        print(interpretation)
        print("="*50 + "\n")
        
        return {
            'model': model_fit,
            'forecast': predictions,
            'test_data': test,
            'metrics': metrics,
            'model_name': f'ARIMA({order[0]},{order[1]},{order[2]})'
        }
    
    def lstm_forecast(self, look_back=30, epochs=50, batch_size=32, test_size=0.2, plot_forecast=True, complex_model=False):
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
            raise ImportError("TensorFlow和Keras未安装，无法使用LSTM模型")
        
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
        data_values = self.data[self.value_column].values.reshape(-1, 1)
        
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
            from tensorflow.keras.layers import Bidirectional, GRU, Conv1D, MaxPooling1D, Flatten, Attention, Input
            from tensorflow.keras.models import Model
            
            # 使用Functional API构建模型
            inputs = Input(shape=(X_train.shape[1], 1))
            
            # 1. 添加CNN层提取特征
            x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(0.2)(x)
            
            # 2. 双向LSTM层
            x = Bidirectional(LSTM(units=100, return_sequences=True))(x)
            x = Dropout(0.2)(x)
            
            # 3. 多层LSTM
            x = LSTM(units=150, return_sequences=True)(x)
            x = Dropout(0.2)(x)
            x = LSTM(units=100, return_sequences=True)(x)
            x = Dropout(0.2)(x)
            
            # 4. 注意力层（使用自注意力）
            attention_output = Attention()([x, x])
            
            # 5. 展平并添加全连接层
            x = Flatten()(attention_output)
            x = Dense(units=64, activation='relu')(x)
            x = Dropout(0.2)(x)
            x = Dense(units=32, activation='relu')(x)
            outputs = Dense(units=1)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
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
        model_suffix = '_complex' if complex_model else ''
        metrics = self.calculate_metrics(Y_test, test_predict[:, 0], model_name=f'lstm_{look_back}_{epochs}_{batch_size}{model_suffix}')
        
        # 保存模型
        if TENSORFLOW_AVAILABLE:
            model_name = f'lstm_{look_back}_{epochs}_{batch_size}{model_suffix}'
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
            plt.ylabel(self.value_column)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        plt.show()
        
        # 显示解读说明
        print("\n" + "="*50)
        interpretation = self.interpret_chart('forecast', model_results={
            'model_name': f'LSTM(look_back={look_back}, epochs={epochs}, batch_size={batch_size})',
            'metrics': metrics,
            'predictions': test_predict.flatten()
        })
        print(interpretation)
        print("="*50 + "\n")
        
        return {
            'model': model,
            'scaler': scaler,
            'look_back': look_back,
            'train_predict': train_predict,
            'test_predict': test_predict,
            'Y_train': Y_train,
            'Y_test': Y_test,
            'metrics': metrics,
            'model_name': f'LSTM(look_back={look_back}, epochs={epochs}, batch_size={batch_size})'
        }
    
    def calculate_metrics(self, y_true, y_pred, model_name='model'):
        """
        计算时间序列预测的评估指标
        
        参数:
        y_true: array-like - 真实值
        y_pred: array-like - 预测值
        model_name: str - 模型名称，用于保存结果
        
        返回:
        dict - 包含各项评估指标的字典
        """
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
    
    def save_data(self, file_path, format='csv'):
        """
        保存处理后的数据
        
        参数:
        file_path: str - 保存路径
        format: str - 文件格式 (csv, excel, feather)
        """
        if self.data is None:
            raise ValueError("没有数据可保存")
        
        if format == 'csv':
            self.data.to_csv(file_path, index=True)
        elif format == 'excel':
            self.data.to_excel(file_path, index=True)
        elif format == 'feather':
            self.data.to_feather(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {format}")
        
        print(f"数据已保存到: {file_path}")
        return self
    
    def save_model(self, model_name, file_path):
        """
        保存训练好的模型
        
        参数:
        model_name: str - 模型名称（在models字典中的键）
        file_path: str - 保存路径
        """
        if model_name not in self.models:
            raise ValueError(f"模型 '{model_name}' 不存在")
        
        joblib.dump(self.models[model_name], file_path)
        print(f"模型 '{model_name}' 已保存到: {file_path}")
        return self
    
    def load_model(self, model_name, file_path):
        """
        加载已保存的模型
        
        参数:
        model_name: str - 模型名称（用于存储在models字典中的键）
        file_path: str - 模型文件路径
        """
        self.models[model_name] = joblib.load(file_path)
        print(f"模型 '{model_name}' 已从 {file_path} 加载")
        return self
    
    def save_evaluation_results(self, file_path):
        """
        保存评估结果到JSON文件
        
        参数:
        file_path: str - 保存路径
        """
        if not self.evaluation_results:
            print("没有评估结果可保存")
            return self
        
        # 将numpy类型转换为Python基本类型
        serializable_results = {}
        for model, metrics in self.evaluation_results.items():
            serializable_results[model] = {}
            for metric, value in metrics.items():
                if isinstance(value, np.generic):
                    serializable_results[model][metric] = value.item()
                else:
                    serializable_results[model][metric] = value
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=4)
        
        print(f"评估结果已保存到: {file_path}")
        return self
    
    def save_plot(self, fig, file_path, dpi=300, format='png'):
        """
        保存图表
        
        参数:
        fig: matplotlib.figure.Figure - 图表对象
        file_path: str - 保存路径
        dpi: int - 图像分辨率
        format: str - 图像格式
        """
        fig.savefig(file_path, dpi=dpi, format=format, bbox_inches='tight')
        print(f"图表已保存到: {file_path}")
        return self
    
    def compare_models(self, plot_results=True, show_interpretation=True):
        """
        比较不同模型的评估指标
        
        参数:
        plot_results: bool - 是否绘制比较结果
        show_interpretation: bool - 是否显示解读说明
        
        返回:
        pandas.DataFrame - 包含所有模型评估指标的DataFrame
        """
        if not self.evaluation_results:
            print("没有评估结果可比较")
            return None
        
        # 转换为DataFrame便于查看和比较
        comparison_df = pd.DataFrame(self.evaluation_results).T
        print("\n模型评估指标比较:")
        print(comparison_df)
        
        if plot_results and len(self.evaluation_results) > 1:
            # 绘制雷达图
            metrics = ['rmse', 'mae', 'mape']
            models = list(self.evaluation_results.keys())
            
            # 标准化指标（越小越好）
            normalized_data = []
            for model in models:
                model_data = []
                for metric in metrics:
                    max_val = max(self.evaluation_results[m][metric] for m in models)
                    min_val = min(self.evaluation_results[m][metric] for m in models)
                    if max_val == min_val:
                        value = 1.0
                    else:
                        value = (max_val - self.evaluation_results[model][metric]) / (max_val - min_val)
                    model_data.append(value)
                normalized_data.append(model_data)
            
            # 绘制雷达图
            num_vars = len(metrics)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            
            for i, (model, data) in enumerate(zip(models, normalized_data)):
                data += data[:1]  # 闭合
                ax.plot(angles, data, linewidth=2, label=model)
                ax.fill(angles, data, alpha=0.25)
            
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # 设置角度标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            
            # 设置刻度范围
            ax.set_ylim(0, 1)
            
            plt.title('模型性能比较（标准化）', size=15, y=1.1)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.tight_layout()
            plt.show()
        
        # 显示解读说明
        if show_interpretation:
            print("\n" + "="*50)
            print("**模型比较分析结论**：")
            
            # 找出最佳模型
            best_rmse_model = comparison_df['rmse'].idxmin()
            best_mae_model = comparison_df['mae'].idxmin()
            best_r2_model = comparison_df['r2'].idxmax()
            
            print(f"- 均方根误差(RMSE)最小的模型：{best_rmse_model} ({comparison_df.loc[best_rmse_model, 'rmse']:.2f})")
            print(f"- 平均绝对误差(MAE)最小的模型：{best_mae_model} ({comparison_df.loc[best_mae_model, 'mae']:.2f})")
            print(f"- 决定系数(R²)最大的模型：{best_r2_model} ({comparison_df.loc[best_r2_model, 'r2']:.2f})")
            
            # 综合评估
            if best_rmse_model == best_mae_model == best_r2_model:
                print(f"\n- 综合评估：{best_rmse_model}在所有指标上都表现最佳，是最优选择")
            else:
                # 计算综合得分
                weights = {'rmse': 0.3, 'mae': 0.3, 'r2': 0.4}
                scores = {}
                for model_name in comparison_df.index:
                    score = (1 - comparison_df.loc[model_name, 'rmse']/comparison_df['rmse'].max()) * weights['rmse'] + \
                            (1 - comparison_df.loc[model_name, 'mae']/comparison_df['mae'].max()) * weights['mae'] + \
                            (comparison_df.loc[model_name, 'r2']/comparison_df['r2'].max()) * weights['r2']
                    scores[model_name] = score
                
                best_overall_model = max(scores, key=scores.get)
                print(f"\n- 综合得分最高的模型：{best_overall_model} (得分: {scores[best_overall_model]:.3f})")
            
            # 模型特点分析
            for model_name in comparison_df.index:
                r2 = comparison_df.loc[model_name, 'r2']
                print(f"\n- {model_name}：")
                if r2 > 0.8:
                    print("  ✓ 拟合效果优秀，能够很好地捕捉数据模式")
                    print("  ✓ 预测结果可靠，适合用于决策支持")
                elif r2 > 0.6:
                    print("  ✓ 拟合效果良好，能够捕捉主要数据模式")
                    print("  ✓ 预测结果基本可靠，可用于初步分析")
                else:
                    print("  ✓ 拟合效果一般，可能需要进一步优化")
                    print("  ✓ 建议结合其他模型或增加特征进行改进")
            
            print("="*50 + "\n")
        
        return comparison_df
    
    def moving_average_forecast(self, window_size=30, test_size=0.2, plot_forecast=True):
        """
        使用移动平均进行预测
        
        参数:
        window_size: int - 移动平均窗口大小
        test_size: float - 测试集比例
        plot_forecast: bool - 是否绘制预测图
        
        返回:
        results: dict - 预测结果和评估指标
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 划分训练集和测试集
        train, test = self.train_test_split(test_size=test_size)
        
        # 计算移动平均
        history = list(train[self.value_column])
        predictions = []
        
        for i in range(len(test)):
            # 计算移动平均
            if len(history) >= window_size:
                prediction = np.mean(history[-window_size:])
            else:
                prediction = np.mean(history)
            predictions.append(prediction)
            # 添加真实值到历史中（模拟实时预测）
            history.append(test[self.value_column].iloc[i])
        
        # 计算评估指标
        metrics = self.calculate_metrics(test[self.value_column], predictions, model_name=f'moving_average_{window_size}')
        
        # 绘制预测图
        if plot_forecast:
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train[self.value_column], label='训练集')
            plt.plot(test.index, test[self.value_column], label='测试集')
            plt.plot(test.index, predictions, label='预测值', color='red')
            plt.title(f'移动平均预测结果 (窗口大小={window_size})')
            plt.xlabel(self.time_column)
            plt.ylabel(self.value_column)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        plt.show()
        
        # 显示解读说明
        print("\n" + "="*50)
        interpretation = self.interpret_chart('forecast', model_results={
            'model_name': f'Moving Average(window={window_size})',
            'metrics': metrics,
            'predictions': predictions
        })
        print(interpretation)
        print("="*50 + "\n")
        
        return {
            'predictions': predictions,
            'test_data': test,
            'metrics': metrics,
            'model_name': f'Moving Average(window={window_size})'
        }

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    date_rng = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    values = np.random.randn(len(date_rng)).cumsum() + 100  # 随机游走数据
    values += np.sin(np.arange(len(date_rng)) * 2 * np.pi / 365) * 20  # 添加年度季节性
    values += np.random.normal(0, 5, len(date_rng))  # 添加噪声
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': date_rng,
        'value': values
    })
    
    # 保存为示例文件
    df.to_csv('c:\\Users\\songdj\\weibo\\weibo-crawler\\timeserial\\sample_data.csv', index=False)
    
    # 使用时序分析器
    analyzer = TimeSeriesAnalyzer()
    analyzer.load_data(
        file_path='c:\\Users\\songdj\\weibo\\weibo-crawler\\timeserial\\sample_data.csv',
        time_column='date',
        value_column='value',
        frequency='D'
    )
    
    # 数据预处理
    analyzer.preprocess_data(impute_method='mean', scaler_type='standard')
    
    # 获取统计信息
    print("数据统计信息:")
    print(analyzer.get_summary_statistics())
    
    # 检查平稳性
    print("\n平稳性检验 (ADF):")
    stationarity_result = analyzer.check_stationarity()
    for key, value in stationarity_result.items():
        print(f"{key}: {value}")
    
    # 检测异常值
    outliers = analyzer.detect_outliers(method='iqr', threshold=1.5)
    print(f"\n检测到 {len(outliers)} 个异常值")
    if len(outliers) > 0:
        print("异常值示例:")
        print(outliers.head())
    
    # 特征提取
    analyzer.add_moving_average(window_sizes=[7, 30], types=['sma', 'ema'])
    analyzer.add_differencing(lags=[1, 2])
    analyzer.add_lag_features(lags=[1, 7, 30])
    analyzer.add_rolling_stats(window_sizes=[7, 30], stats=['min', 'max', 'mean', 'std'])
    analyzer.add_time_features(features=['year', 'month', 'day', 'dayofweek', 'quarter'])
    analyzer.add_seasonal_features()
    
    print("\n特征提取完成，数据列名:")
    print(analyzer.data.columns.tolist())
    print("\n特征数据前5行:")
    print(analyzer.data.tail(5))
    
    # 绘制可视化图表
    analyzer.plot_time_series(title="示例时序数据趋势图")
    analyzer.plot_decomposition(model='additive', title="示例时序数据分解图")
    analyzer.plot_acf_pacf(lags=30)
    analyzer.plot_boxplot(by='month', title="按月分组的箱线图")
    analyzer.plot_heatmap(title="时序数据热力图")
    
    # 预测模型演示
    print("\n=== 预测模型演示 ===")
    
    # 移动平均预测
    print("\n1. 移动平均预测:")
    ma_results = analyzer.moving_average_forecast(window_size=30, test_size=0.2)
    print(f"移动平均预测指标: RMSE={ma_results['metrics']['rmse']:.2f}, MAE={ma_results['metrics']['mae']:.2f}, R2={ma_results['metrics']['r2']:.2f}")
    
    # ARIMA预测
    print("\n2. ARIMA预测:")
    arima_results = analyzer.arima_forecast(order=(1, 1, 1), test_size=0.2)
    print(f"ARIMA预测指标: RMSE={arima_results['metrics']['rmse']:.2f}, MAE={arima_results['metrics']['mae']:.2f}, R2={arima_results['metrics']['r2']:.2f}")
    
    # LSTM预测（如果安装了tensorflow）
    try:
        print("\n3. LSTM预测:")
        lstm_results = analyzer.lstm_forecast(look_back=30, epochs=10, batch_size=32, test_size=0.2)
        print(f"LSTM预测指标: RMSE={lstm_results['metrics']['rmse']:.2f}, MAE={lstm_results['metrics']['mae']:.2f}, R2={lstm_results['metrics']['r2']:.2f}")
    except ImportError:
        print("LSTM模型: TensorFlow未安装，跳过LSTM预测")
    
    # 比较模型
    print("\n=== 模型比较 ===")
    comparison_df = analyzer.compare_models()
    
    # 保存结果
    print("\n=== 保存结果 ===")
    analyzer.save_data('processed_data.csv')
    analyzer.save_evaluation_results('evaluation_results.json')
    
    # 保存ARIMA模型
    if 'arima_1_1_1' in analyzer.models:
        analyzer.save_model('arima_1_1_1', 'arima_model.joblib')
    
    print("\n所有演示完成！")