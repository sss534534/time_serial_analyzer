#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：抓取真实股价数据并进行回归验证
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_price_predictor import StockPricePredictor

def test_real_stock_data():
    """
    测试真实股价数据抓取和回归验证功能
    """
    print("=" * 60)
    print("测试脚本：抓取真实股价数据并进行回归验证")
    print("=" * 60)
    
    try:
        # 1. 创建预测器实例
        predictor = StockPricePredictor()
        
        # 2. 从Yahoo Finance获取真实股价数据
        print("\n1. 从Yahoo Finance获取真实股价数据...")
        ticker = 'AAPL'  # 苹果公司股票代码
        start_date = '2020-01-01'
        end_date = '2025-12-03'
        
        predictor.fetch_yfinance_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            frequency='D'
        )
        
        print(f"✓ 数据获取成功")
        print(f"   股票代码: {ticker}")
        print(f"   数据规模: {len(predictor.data)} 行")
        print(f"   时间范围: {predictor.data.index.min()} 到 {predictor.data.index.max()}")
        print(f"   数据列: {list(predictor.data.columns)}")
        
        # 3. 数据预处理
        print("\n2. 数据预处理...")
        predictor.preprocess_data(
            impute_method='mean',
            scaler_type='minmax'
        )
        print(f"✓ 数据预处理完成")
        
        # 4. 运行多种预测策略
        print("\n3. 运行预测策略...")
        strategies = [
            {
                'algo': 'moving_average',
                'params': {'window_size': 5, 'plot_forecast': True}
            },
            {
                'algo': 'moving_average',
                'params': {'window_size': 10, 'plot_forecast': True}
            },
            {
                'algo': 'arima',
                'params': {'auto_tune': True, 'plot_forecast': True}
            },
            {
                'algo': 'lstm',
                'params': {
                    'look_back': 30, 
                    'epochs': 30, 
                    'batch_size': 32,
                    'test_size': 0.2,
                    'plot_forecast': True
                }
            },
            {
                'algo': 'lstm',
                'params': {
                    'look_back': 30, 
                    'epochs': 30, 
                    'batch_size': 32,
                    'test_size': 0.2,
                    'plot_forecast': True,
                    'complex_model': True
                }
            }
        ]
        
        results = predictor.run_strategies(strategies)
        print(f"✓ 预测策略运行完成，共运行 {len(results)} 种策略")
        
        # 5. 选择最佳模型
        print("\n4. 选择最佳模型...")
        best_model_name, best_metrics = predictor.find_best_model(metric='rmse')
        print(f"✓ 最佳模型选择完成: {best_model_name}")
        
        # 6. 进行回归验证
        print("\n5. 进行回归验证...")
        validation_results = predictor.regression_validation()
        print(f"✓ 回归验证完成")
        
        # 7. 使用最佳模型预测未来股价
        print("\n6. 预测未来股价...")
        future_days = 15
        future_forecast = predictor.predict_with_best_model(steps=future_days)
        print(f"✓ 未来 {future_days} 天股价预测完成")
        print("\n未来15天股价预测结果:")
        print(future_forecast.head(10))
        
        # 8. 保存结果
        print("\n7. 保存结果...")
        output_dir = f'./test_results_{ticker}'
        predictor.save_results(output_dir=output_dir)
        print(f"✓ 结果保存完成，输出目录: {output_dir}")
        
        # 8. 可视化真实数据和预测结果
        print("\n8. 可视化结果...")
        plt.figure(figsize=(15, 8))
        
        # 绘制真实股价
        plt.subplot(2, 1, 1)
        plt.plot(predictor.data.index, predictor.data[predictor.price_column], label='真实股价')
        plt.title(f'{ticker} 股价历史走势 (2020-2023)')
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        
        # 绘制未来预测
        plt.subplot(2, 1, 2)
        plt.plot(predictor.data.index, predictor.data[predictor.price_column], label='真实股价')
        plt.plot(future_forecast.index, future_forecast['forecast_price'], 
                 label='未来预测', color='red', linestyle='--')
        plt.title(f'{ticker} 未来 {future_days} 天股价预测')
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/forecast_chart.png')
        plt.show()
        
        print("\n" + "=" * 60)
        print("测试完成！所有功能验证成功。")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_real_stock_data()