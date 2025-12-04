# coding: utf-8
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from stock_price_predictor import StockPricePredictor


def test_stock_price_predictor():
    """
    测试股价预测器功能
    """
    print("=" * 60)
    print("股价预测框架测试")
    print("=" * 60)
    
    # 创建股价预测器
    predictor = StockPricePredictor()
    
    # 加载数据
    try:
        predictor.load_data(
            file_path='./sample_data.csv',
            time_column='date',
            price_column='value',
            frequency='D'
        )
        print("✓ 数据加载成功")
        print(f"数据规模: {len(predictor.data)} 行")
        print(f"数据时间范围: {predictor.data.index.min()} 到 {predictor.data.index.max()}")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False
    
    # 数据预处理
    try:
        predictor.preprocess_data()
        print("✓ 数据预处理完成")
    except Exception as e:
        print(f"✗ 数据预处理失败: {e}")
        return False
    
    # 定义简化的预测策略（减少运行时间）
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
        }
    ]
    
    # 运行预测策略
    try:
        print("\n开始运行预测策略...")
        results = predictor.run_strategies(strategies)
        print(f"✓ 预测策略运行完成，共运行 {len(results)} 种策略")
    except Exception as e:
        print(f"✗ 预测策略运行失败: {e}")
        return False
    
    # 检查评估结果
    if not predictor.evaluation_results:
        print("✗ 没有生成评估结果")
        return False
    
    print("\n评估结果汇总:")
    for model_name, metrics in predictor.evaluation_results.items():
        print(f"  {model_name}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%, R2={metrics['r2']:.4f}")
    
    # 找出最佳模型
    try:
        best_model_name, best_metrics = predictor.find_best_model(metric='rmse')
        print(f"✓ 最佳模型选择完成: {best_model_name}")
    except Exception as e:
        print(f"✗ 最佳模型选择失败: {e}")
        return False
    
    # 使用最佳模型预测未来股价
    try:
        print("\n使用最佳模型预测未来15天股价...")
        future_forecast = predictor.predict_with_best_model(steps=15)
        print(f"✓ 未来股价预测完成")
        print("\n未来15天股价预测（前5条）:")
        print(future_forecast.head())
    except Exception as e:
        print(f"✗ 未来股价预测失败: {e}")
        return False
    
    # 保存结果
    try:
        predictor.save_results(output_dir='./test_results')
        print("✓ 结果保存完成")
    except Exception as e:
        print(f"✗ 结果保存失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("测试全部通过！股价预测框架工作正常")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_stock_price_predictor()
