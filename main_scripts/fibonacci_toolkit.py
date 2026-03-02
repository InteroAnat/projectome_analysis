"""
斐波那契回撤分析工具包 (Fibonacci Retracement Toolkit)
用于金融市场技术分析，支持股票、期货、外汇等品种
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance未安装，在线数据获取功能不可用")


class FibonacciAnalyzer:
    """
    斐波那契回撤分析器

    核心功能:
    1. 计算斐波那契回撤水平
    2. 识别支撑阻力位
    3. 生成交易信号
    4. 可视化分析图表
    """

    # 标准斐波那契回撤比例
    FIB_LEVELS = {
        '0%': 0.0,
        '23.6%': 0.236,
        '38.2%': 0.382,
        '50%': 0.5,
        '61.8%': 0.618,
        '78.6%': 0.786,
        '100%': 1.0
    }

    # 颜色映射
    LEVEL_COLORS = {
        '0%': '#FF0000',
        '23.6%': '#FF6600',
        '38.2%': '#FF9900',
        '50%': '#CCCC00',
        '61.8%': '#00CC00',
        '78.6%': '#0099FF',
        '100%': '#0000FF'
    }

    def __init__(self, data: pd.DataFrame = None):
        """
        初始化分析器

        参数:
        data: DataFrame包含价格数据，需有'High', 'Low', 'Close'列
        """
        self.data = data
        self.fib_levels = {}
        self.swing_high = None
        self.swing_low = None

    def fetch_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """
        从Yahoo Finance获取数据

        参数:
        symbol: 股票代码 (如 'GC=F' 黄金, 'AAPL' 苹果)
        period: 时间周期 ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: 数据间隔 ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        返回:
        DataFrame包含价格数据
        """
        if not YFINANCE_AVAILABLE:
            print("❌ yfinance未安装，无法获取在线数据")
            return None

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                print(f"❌ 未获取到 {symbol} 的数据")
                return None

            df.reset_index(inplace=True)

            # 标准化列名
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

            # 确保必要的列存在
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            for col in required_cols:
                if col not in df.columns:
                    print(f"❌ 数据缺少必要列: {col}")
                    return None

            self.data = df
            print(f"✅ 成功获取 {symbol} 数据: {len(df)} 条记录")
            print(f"   时间范围: {df['Date'].min().strftime('%Y-%m-%d')} 至 {df['Date'].max().strftime('%Y-%m-%d')}")
            return df
        except Exception as e:
            print(f"❌ 获取数据失败: {e}")
            return None

    def load_from_csv(self, filepath: str, date_col: str = 'Date') -> pd.DataFrame:
        """
        从CSV文件加载数据

        参数:
        filepath: CSV文件路径
        date_col: 日期列名

        返回:
        DataFrame包含价格数据
        """
        try:
            df = pd.read_csv(filepath)
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)

            # 确保必要的列存在
            required_cols = ['High', 'Low', 'Close']
            for col in required_cols:
                if col not in df.columns:
                    print(f"❌ CSV数据缺少必要列: {col}")
                    return None

            self.data = df
            print(f"✅ 成功从CSV加载数据: {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"❌ 加载CSV失败: {e}")
            return None

    def calculate_fibonacci(self, high: float = None, low: float = None) -> Dict[str, float]:
        """
        计算斐波那契回撤水平

        参数:
        high: 最高价 (默认使用数据中的最高价)
        low: 最低价 (默认使用数据中的最低价)

        返回:
        包含斐波那契水平的字典
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        if high is None:
            high = self.data['High'].max()
        if low is None:
            low = self.data['Low'].min()

        self.swing_high = high
        self.swing_low = low
        diff = high - low

        self.fib_levels = {}
        for level_name, ratio in self.FIB_LEVELS.items():
            price = high - ratio * diff
            self.fib_levels[level_name] = price

        return self.fib_levels

    def get_current_zone(self, current_price: float = None) -> Dict:
        """
        获取当前价格所在的斐波那契区间

        参数:
        current_price: 当前价格 (默认使用最新收盘价)

        返回:
        包含区间信息的字典
        """
        if current_price is None:
            current_price = self.data['Close'].iloc[-1]

        if not self.fib_levels:
            self.calculate_fibonacci()

        levels = list(self.fib_levels.items())

        for i in range(len(levels) - 1):
            upper_level, upper_price = levels[i]
            lower_level, lower_price = levels[i + 1]

            if lower_price <= current_price <= upper_price:
                return {
                    'current_price': current_price,
                    'upper_level': upper_level,
                    'upper_price': upper_price,
                    'lower_level': lower_level,
                    'lower_price': lower_price,
                    'zone': f"{upper_level}-{lower_level}",
                    'position_in_zone': (upper_price - current_price) / (upper_price - lower_price)
                }

        # 如果价格超出范围
        if current_price > self.swing_high:
            return {
                'current_price': current_price,
                'status': 'above_high',
                'message': '价格高于52周高点，处于突破状态'
            }
        else:
            return {
                'current_price': current_price,
                'status': 'below_low',
                'message': '价格低于52周低点，处于超跌状态'
            }

    def generate_signals(self, current_price: float = None) -> Dict:
        """
        生成交易信号

        参数:
        current_price: 当前价格

        返回:
        包含交易信号的字典
        """
        if current_price is None:
            current_price = self.data['Close'].iloc[-1]

        zone_info = self.get_current_zone(current_price)
        signals = {
            'price': current_price,
            'zone': zone_info.get('zone', 'N/A'),
            'actions': []
        }

        # 根据所在区间生成信号
        if 'zone' in zone_info:
            zone = zone_info['zone']

            if zone == '0%-23.6%':
                signals['actions'] = [
                    {'type': 'hold', 'desc': '持有多单，止损设于23.6%下方'},
                    {'type': 'wait', 'desc': '等待回调至23.6%再买入'}
                ]
                signals['trend'] = 'strong_bullish'
            elif zone == '23.6%-38.2%':
                signals['actions'] = [
                    {'type': 'buy', 'desc': '理想买入区，38.2%附近建仓'},
                    {'type': 'hold', 'desc': '持有多单，止损设于50%下方'}
                ]
                signals['trend'] = 'bullish'
            elif zone == '38.2%-50%':
                signals['actions'] = [
                    {'type': 'cautious_buy', 'desc': '谨慎买入，控制仓位'},
                    {'type': 'reduce', 'desc': '考虑减仓部分获利仓位'}
                ]
                signals['trend'] = 'neutral'
            elif zone == '50%-61.8%':
                signals['actions'] = [
                    {'type': 'watch', 'desc': '观望为主，等待企稳信号'},
                    {'type': 'stop_loss', 'desc': '跌破61.8%必须止损'}
                ]
                signals['trend'] = 'bearish'
            else:
                signals['actions'] = [
                    {'type': 'avoid', 'desc': '避免买入，趋势可能反转'},
                    {'type': 'stop_loss', 'desc': '严格止损，控制亏损'}
                ]
                signals['trend'] = 'strong_bearish'

        return signals

    def plot_analysis(self, save_path: str = None, figsize: Tuple = (16, 10)) -> plt.Figure:
        """
        绘制斐波那契分析图表

        参数:
        save_path: 保存路径
        figsize: 图表大小

        返回:
        matplotlib Figure对象
        """
        if self.data is None or len(self.data) == 0:
            print("❌ 无数据可供绘图")
            return None

        if not self.fib_levels:
            self.calculate_fibonacci()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]})

        # 绘制价格线
        ax1.plot(self.data['Date'], self.data['Close'], 
                label='收盘价', color='#FFD700', linewidth=2)
        ax1.fill_between(self.data['Date'], self.data['Low'], self.data['High'], 
                        alpha=0.2, color='#FFD700', label='日内波动')

        # 绘制斐波那契水平线
        for level, price in self.fib_levels.items():
            color = self.LEVEL_COLORS.get(level, 'gray')
            ax1.axhline(y=price, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
            ax1.text(self.data['Date'].iloc[-1], price, f' {level}: ${price:.0f}', 
                    fontsize=9, va='center', color=color, fontweight='bold')

        # 标记高低点
        max_idx = self.data['High'].idxmax()
        min_idx = self.data['Low'].idxmin()
        ax1.scatter(self.data.loc[max_idx, 'Date'], self.data.loc[max_idx, 'High'], 
                   color='red', s=80, zorder=5, label=f'高点: ${self.swing_high:.0f}')
        ax1.scatter(self.data.loc[min_idx, 'Date'], self.data.loc[min_idx, 'Low'], 
                   color='blue', s=80, zorder=5, label=f'低点: ${self.swing_low:.0f}')

        # 设置图表属性
        ax1.set_title('斐波那契回撤分析', fontsize=16, fontweight='bold')
        ax1.set_ylabel('价格', fontsize=12)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(self.swing_low * 0.95, self.swing_high * 1.05)

        # 绘制成交量
        if 'Volume' in self.data.columns:
            ax2.bar(self.data['Date'], self.data['Volume'], 
                   color='gray', alpha=0.6, width=1)
            ax2.set_title('成交量', fontsize=12)
            ax2.set_xlabel('日期', fontsize=12)
            ax2.set_ylabel('成交量', fontsize=12)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 图表已保存至: {save_path}")

        return fig

    def get_support_resistance(self) -> Dict:
        """
        获取支撑阻力位

        返回:
        包含支撑阻力位的字典
        """
        if not self.fib_levels:
            self.calculate_fibonacci()

        current_price = self.data['Close'].iloc[-1]

        # 找到最近的支撑和阻力
        levels_above = [(k, v) for k, v in self.fib_levels.items() if v > current_price]
        levels_below = [(k, v) for k, v in self.fib_levels.items() if v < current_price]

        resistance = min(levels_above, key=lambda x: x[1]) if levels_above else None
        support = max(levels_below, key=lambda x: x[1]) if levels_below else None

        return {
            'current_price': current_price,
            'resistance_level': resistance[0] if resistance else None,
            'resistance_price': resistance[1] if resistance else None,
            'support_level': support[0] if support else None,
            'support_price': support[1] if support else None,
            'all_levels': self.fib_levels
        }

    def export_report(self, filepath: str):
        """
        导出分析报告

        参数:
        filepath: 报告保存路径
        """
        if not self.fib_levels:
            self.calculate_fibonacci()

        current_price = self.data['Close'].iloc[-1]
        zone_info = self.get_current_zone(current_price)
        signals = self.generate_signals(current_price)
        sr_levels = self.get_support_resistance()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("斐波那契回撤分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write("📊 价格信息\n")
            f.write("-" * 80 + "\n")
            f.write(f"当前价格: ${current_price:.2f}\n")
            f.write(f"52周高点: ${self.swing_high:.2f}\n")
            f.write(f"52周低点: ${self.swing_low:.2f}\n")
            f.write(f"波动区间: ${self.swing_high - self.swing_low:.2f}\n\n")

            f.write("📐 斐波那契水平\n")
            f.write("-" * 80 + "\n")
            for level, price in self.fib_levels.items():
                f.write(f"{level:>6}: ${price:>10.2f}\n")
            f.write("\n")

            f.write("🎯 支撑阻力\n")
            f.write("-" * 80 + "\n")
            f.write(f"最近阻力位: {sr_levels['resistance_level']} (${sr_levels['resistance_price']:.2f})\n")
            f.write(f"最近支撑位: {sr_levels['support_level']} (${sr_levels['support_price']:.2f})\n\n")

            f.write("💡 交易建议\n")
            f.write("-" * 80 + "\n")
            for action in signals['actions']:
                f.write(f"• {action['desc']}\n")

        print(f"✅ 报告已导出至: {filepath}")


def quick_analysis(symbol: str = None, csv_path: str = None, period: str = '1y', save_chart: str = None) -> FibonacciAnalyzer:
    """
    快速分析函数

    参数:
    symbol: 股票代码 (优先使用在线数据)
    csv_path: CSV文件路径 (当symbol为None时使用)
    period: 时间周期
    save_chart: 图表保存路径

    返回:
    FibonacciAnalyzer实例
    """
    analyzer = FibonacciAnalyzer()

    if symbol:
        analyzer.fetch_data(symbol, period)
    elif csv_path:
        analyzer.load_from_csv(csv_path)
    else:
        print("❌ 请提供symbol或csv_path")
        return None

    if analyzer.data is None:
        return None

    analyzer.calculate_fibonacci()

    current_price = analyzer.data['Close'].iloc[-1]
    zone = analyzer.get_current_zone(current_price)
    signals = analyzer.generate_signals(current_price)

    print("\n" + "=" * 80)
    print(f"🎯 斐波那契分析结果")
    print("=" * 80)
    print(f"当前价格: ${current_price:.2f}")
    print(f"所在区间: {zone.get('zone', 'N/A')}")
    print(f"趋势判断: {signals.get('trend', 'N/A')}")
    print("\n交易建议:")
    for action in signals['actions']:
        print(f"  • {action['desc']}")

    if save_chart:
        analyzer.plot_analysis(save_chart)

    return analyzer


if __name__ == '__main__':
    # 示例: 从CSV分析黄金
    analyzer = quick_analysis(csv_path='gold_prices.csv', save_chart='gold_fib_chart.png')
