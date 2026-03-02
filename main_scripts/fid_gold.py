"""
斐波那契回撤分析器 - GUI版本
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import sys
import os

# 添加工具包路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fibonacci_toolkit import FibonacciAnalyzer


class FibonacciGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("斐波那契回撤分析器")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        self.analyzer = None

        self.create_widgets()

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # ===== 数据输入区域 =====
        input_frame = ttk.LabelFrame(main_frame, text="数据输入", padding="10")
        input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # CSV文件选择
        ttk.Label(input_frame, text="CSV文件:").grid(row=0, column=0, sticky=tk.W)
        self.csv_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.csv_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(input_frame, text="浏览", command=self.browse_csv).grid(row=0, column=2)
        ttk.Button(input_frame, text="加载数据", command=self.load_data).grid(row=0, column=3, padx=5)

        # 在线数据获取
        ttk.Label(input_frame, text="或输入代码:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.symbol_var = tk.StringVar(value="GC=F")
        ttk.Entry(input_frame, textvariable=self.symbol_var, width=15).grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(input_frame, text="周期:").grid(row=1, column=1, sticky=tk.E, padx=80)
        self.period_var = tk.StringVar(value="1y")
        period_combo = ttk.Combobox(input_frame, textvariable=self.period_var, width=8, 
                                   values=["1mo", "3mo", "6mo", "1y", "2y", "5y"])
        period_combo.grid(row=1, column=2, sticky=tk.W)
        ttk.Button(input_frame, text="获取在线数据", command=self.fetch_online_data).grid(row=1, column=3)

        # ===== 分析控制区域 =====
        control_frame = ttk.LabelFrame(main_frame, text="分析控制", padding="10")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(control_frame, text="计算斐波那契", command=self.calculate_fib).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="生成图表", command=self.generate_chart).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="导出报告", command=self.export_report).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="清空", command=self.clear_all).grid(row=0, column=3, padx=5)

        # ===== 结果显示区域 =====
        result_frame = ttk.LabelFrame(main_frame, text="分析结果", padding="10")
        result_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        # 结果文本框
        self.result_text = scrolledtext.ScrolledText(result_frame, width=50, height=30, wrap=tk.WORD)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ===== 图表显示区域 =====
        chart_frame = ttk.LabelFrame(main_frame, text="图表", padding="10")
        chart_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)

        self.chart_canvas = None

    def browse_csv(self):
        filename = filedialog.askopenfilename(
            title="选择CSV文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_path_var.set(filename)

    def load_data(self):
        csv_path = self.csv_path_var.get()
        if not csv_path:
            messagebox.showwarning("警告", "请选择CSV文件")
            return

        try:
            self.analyzer = FibonacciAnalyzer()
            self.analyzer.load_from_csv(csv_path)
            messagebox.showinfo("成功", f"已加载 {len(self.analyzer.data)} 条数据")
            self.show_data_summary()
        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {str(e)}")

    def fetch_online_data(self):
        symbol = self.symbol_var.get()
        period = self.period_var.get()

        if not symbol:
            messagebox.showwarning("警告", "请输入股票代码")
            return

        try:
            self.analyzer = FibonacciAnalyzer()
            self.analyzer.fetch_data(symbol, period)
            if self.analyzer.data is not None:
                messagebox.showinfo("成功", f"已获取 {symbol} 的 {len(self.analyzer.data)} 条数据")
                self.show_data_summary()
        except Exception as e:
            messagebox.showerror("错误", f"获取数据失败: {str(e)}")

    def show_data_summary(self):
        if self.analyzer is None or self.analyzer.data is None:
            return

        df = self.analyzer.data
        summary = f"""📊 数据概览
{'='*40}
数据条数: {len(df)}
时间范围: {df['Date'].min().strftime('%Y-%m-%d')} 至 {df['Date'].max().strftime('%Y-%m-%d')}

💰 价格统计
{'='*40}
最高价: ${df['High'].max():.2f}
最低价: ${df['Low'].min():.2f}
最新价: ${df['Close'].iloc[-1]:.2f}
平均价: ${df['Close'].mean():.2f}
"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, summary)

    def calculate_fib(self):
        if self.analyzer is None or self.analyzer.data is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            fib_levels = self.analyzer.calculate_fibonacci()
            current_price = self.analyzer.data['Close'].iloc[-1]
            zone_info = self.analyzer.get_current_zone(current_price)
            signals = self.analyzer.generate_signals(current_price)
            sr = self.analyzer.get_support_resistance()

            result = f"""📐 斐波那契回撤水平
{'='*40}
"""
            for level, price in fib_levels.items():
                result += f"{level:>6}: ${price:>10.2f}\n"

            result += f"""
🎯 当前位置
{'='*40}
当前价格: ${current_price:.2f}
所在区间: {zone_info.get('zone', 'N/A')}

📈 支撑阻力
{'='*40}
阻力位: {sr['resistance_level']} (${sr['resistance_price']:.2f})
支撑位: {sr['support_level']} (${sr['support_price']:.2f})

💡 交易建议
{'='*40}
趋势: {signals['trend']}
"""
            for action in signals['actions']:
                result += f"• {action['desc']}\n"

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result)

        except Exception as e:
            messagebox.showerror("错误", f"计算失败: {str(e)}")

    def generate_chart(self):
        if self.analyzer is None or self.analyzer.data is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            # 创建新窗口显示图表
            chart_window = tk.Toplevel(self.root)
            chart_window.title("斐波那契分析图表")
            chart_window.geometry("1200x800")

            # 生成图表
            fig = self.analyzer.plot_analysis(figsize=(14, 10))

            # 嵌入到tkinter
            canvas = FigureCanvasTkAgg(fig, master=chart_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # 保存按钮
            ttk.Button(chart_window, text="保存图表", 
                      command=lambda: self.save_chart(fig)).pack(pady=5)

        except Exception as e:
            messagebox.showerror("错误", f"生成图表失败: {str(e)}")

    def save_chart(self, fig):
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filename:
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            messagebox.showinfo("成功", f"图表已保存至: {filename}")

    def export_report(self):
        if self.analyzer is None or self.analyzer.data is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.analyzer.export_report(filename)
                messagebox.showinfo("成功", f"报告已导出至: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败: {str(e)}")

    def clear_all(self):
        self.analyzer = None
        self.csv_path_var.set("")
        self.result_text.delete(1.0, tk.END)
        messagebox.showinfo("提示", "已清空所有数据")


def main():
    root = tk.Tk()
    app = FibonacciGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()