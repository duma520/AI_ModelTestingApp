import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import subprocess
from PIL import Image, ImageTk
import threading

class ModelTestingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("深度学习模型验证与测试工具 v1.0")
        self.root.geometry("900x700")
        
        # 设置紧凑布局
        self.root.option_add("*TButton*Padding", 2)
        self.root.option_add("*TLabel*Padding", 2)
        self.root.option_add("*TEntry*Padding", 2)
        
        self.create_widgets()
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 模型选择区域
        model_frame = ttk.LabelFrame(main_frame, text="模型配置", padding=(5, 5))
        model_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # 模型类型选择
        ttk.Label(model_frame, text="模型类型:").grid(row=0, column=0, sticky=tk.W)
        self.model_type = tk.StringVar(value="YOLO")
        model_types = ["YOLO", "MobileNet", "自定义"]
        ttk.Combobox(model_frame, textvariable=self.model_type, values=model_types, width=10).grid(row=0, column=1, sticky=tk.W)
        
        # 框架选择
        ttk.Label(model_frame, text="推理框架:").grid(row=0, column=2, sticky=tk.W)
        self.framework = tk.StringVar(value="TensorRT")
        frameworks = ["TensorRT", "ONNX", "OpenVINO", "原生"]
        ttk.Combobox(model_frame, textvariable=self.framework, values=frameworks, width=10).grid(row=0, column=3, sticky=tk.W)
        
        # 模型文件路径
        ttk.Label(model_frame, text="模型路径:").grid(row=1, column=0, sticky=tk.W)
        self.model_path = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.model_path, width=40).grid(row=1, column=1, columnspan=2, sticky=tk.W)
        ttk.Button(model_frame, text="浏览", command=self.browse_model).grid(row=1, column=3, sticky=tk.W)
        
        # 配置文件路径 (YOLO专用)
        self.config_label = ttk.Label(model_frame, text="配置文件:")
        self.config_label.grid(row=2, column=0, sticky=tk.W)
        self.config_path = tk.StringVar()
        self.config_entry = ttk.Entry(model_frame, textvariable=self.config_path, width=40)
        self.config_entry.grid(row=2, column=1, columnspan=2, sticky=tk.W)
        ttk.Button(model_frame, text="浏览", command=self.browse_config).grid(row=2, column=3, sticky=tk.W)
        
        # 类名文件路径
        ttk.Label(model_frame, text="类名文件:").grid(row=3, column=0, sticky=tk.W)
        self.classes_path = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.classes_path, width=40).grid(row=3, column=1, columnspan=2, sticky=tk.W)
        ttk.Button(model_frame, text="浏览", command=self.browse_classes).grid(row=3, column=3, sticky=tk.W)
        
        # 输入配置区域
        input_frame = ttk.LabelFrame(main_frame, text="输入配置", padding=(5, 5))
        input_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # 输入类型
        ttk.Label(input_frame, text="输入类型:").grid(row=0, column=0, sticky=tk.W)
        self.input_type = tk.StringVar(value="图像")
        input_types = ["图像", "视频", "摄像头", "目录"]
        ttk.Combobox(input_frame, textvariable=self.input_type, values=input_types, width=10).grid(row=0, column=1, sticky=tk.W)
        
        # 输入路径
        ttk.Label(input_frame, text="输入路径:").grid(row=1, column=0, sticky=tk.W)
        self.input_path = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_path, width=40).grid(row=1, column=1, columnspan=2, sticky=tk.W)
        ttk.Button(input_frame, text="浏览", command=self.browse_input).grid(row=1, column=3, sticky=tk.W)
        
        # 摄像头索引
        self.cam_index_label = ttk.Label(input_frame, text="摄像头索引:")
        self.cam_index_label.grid(row=2, column=0, sticky=tk.W)
        self.cam_index = tk.StringVar(value="0")
        ttk.Entry(input_frame, textvariable=self.cam_index, width=10).grid(row=2, column=1, sticky=tk.W)
        
        # 高级参数区域
        advanced_frame = ttk.LabelFrame(main_frame, text="高级参数", padding=(5, 5))
        advanced_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # 输入尺寸
        ttk.Label(advanced_frame, text="输入尺寸:").grid(row=0, column=0, sticky=tk.W)
        self.input_width = tk.StringVar(value="640")
        ttk.Entry(advanced_frame, textvariable=self.input_width, width=6).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(advanced_frame, text="x").grid(row=0, column=2)
        self.input_height = tk.StringVar(value="640")
        ttk.Entry(advanced_frame, textvariable=self.input_height, width=6).grid(row=0, column=3, sticky=tk.W)
        
        # 置信度阈值
        ttk.Label(advanced_frame, text="置信度阈值:").grid(row=0, column=4, sticky=tk.W, padx=(10, 0))
        self.conf_thres = tk.StringVar(value="0.5")
        ttk.Entry(advanced_frame, textvariable=self.conf_thres, width=6).grid(row=0, column=5, sticky=tk.W)
        
        # NMS阈值
        ttk.Label(advanced_frame, text="NMS阈值:").grid(row=0, column=6, sticky=tk.W, padx=(10, 0))
        self.nms_thres = tk.StringVar(value="0.4")
        ttk.Entry(advanced_frame, textvariable=self.nms_thres, width=6).grid(row=0, column=7, sticky=tk.W)
        
        # FP16推理
        self.fp16 = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_frame, text="FP16推理", variable=self.fp16).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        # INT8量化
        self.int8 = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_frame, text="INT8量化", variable=self.int8).grid(row=1, column=2, columnspan=2, sticky=tk.W)
        
        # 批处理大小
        ttk.Label(advanced_frame, text="批处理大小:").grid(row=1, column=4, sticky=tk.W)
        self.batch_size = tk.StringVar(value="1")
        ttk.Entry(advanced_frame, textvariable=self.batch_size, width=6).grid(row=1, column=5, sticky=tk.W)
        
        # 设备选择
        ttk.Label(advanced_frame, text="设备:").grid(row=1, column=6, sticky=tk.W)
        self.device = tk.StringVar(value="0")
        ttk.Entry(advanced_frame, textvariable=self.device, width=6).grid(row=1, column=7, sticky=tk.W)
        
        # 输出配置区域
        output_frame = ttk.LabelFrame(main_frame, text="输出配置", padding=(5, 5))
        output_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # 输出路径
        ttk.Label(output_frame, text="输出路径:").grid(row=0, column=0, sticky=tk.W)
        self.output_path = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path, width=40).grid(row=0, column=1, columnspan=2, sticky=tk.W)
        ttk.Button(output_frame, text="浏览", command=self.browse_output).grid(row=0, column=3, sticky=tk.W)
        
        # 显示结果
        self.show_result = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="显示结果", variable=self.show_result).grid(row=1, column=0, sticky=tk.W)
        
        # 保存结果
        self.save_result = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="保存结果", variable=self.save_result).grid(row=1, column=1, sticky=tk.W)
        
        # 性能分析
        self.profile = tk.BooleanVar(value=False)
        ttk.Checkbutton(output_frame, text="性能分析", variable=self.profile).grid(row=1, column=2, sticky=tk.W)
        
        # 日志级别
        ttk.Label(output_frame, text="日志级别:").grid(row=1, column=3, sticky=tk.W)
        self.log_level = tk.StringVar(value="INFO")
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        ttk.Combobox(output_frame, textvariable=self.log_level, values=log_levels, width=10).grid(row=1, column=4, sticky=tk.W)
        
        # 控制按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="验证模型", command=self.validate_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="测试模型", command=self.test_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="导出模型", command=self.export_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="性能测试", command=self.benchmark_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="清空日志", command=self.clear_log).pack(side=tk.RIGHT, padx=2)
        
        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="日志输出", padding=(5, 5))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, padx=5, pady=2)
        
        # 绑定事件
        self.model_type.trace_add("write", self.update_ui)
        self.input_type.trace_add("write", self.update_ui)
        
        # 初始化UI状态
        self.update_ui()
    
    def update_ui(self, *args):
        # 根据模型类型显示/隐藏配置选项
        if self.model_type.get() == "YOLO":
            self.config_label.grid()
            self.config_entry.grid()
        else:
            self.config_label.grid_remove()
            self.config_entry.grid_remove()
        
        # 根据输入类型显示/隐藏摄像头索引
        if self.input_type.get() == "摄像头":
            self.cam_index_label.grid()
            self.cam_index_label.grid()
        else:
            self.cam_index_label.grid_remove()
            self.cam_index_label.grid_remove()
    
    def browse_model(self):
        filetypes = [
            ("模型文件", "*.pt *.onnx *.engine *.xml *.bin *.pb *.h5"),
            ("所有文件", "*.*")
        ]
        filename = filedialog.askopenfilename(title="选择模型文件", filetypes=filetypes)
        if filename:
            self.model_path.set(filename)
    
    def browse_config(self):
        filetypes = [
            ("配置文件", "*.cfg *.yaml *.yml"),
            ("所有文件", "*.*")
        ]
        filename = filedialog.askopenfilename(title="选择配置文件", filetypes=filetypes)
        if filename:
            self.config_path.set(filename)
    
    def browse_classes(self):
        filetypes = [
            ("文本文件", "*.txt *.names"),
            ("所有文件", "*.*")
        ]
        filename = filedialog.askopenfilename(title="选择类名文件", filetypes=filetypes)
        if filename:
            self.classes_path.set(filename)
    
    def browse_input(self):
        input_type = self.input_type.get()
        if input_type == "图像":
            filetypes = [
                ("图像文件", "*.jpg *.jpeg *.png *.bmp"),
                ("所有文件", "*.*")
            ]
            filename = filedialog.askopenfilename(title="选择输入图像", filetypes=filetypes)
            if filename:
                self.input_path.set(filename)
        elif input_type == "视频":
            filetypes = [
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                ("所有文件", "*.*")
            ]
            filename = filedialog.askopenfilename(title="选择输入视频", filetypes=filetypes)
            if filename:
                self.input_path.set(filename)
        elif input_type == "目录":
            dirname = filedialog.askdirectory(title="选择输入目录")
            if dirname:
                self.input_path.set(dirname)
    
    def browse_output(self):
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_path.set(dirname)
    
    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.status_var.set(message)
        self.root.update()
    
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        self.status_var.set("日志已清空")
    
    def validate_model(self):
        if not self.model_path.get():
            messagebox.showerror("错误", "请先选择模型文件")
            return
        
        self.log_message("开始验证模型...")
        
        # 在实际应用中，这里应该调用模型验证代码
        # 这里只是模拟
        def validate_thread():
            try:
                # 模拟验证过程
                import time
                time.sleep(2)
                
                self.log_message("模型验证成功")
                self.log_message(f"模型类型: {self.model_type.get()}")
                self.log_message(f"推理框架: {self.framework.get()}")
                self.log_message(f"输入尺寸: {self.input_width.get()}x{self.input_height.get()}")
            except Exception as e:
                self.log_message(f"模型验证失败: {str(e)}")
        
        threading.Thread(target=validate_thread, daemon=True).start()
    
    def test_model(self):
        if not self.model_path.get():
            messagebox.showerror("错误", "请先选择模型文件")
            return
        
        input_type = self.input_type.get()
        if input_type != "摄像头" and not self.input_path.get():
            messagebox.showerror("错误", "请先选择输入源")
            return
        
        self.log_message("开始测试模型...")
        
        # 在实际应用中，这里应该调用模型测试代码
        # 这里只是模拟
        def test_thread():
            try:
                # 模拟测试过程
                import time
                for i in range(1, 6):
                    time.sleep(1)
                    self.log_message(f"处理中... {i*20}%")
                
                self.log_message("模型测试完成")
                self.log_message(f"输入类型: {input_type}")
                if input_type == "摄像头":
                    self.log_message(f"摄像头索引: {self.cam_index.get()}")
                else:
                    self.log_message(f"输入路径: {self.input_path.get()}")
            except Exception as e:
                self.log_message(f"模型测试失败: {str(e)}")
        
        threading.Thread(target=test_thread, daemon=True).start()
    
    def export_model(self):
        if not self.model_path.get():
            messagebox.showerror("错误", "请先选择模型文件")
            return
        
        self.log_message("开始导出模型...")
        
        # 在实际应用中，这里应该调用模型导出代码
        # 这里只是模拟
        def export_thread():
            try:
                # 模拟导出过程
                import time
                time.sleep(3)
                
                self.log_message("模型导出成功")
                framework = self.framework.get()
                if framework == "TensorRT":
                    self.log_message("导出为TensorRT引擎")
                elif framework == "ONNX":
                    self.log_message("导出为ONNX格式")
                elif framework == "OpenVINO":
                    self.log_message("导出为OpenVINO IR格式")
            except Exception as e:
                self.log_message(f"模型导出失败: {str(e)}")
        
        threading.Thread(target=export_thread, daemon=True).start()
    
    def benchmark_model(self):
        if not self.model_path.get():
            messagebox.showerror("错误", "请先选择模型文件")
            return
        
        self.log_message("开始性能测试...")
        
        # 在实际应用中，这里应该调用性能测试代码
        # 这里只是模拟
        def benchmark_thread():
            try:
                # 模拟性能测试过程
                import time
                import random
                time.sleep(1)
                
                fps = random.uniform(30, 100)
                latency = random.uniform(5, 20)
                memory = random.uniform(500, 2000)
                
                self.log_message("性能测试完成:")
                self.log_message(f"平均FPS: {fps:.2f}")
                self.log_message(f"平均延迟: {latency:.2f} ms")
                self.log_message(f"显存占用: {memory:.2f} MB")
                
                if self.fp16.get():
                    self.log_message("FP16模式已启用")
                if self.int8.get():
                    self.log_message("INT8量化已启用")
            except Exception as e:
                self.log_message(f"性能测试失败: {str(e)}")
        
        threading.Thread(target=benchmark_thread, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelTestingApp(root)
    root.mainloop()