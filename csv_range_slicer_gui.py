#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Range Slicer GUI
- CSVを読み込み
- index列/値列を選択
- matplotlibでプロット
- 範囲選択(SpanSelector)→選択区間のみCSV出力

依存: pandas, matplotlib
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import numpy as np

class CsvRangeSlicerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV Range Slicer (pandas + matplotlib)")
        self.geometry("1100x700")
        
        # State
        self.df = None
        self.current_index_name = None
        self.current_value_name = None
        self.index_is_datetime = False
        self.span = None
        self.sel_xmin = None
        self.sel_xmax = None
        
        self._build_ui()
    
    def _build_ui(self):
        # Top controls
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
        
        btn_open = ttk.Button(top, text="CSVを開く...", command=self.open_csv)
        btn_open.pack(side=tk.LEFT, padx=4)
        
        ttk.Label(top, text="index列:").pack(side=tk.LEFT, padx=(12,4))
        self.index_cmb = ttk.Combobox(top, state="readonly", width=24, values=[])
        self.index_cmb.bind("<<ComboboxSelected>>", lambda e: self.on_index_change())
        self.index_cmb.pack(side=tk.LEFT)
        
        self.chk_parse_dt_var = tk.BooleanVar(value=True)
        self.chk_parse_dt = ttk.Checkbutton(top, text="indexを日時として解釈(可能なら)", variable=self.chk_parse_dt_var, command=self.on_index_change)
        self.chk_parse_dt.pack(side=tk.LEFT, padx=8)
        
        ttk.Label(top, text="値列:").pack(side=tk.LEFT, padx=(12,4))
        self.value_cmb = ttk.Combobox(top, state="readonly", width=24, values=[])
        self.value_cmb.bind("<<ComboboxSelected>>", lambda e: self.redraw_plot())
        self.value_cmb.pack(side=tk.LEFT)
        
        btn_plot = ttk.Button(top, text="プロット更新", command=self.redraw_plot)
        btn_plot.pack(side=tk.LEFT, padx=8)
        
        btn_export = ttk.Button(top, text="選択区間をCSV出力...", command=self.export_csv)
        btn_export.pack(side=tk.RIGHT, padx=4)
        
        # Selection info
        self.sel_label = ttk.Label(self, text="選択区間: 未選択", anchor="w")
        self.sel_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0,6))
        
        # Matplotlib Figure
        fig = Figure(figsize=(6,4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("グラフ (CSVを読み込んでください)")
        self.ax.set_xlabel("index")
        self.ax.set_ylabel("value")
        
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        
        # Span selector (created after data is plotted)
        self._create_span_selector()
    
    def _create_span_selector(self):
        # Remove existing span if any
        if self.span is not None:
            try:
                self.span.disconnect_events()
            except Exception:
                pass
            self.span = None
        
        def onselect(xmin, xmax):
            # Normalize xmin/xmax (ensure xmin <= xmax)
            if xmin is None or xmax is None:
                return
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            self.sel_xmin, self.sel_xmax = xmin, xmax
            self._update_sel_label()
        
        # Use SpanSelector on current axes
        self.span = SpanSelector(self.ax, onselect, direction="horizontal", useblit=True,
                                 props=dict(alpha=0.2), interactive=True, drag_from_anywhere=True)
        self.canvas.draw_idle()
    
    def _update_sel_label(self):
        if self.sel_xmin is None or self.sel_xmax is None:
            self.sel_label.config(text="選択区間: 未選択")
            return
        
        if self.index_is_datetime:
            # Convert matplotlib date numbers to pandas Timestamps for display
            import matplotlib.dates as mdates
            ts_min = pd.to_datetime(mdates.num2date(self.sel_xmin))
            ts_max = pd.to_datetime(mdates.num2date(self.sel_xmax))
            self.sel_label.config(text=f"選択区間: {ts_min} 〜 {ts_max}")
        else:
            self.sel_label.config(text=f"選択区間: {self.sel_xmin:.6g} 〜 {self.sel_xmax:.6g}")
    
    def open_csv(self):
        path = filedialog.askopenfilename(
            title="CSVファイルを選択",
            filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path:
            return
        try:
            # Basic load (do not set index yet)
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("読み込みエラー", f"CSVを読み込めませんでした:\n{e}")
            return
        
        if df.empty:
            messagebox.showwarning("空のCSV", "データがありません。")
            return
        
        self.df = df
        cols = list(df.columns)
        
        # 推定: index候補は1列目
        default_index = cols[0] if cols else None
        # 値候補は index 以外の最初の数値列
        default_value = None
        for c in cols:
            if c != default_index and pd.api.types.is_numeric_dtype(df[c]):
                default_value = c
                break
        if default_value is None and len(cols) >= 2:
            default_value = cols[1]
        
        # Set comboboxes
        self.index_cmb["values"] = cols
        self.value_cmb["values"] = cols
        if default_index:
            self.index_cmb.set(default_index)
        if default_value:
            self.value_cmb.set(default_value)
        
        # Reset selection
        self.sel_xmin = self.sel_xmax = None
        self._update_sel_label()
        
        self.on_index_change()
    
    def on_index_change(self):
        if self.df is None:
            return
        idx_name = self.index_cmb.get()
        if not idx_name:
            return
        
        # Try to parse datetime if checked
        series = self.df[idx_name].copy()
        self.index_is_datetime = False
        if self.chk_parse_dt_var.get():
            # Try datetime parsing; if most values succeed, use it
            try:
                parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, utc=False)
                valid_ratio = parsed.notna().mean()
                if valid_ratio > 0.8:
                    series = parsed
                    self.index_is_datetime = True
            except Exception:
                pass
        
        # Set as index on a working frame for plotting
        tmp = self.df.copy()
        tmp.index = series
        self.current_index_name = idx_name
        
        # Keep for plotting/export
        self._working_df = tmp
        
        # Redraw
        self.redraw_plot()
    
    def redraw_plot(self):
        if self.df is None:
            return
        idx_name = self.current_index_name or self.index_cmb.get()
        val_name = self.value_cmb.get()
        if not idx_name or not val_name:
            return
        
        if val_name not in self._working_df.columns:
            messagebox.showwarning("列が見つかりません", f"値列 '{val_name}' がデータフレームに存在しません。")
            return
        
        y = self._working_df[val_name]
        
        # Clear plot
        self.ax.clear()
        self.ax.set_title(f"{val_name} vs {idx_name}")
        self.ax.set_xlabel(idx_name + (" (datetime)" if self.index_is_datetime else ""))
        self.ax.set_ylabel(val_name)
        
        if self.index_is_datetime:
            # matplotlib handles datetime index natively
            self.ax.plot(self._working_df.index, y)
        else:
            # try numeric conversion for x; if fails, use positional index
            try:
                x = pd.to_numeric(self._working_df.index, errors="coerce")
                if x.notna().mean() > 0.8:
                    self.ax.plot(x, y)
                else:
                    self.ax.plot(np.arange(len(y)), y)
                    self.ax.set_xlabel(f"{idx_name} (非数値: 順序インデックス表示)")
            except Exception:
                self.ax.plot(np.arange(len(y)), y)
                self.ax.set_xlabel(f"{idx_name} (順序インデックス表示)")
        
        self.canvas.draw_idle()
        
        # Re-create span selector for fresh axes
        self._create_span_selector()
        # Reset selection
        self.sel_xmin = self.sel_xmax = None
        self._update_sel_label()
    
    def export_csv(self):
        if self.df is None:
            messagebox.showinfo("CSV未読込", "先にCSVを読み込んでください。")
            return
        if self.sel_xmin is None or self.sel_xmax is None:
            messagebox.showinfo("区間未選択", "エクスポートする前に、グラフ上で範囲をドラッグして選択してください。")
            return
        
        # Build sliced DataFrame by index range
        dfw = self._working_df.copy()
        # Ensure index name set for output clarity
        dfw.index.name = self.current_index_name or dfw.index.name
        
        if self.index_is_datetime:
            # Convert matplotlib date numbers to Timestamps for slicing
            import matplotlib.dates as mdates
            tmin = pd.Timestamp(mdates.num2date(self.sel_xmin))
            tmax = pd.Timestamp(mdates.num2date(self.sel_xmax))
            if tmax < tmin:
                tmin, tmax = tmax, tmin
            # Boolean mask to preserve original order even if index not sorted
            mask = (dfw.index >= tmin) & (dfw.index <= tmax)
            sliced = dfw.loc[mask]
        else:
            # Try to interpret index as numeric; fallback to positional slice by x-range over numeric-projected index
            try:
                idx_num = pd.to_numeric(dfw.index, errors="coerce")
                xmin, xmax = self.sel_xmin, self.sel_xmax
                if xmax < xmin:
                    xmin, xmax = xmax, xmin
                mask = (idx_num >= xmin) & (idx_num <= xmax)
                sliced = dfw.loc[mask]
            except Exception:
                messagebox.showerror("エクスポート失敗", "indexを数値として扱えないため、切り出しに失敗しました。数値または日時のindexを選んでください。")
                return
        
        if sliced.empty:
            messagebox.showwarning("空の結果", "選択区間にデータがありません。")
            return
        
        path = filedialog.asksaveasfilename(
            title="選択区間のCSVを書き出す",
            defaultextension=".csv",
            filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path:
            return
        
        try:
            # Reset index so the chosen index becomes明示的な列として書き出す
            out = sliced.reset_index()
            out.to_csv(path, index=False)
        except Exception as e:
            messagebox.showerror("書き出しエラー", f"CSVの書き出しに失敗しました:\n{e}")
            return
        
        messagebox.showinfo("完了", f"選択区間のCSVを書き出しました:\n{path}")
        

if __name__ == "__main__":
    app = CsvRangeSlicerApp()
    app.mainloop()
