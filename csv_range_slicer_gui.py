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
from matplotlib import font_manager
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

        self._configure_matplotlib_fonts()
        
        # State
        self.df = None
        self.current_index_name = None
        self.index_is_datetime = False
        self.span = None
        self.sel_xmin = None
        self.sel_xmax = None
        self._working_df = None
        self.value_columns = []
        self.value_visibility = {}
        self._line_by_column = {}
        self._legend_entry_by_column = {}
        self._legend_artist_map = {}
        self._legend_pick_cid = None
        self._plot_title_base = "グラフ"
        self._x_axis_mode = "numeric"
        self._x_axis_source_index = None
        self._selected_pos_range = None
        
        self._build_ui()

    def _configure_matplotlib_fonts(self):
        """Ensure matplotlib uses a font that can render Japanese labels."""
        candidates = [
            "IPAexGothic",
            "IPAPGothic",
            "TakaoPGothic",
            "Yu Gothic",
            "Hiragino Sans",
            "Hiragino Kaku Gothic ProN",
            "Meiryo",
            "Noto Sans CJK JP",
            "Noto Sans JP",
            "Source Han Sans JP",
        ]
        for name in candidates:
            try:
                font_manager.findfont(name, fallback_to_default=False)
            except Exception:
                continue
            matplotlib.rcParams["font.family"] = name
            break
        matplotlib.rcParams["axes.unicode_minus"] = False
    
    def _build_ui(self):
        # Top controls
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
        
        btn_open = ttk.Button(top, text="CSVを開く...", command=self.open_csv)
        btn_open.pack(side=tk.LEFT, padx=4)
        
        ttk.Label(top, text="横軸(インデックス)列:").pack(side=tk.LEFT, padx=(12,4))
        self.index_cmb = ttk.Combobox(top, state="readonly", width=24, values=[])
        self.index_cmb.bind("<<ComboboxSelected>>", lambda e: self.on_index_change())
        self.index_cmb.pack(side=tk.LEFT)
        
        self.chk_parse_dt_var = tk.BooleanVar(value=False)
        self.chk_parse_dt = ttk.Checkbutton(top, text="indexを日時として解釈(可能なら)", variable=self.chk_parse_dt_var, command=self.on_index_change)
        self.chk_parse_dt.pack(side=tk.LEFT, padx=8)
        
        self.value_info = ttk.Label(top, text="値列: index以外の全数値列をプロット")
        self.value_info.pack(side=tk.LEFT, padx=(12,4))
        
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

        # Enable legend interaction
        self._legend_pick_cid = self.canvas.mpl_connect("pick_event", self._on_legend_pick)
    
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
            self._selected_pos_range = None
            return

        mode = getattr(self, "_x_axis_mode", "numeric")
        xmin, xmax = self.sel_xmin, self.sel_xmax
        if mode == "datetime":
            import matplotlib.dates as mdates

            def _to_naive(ts):
                if getattr(ts, "tzinfo", None) is None:
                    return ts
                try:
                    return ts.tz_convert(None)
                except Exception:
                    try:
                        return ts.tz_localize(None)
                    except Exception:
                        return ts

            ts_min = _to_naive(pd.to_datetime(mdates.num2date(xmin)))
            ts_max = _to_naive(pd.to_datetime(mdates.num2date(xmax)))
            self._selected_pos_range = None
            self.sel_label.config(text=f"選択区間: {ts_min} 〜 {ts_max}")
        elif mode == "positional":
            index_values = self._x_axis_source_index
            if index_values is None or len(index_values) == 0:
                self._selected_pos_range = None
                self.sel_label.config(text=f"選択区間: {xmin:.6g} 〜 {xmax:.6g}")
                return
            n = len(index_values)
            start_pos = int(np.clip(np.floor(xmin + 0.5), 0, n - 1))
            end_pos = int(np.clip(np.floor(xmax + 0.5), 0, n - 1))
            if end_pos < start_pos:
                start_pos, end_pos = end_pos, start_pos
            self._selected_pos_range = (start_pos, end_pos)
            start_val = index_values[start_pos]
            end_val = index_values[end_pos]
            self.sel_label.config(
                text=f"選択区間: {start_val} 〜 {end_val} (位置 {start_pos}〜{end_pos})")
        else:
            self._selected_pos_range = None
            self.sel_label.config(text=f"選択区間: {xmin:.6g} 〜 {xmax:.6g}")
    
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
        
        # 推定: index候補はtimestampらしき列
        default_index = self._guess_timestamp_like_column(df)
        if default_index is None and cols:
            default_index = cols[0]
        # Set comboboxes
        self.index_cmb["values"] = cols
        if default_index:
            self.index_cmb.set(default_index)

        # Reset selection
        self.sel_xmin = self.sel_xmax = None
        self._update_sel_label()

        # Reset plot column info until index handling runs
        self.value_columns = []
        self.value_visibility = {}
        self._sync_value_visibility()
        self._line_by_column = {}
        self._legend_entry_by_column = {}
        self._legend_artist_map = {}
        self._x_axis_mode = "numeric"
        self._x_axis_source_index = None
        self._selected_pos_range = None
        self._update_value_info()

        self.on_index_change()

    def _guess_timestamp_like_column(self, df):
        """Return column name that looks like timestamp, otherwise None."""
        if df is None or df.empty:
            return None

        keywords = ("timestamp", "time", "date", "datetime", "ts")
        best_col = None
        best_score = -1.0

        for col in df.columns:
            series = df[col]
            name_lower = str(col).lower()
            contains_keyword = any(k in name_lower for k in keywords)

            # Skip columns that are clearly not timestamp-like
            if pd.api.types.is_numeric_dtype(series) and not contains_keyword and not pd.api.types.is_datetime64_any_dtype(series):
                continue

            valid_ratio = 0.0
            if pd.api.types.is_datetime64_any_dtype(series):
                valid_ratio = 1.0
            else:
                try:
                    parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, utc=False)
                    valid_ratio = parsed.notna().mean()
                except Exception:
                    valid_ratio = 0.0

            # Require sufficient success to avoid accidental matches
            if valid_ratio < 0.6 and not (contains_keyword and valid_ratio >= 0.4):
                continue

            score = valid_ratio
            if contains_keyword:
                score += 0.3
            if pd.api.types.is_datetime64_any_dtype(series):
                score += 0.1

            if score > best_score:
                best_score = score
                best_col = col

        return best_col

    def _select_plot_columns(self, idx_name):
        """Determine which columns should be plotted (exclude index column)."""
        if self.df is None or not idx_name:
            return []

        numeric_cols = []
        convertible_cols = []
        for col in self.df.columns:
            if col == idx_name:
                continue
            series = self.df[col]
            if pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
            else:
                try:
                    converted = pd.to_numeric(series, errors="coerce")
                except Exception:
                    continue
                if converted.notna().mean() > 0.7:
                    convertible_cols.append(col)

        if numeric_cols:
            return numeric_cols
        return convertible_cols

    def _update_value_info(self):
        if not hasattr(self, "value_info"):
            return
        if not self.value_columns:
            self.value_info.config(text="値列: プロット可能な列がありません")
            return
        visible_cols = [c for c in self.value_columns if self.value_visibility.get(c, True)]
        total = len(self.value_columns)
        visible = len(visible_cols)
        if total <= 4:
            joined = ", ".join(f"{c}{'' if self.value_visibility.get(c, True) else '(非表示)'}" for c in self.value_columns)
            self.value_info.config(text=f"値列: {joined} (凡例クリックで切替)")
        else:
            self.value_info.config(text=f"値列: {visible} / {total} 列を表示中 (凡例クリックで切替)")

    def _sync_value_visibility(self):
        if self.value_columns is None:
            self.value_visibility = {}
            return
        new_vis = {}
        for col in self.value_columns:
            new_vis[col] = self.value_visibility.get(col, True)
        self.value_visibility = new_vis
    
    def _update_legend_style(self, col):
        entry = self._legend_entry_by_column.get(col)
        if not entry:
            return
        handle, text = entry
        alpha = 1.0 if self.value_visibility.get(col, True) else 0.2
        try:
            handle.set_alpha(alpha)
        except Exception:
            pass
        try:
            text.set_alpha(alpha)
        except Exception:
            pass

    def _setup_legend_interactivity(self):
        legend = self.ax.get_legend()
        self._legend_artist_map = {}
        self._legend_entry_by_column = {}
        if legend is None:
            return
        handles = []
        if hasattr(legend, "legendHandles"):
            try:
                handles = list(legend.legendHandles)
            except Exception:
                handles = []
        if not handles and hasattr(legend, "legend_handles"):
            try:
                handles = list(legend.legend_handles)
            except Exception:
                handles = []
        if not handles:
            try:
                handles = list(legend.get_lines())
            except Exception:
                handles = []
        texts = list(legend.get_texts())
        for handle, text in zip(handles, texts):
            label = text.get_text()
            if label not in self._line_by_column:
                continue
            try:
                handle.set_picker(10)
            except Exception:
                pass
            if hasattr(handle, "set_pickradius"):
                try:
                    handle.set_pickradius(12)
                except Exception:
                    pass
            try:
                text.set_picker(10)
            except Exception:
                pass
            self._legend_artist_map[handle] = label
            self._legend_artist_map[text] = label
            self._legend_entry_by_column[label] = (handle, text)
            self._update_legend_style(label)

    def _refresh_plot_title(self, has_visible=None):
        if has_visible is None:
            has_visible = any(line.get_visible() for line in self._line_by_column.values())
        base = getattr(self, "_plot_title_base", "グラフ") or "グラフ"
        if has_visible:
            self.ax.set_title(base)
        else:
            self.ax.set_title(base + "\n(凡例をクリックで列を表示)")

    def _autoscale_to_visible(self):
        """Autoscale axes considering only currently visible lines."""
        try:
            self.ax.relim(visible_only=True)
        except TypeError:
            # Older matplotlib without visible_only argument
            self.ax.relim()
        except Exception:
            return
        try:
            self.ax.autoscale_view()
        except Exception:
            pass

    def _on_legend_pick(self, event):
        artist = getattr(event, "artist", None)
        if artist is None:
            return
        label = self._legend_artist_map.get(artist)
        if label is None and hasattr(artist, "get_label"):
            try:
                label = artist.get_label()
            except Exception:
                label = None
        if label is None or label not in self.value_columns:
            return
        self.value_visibility[label] = not self.value_visibility.get(label, True)
        line = self._line_by_column.get(label)
        if line is not None:
            line.set_visible(self.value_visibility[label])
        self._update_legend_style(label)
        self._update_value_info()
        has_visible = any(line.get_visible() for line in self._line_by_column.values())
        if has_visible:
            self._autoscale_to_visible()
        self._refresh_plot_title(has_visible)
        self.canvas.draw_idle()

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
        if idx_name in tmp.columns:
            # Drop the original index column to avoid duplicates when exporting/resetting
            tmp = tmp.drop(columns=[idx_name])
        self.current_index_name = idx_name

        # Keep for plotting/export
        self._working_df = tmp
        self.value_columns = self._select_plot_columns(idx_name)
        self._sync_value_visibility()
        self._line_by_column = {}
        self._legend_entry_by_column = {}
        self._legend_artist_map = {}
        self._selected_pos_range = None
        self._update_value_info()

        # Redraw
        self.redraw_plot()
    
    def redraw_plot(self):
        if self.df is None:
            return
        idx_name = self.current_index_name or self.index_cmb.get()
        if not idx_name:
            return

        if self._working_df is None:
            return

        if not self.value_columns:
            self.ax.clear()
            xlabel = idx_name + (" (datetime)" if self.index_is_datetime else "")
            self.ax.set_title("プロット可能な値列がありません")
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel("値")
            self._line_by_column = {}
            self._legend_artist_map = {}
            self._legend_entry_by_column = {}
            self._selected_pos_range = None
            self.canvas.draw_idle()
            self._create_span_selector()
            self.sel_xmin = self.sel_xmax = None
            self._update_sel_label()
            return

        # Determine x-axis values once
        if self.index_is_datetime:
            x_values = self._working_df.index
            xlabel = idx_name + " (datetime)"
            self._x_axis_mode = "datetime"
            self._x_axis_source_index = self._working_df.index
        else:
            try:
                numeric_index = pd.to_numeric(self._working_df.index, errors="coerce")
                if numeric_index.notna().mean() > 0.8:
                    x_values = numeric_index
                    xlabel = idx_name
                    self._x_axis_mode = "numeric"
                    self._x_axis_source_index = None
                else:
                    x_values = np.arange(len(self._working_df.index))
                    xlabel = f"{idx_name} (非数値: 順序インデックス表示)"
                    self._x_axis_mode = "positional"
                    self._x_axis_source_index = self._working_df.index
            except Exception:
                x_values = np.arange(len(self._working_df.index))
                xlabel = f"{idx_name} (順序インデックス表示)"
                self._x_axis_mode = "positional"
                self._x_axis_source_index = self._working_df.index

        # Clear plot and draw each value column
        self.ax.clear()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("値")

        title_base = f"{idx_name} に対する各列の推移"
        self._plot_title_base = title_base
        self._line_by_column = {}

        plotted_any = False
        for col in self.value_columns:
            if col not in self._working_df.columns:
                continue
            series = self._working_df[col]
            if not pd.api.types.is_numeric_dtype(series):
                series = pd.to_numeric(series, errors="coerce")
            if hasattr(series, "notna") and series.notna().sum() == 0:
                continue
            line, = self.ax.plot(x_values, series, label=str(col))
            visible = self.value_visibility.get(col, True)
            line.set_visible(visible)
            self._line_by_column[col] = line
            plotted_any = True

        if not plotted_any:
            self._line_by_column = {}
            self._legend_artist_map = {}
            self._legend_entry_by_column = {}
            self._plot_title_base = "プロット可能な値列がありません"
            self.ax.clear()
            self.ax.set_title("プロット可能な値列がありません")
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel("値")
            self.canvas.draw_idle()
            self._create_span_selector()
            self.sel_xmin = self.sel_xmax = None
            self._selected_pos_range = None
            self._update_sel_label()
            return

        if self._line_by_column:
            self.ax.legend(loc="upper right")
        self._setup_legend_interactivity()

        has_visible = any(line.get_visible() for line in self._line_by_column.values())
        self._refresh_plot_title(has_visible)

        if has_visible:
            self._autoscale_to_visible()

        self.canvas.draw_idle()

        # Re-create span selector for fresh axes
        self._create_span_selector()
        # Reset selection
        self.sel_xmin = self.sel_xmax = None
        self._selected_pos_range = None
        self._update_sel_label()
    
    def export_csv(self):
        if self.df is None:
            messagebox.showinfo("CSV未読込", "先にCSVを読み込んでください。")
            return
        if self.sel_xmin is None or self.sel_xmax is None:
            messagebox.showinfo("区間未選択", "エクスポートする前に、グラフ上で範囲をドラッグして選択してください。")
            return

        if self._working_df is None:
            messagebox.showerror("エクスポート失敗", "横軸が設定されていないため、データを切り出せません。CSVを読み込み直してください。")
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
            if self._x_axis_mode == "positional" and self._selected_pos_range is not None:
                start_pos, end_pos = self._selected_pos_range
                if end_pos < start_pos:
                    start_pos, end_pos = end_pos, start_pos
                end_pos = min(end_pos, len(dfw) - 1)
                sliced = dfw.iloc[start_pos:end_pos + 1]
            else:
                try:
                    idx_num = pd.to_numeric(dfw.index, errors="coerce")
                except Exception:
                    idx_num = None
                if idx_num is None:
                    messagebox.showerror("エクスポート失敗", "indexを数値として扱えないため、切り出しに失敗しました。数値または日時のindexを選んでください。")
                    return
                xmin, xmax = self.sel_xmin, self.sel_xmax
                if xmax < xmin:
                    xmin, xmax = xmax, xmin
                mask = (idx_num >= xmin) & (idx_num <= xmax)
                sliced = dfw.loc[mask]
        
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
