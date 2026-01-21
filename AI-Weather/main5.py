#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Control System — Tkinter GUI (Touch-Optimized Fullscreen + Scrollable Tabs)
-------------------------------------------------------------------------------
- เต็มหน้าจอ (F11 toggle / Esc ออกจาก fullscreen)
- สเกลอัตโนมัติตามความละเอียดจอ (base = 800x480)
- ปุ่ม/ช่องกรอก/แท็บ/ตาราง ขนาดใหญ่ แตะง่าย
- แท็บยาว (AI Mode, Predict Only) เลื่อนในแท็บเอง (แนวตั้ง+แนวนอน) ไม่ทับ Log
"""
from __future__ import annotations
import sys, time, json, random, calendar, threading, queue
from datetime import datetime, date, timedelta
from pathlib import Path
from dataclasses import dataclass

# ================== GPIO / Mock ==================
try:
    import Jetson.GPIO as REAL_GPIO  # type: ignore
    ON_JETSON = True
    GPIO = REAL_GPIO
except Exception:
    ON_JETSON = False
    class MockGPIO:
        BCM = "BCM"; OUT = "OUT"; HIGH = 1; LOW = 0
        def setmode(self, mode): print(f"[MockGPIO] setmode({mode})")
        def setwarnings(self, flag): pass
        def setup(self, pin, mode, initial=None):
            print(f"[MockGPIO] setup pin {pin} as {mode} initial={initial}")
        def output(self, pin, state):
            print(f"[MockGPIO] pin {pin} -> {'ON' if state==0 else 'OFF'}")
        def cleanup(self): print("[MockGPIO] cleanup()")
    GPIO = MockGPIO()

# กำหนดขา (ปรับตามของจริง)
UVA_PIN       = 17   # หลอด UVA-340
FAN_PIN       = 27   # พัดลม
SPRINKLER_PIN = 22   # สปริงเกอร์

ACTIVE_LOW = True  # ถ้ารีเลย์ Active-HIGH ให้เปลี่ยนเป็น False

def gpio_on(pin):  GPIO.output(pin, GPIO.LOW  if ACTIVE_LOW else GPIO.HIGH)
def gpio_off(pin): GPIO.output(pin, GPIO.HIGH if ACTIVE_LOW else GPIO.LOW)

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(UVA_PIN,       GPIO.OUT, initial=GPIO.HIGH if ACTIVE_LOW else GPIO.LOW)
    GPIO.setup(FAN_PIN,       GPIO.OUT, initial=GPIO.HIGH if ACTIVE_LOW else GPIO.LOW)
    GPIO.setup(SPRINKLER_PIN, GPIO.OUT, initial=GPIO.HIGH if ACTIVE_LOW else GPIO.LOW)

def all_off():
    try:
        gpio_off(UVA_PIN); gpio_off(FAN_PIN); gpio_off(SPRINKLER_PIN)
    except Exception:
        pass

# ================== เชื่อมกับ Ai.py ==================
import importlib.util, subprocess

def _resolve_abs(p: str) -> str:
    return str(Path(p).expanduser().resolve())

def _load_ai_from_path(ai_py_path: str):
    p = Path(ai_py_path)
    if not p.exists():
        raise FileNotFoundError(f"Ai.py not found at: {p}")
    spec = importlib.util.spec_from_file_location("Ai", str(p))
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def predict_weather_with_ai(models_dir: str, date_str: str, thr: float|None,
                            ai_py_path: str|None) -> dict:
    """เรียก AI แบบเดียวกับ CLI: import จากไฟล์/โมดูล -> fallback subprocess"""
    models_dir = _resolve_abs(models_dir)

    # 1) import จาก path ที่ระบุ
    if ai_py_path:
        try:
            mod = _load_ai_from_path(ai_py_path)
            return mod.predict_one(Path(models_dir), date_str, thr=thr)  # type: ignore
        except Exception as e:
            print(f"[AI] import by path failed, fallback to subprocess: {e}")

    # 2) import โมดูลชื่อ Ai ใน sys.path
    try:
        import Ai as AiMod  # type: ignore
        return AiMod.predict_one(Path(models_dir), date_str, thr=thr)  # type: ignore
    except Exception as e:
        print(f"[AI] normal import failed: {e}")

    # 3) subprocess
    cwd = None
    ai_file = "Ai.py"
    if ai_py_path:
        ap = Path(ai_py_path); cwd = str(ap.parent); ai_file = ap.name
    cmd = [sys.executable, ai_file, "predict", "--models_dir", models_dir, "--date", date_str]
    if thr is not None:
        cmd += ["--thr", str(thr)]
    cp = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if cp.returncode != 0:
        raise RuntimeError(f"AI subprocess error (code {cp.returncode}): {cp.stderr.strip()}")
    return json.loads(cp.stdout.strip())

# ================== Utility: เวลา / สุ่มวัน ==================
def _pick_unique_days_in_month(year:int, month:int, k:int) -> list[date]:
    _, ndays = calendar.monthrange(year, month)
    if k > ndays: raise ValueError(f"ขอ {k} วัน แต่เดือนนี้มี {ndays} วัน")
    days = random.sample(range(1, ndays+1), k)
    return sorted(date(year, month, d) for d in days)

def _dates_in_range(start:date, end:date) -> list[date]:
    n = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n)]

def _pick_unique_days_in_season(base_year:int, season_code:str, k:int) -> list[date]:
    if season_code == "1":
        start, end = date(base_year,2,15),  date(base_year,5,14)   # ร้อน
    elif season_code == "2":
        start, end = date(base_year,5,15),  date(base_year,10,14)  # ฝน
    elif season_code == "3":
        start, end = date(base_year,10,15), date(base_year+1,2,14) # หนาว
    else:
        raise ValueError("season_code ต้องเป็น 1/2/3")
    pool = _dates_in_range(start, end)
    if k > len(pool): raise ValueError(f"ขอ {k} วัน แต่ฤดูนี้มี {len(pool)} วัน")
    return sorted(random.sample(pool, k))

# ================== GUI ==================
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# --------- Scrollable Frame (แนวตั้ง + แนวนอน) ----------
class ScrollFrame(ttk.Frame):
    """เฟรมที่เลื่อนได้สองแกน (Canvas + inner Frame) ใช้กับแท็บยาว"""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical",   command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)

        # layout: canvas กินพื้นที่, vbar ขวา, hbar ล่าง
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # เนื้อหาภายใน
        self.body = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        # อัปเดต scrollregion ทุกครั้งที่ข้างในเปลี่ยนขนาด
        self.body.bind("<Configure>", self._on_body_config)
        # ปรับความกว้างของ inner อย่างฉลาด (อย่างน้อยเท่ากับ canvas)
        self.canvas.bind("<Configure>", self._on_canvas_config)

        # เมาส์/ทัชแพด: เลื่อนแนวตั้ง, กด Shift เพื่อเลื่อนแนวนอน
        self.body.bind_all("<MouseWheel>", self._on_mousewheel)           # Windows/macOS
        self.body.bind_all("<Shift-MouseWheel>", self._on_shiftwheel)
        # Linux (X11) ล้อเมาส์เป็น Button-4/5
        self.body.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.body.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll( 1, "units"))
        self.body.bind_all("<Shift-Button-4>", lambda e: self.canvas.xview_scroll(-1, "units"))
        self.body.bind_all("<Shift-Button-5>", lambda e: self.canvas.xview_scroll( 1, "units"))

    def _on_body_config(self, _):
        # กำหนดขอบเขตเลื่อนตามขนาดจริงของเนื้อหา
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_config(self, event):
        # ถ้าเนื้อหากว้างน้อยกว่าพื้นที่ ให้ขยาย inner ให้เท่ากับ canvas
        # แต่ถ้าเนื้อหากว้างกว่า ให้คงความกว้างจริงไว้เพื่อเลื่อนแนวนอนได้
        req_w = self.body.winfo_reqwidth()
        self.canvas.itemconfig(self._win, width=max(req_w, event.width))

    def _on_mousewheel(self, event):
        try:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        except Exception:
            pass

    def _on_shiftwheel(self, event):
        try:
            self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        except Exception:
            pass

# --------- Touch scaling helpers ----------
def _apply_touch_scaling(root: tk.Tk):
    """ตั้ง fullscreen + คำนวณสเกลตามจอ (base 800x480) + ปรับฟอนต์/สไตล์"""
    try:
        root.attributes("-fullscreen", True)
    except Exception:
        pass
    root.bind("<F11>", lambda e: root.attributes("-fullscreen", not bool(root.attributes("-fullscreen"))))
    root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

    root.update_idletasks()
    sw = root.winfo_screenwidth() or 800
    sh = root.winfo_screenheight() or 480
    scale = max(1.0, min(sw/800, sh/480))  # ไม่ลดต่ำกว่า 1

    try:
        root.tk.call('tk', 'scaling', 1.0 * scale)
    except Exception:
        pass

    base    = int(14 * scale)
    base_sm = max(12, int(13 * scale))
    base_lg = int(16 * scale)

    root.option_add("*Font", ("Segoe UI", base))
    root.option_add("*TButton.Font", ("Segoe UI", base))
    root.option_add("*TLabel.Font", ("Segoe UI", base))
    root.option_add("*TEntry.Font", ("Segoe UI", base))
    root.option_add("*Treeview.Font", ("Segoe UI", base_sm))
    root.option_add("*Treeview.Heading.Font", ("Segoe UI Semibold", base))
    root.option_add("*TNotebook.Tab.Font", ("Segoe UI", base_sm))

    style = ttk.Style()
    for th in ("vista", "xpnative", "clam", "alt", "default"):
        try:
            style.theme_use(th); break
        except Exception:
            continue

    pad_x = int(16 * scale)
    pad_y = int(10 * scale)
    style.configure("TButton", padding=(pad_x, pad_y))
    style.configure("TEntry", padding=(int(8*scale), int(8*scale)))
    style.configure("TNotebook.Tab", padding=(int(20*scale), int(10*scale)))
    style.configure("Treeview", rowheight=int(38 * scale))
    style.configure("Danger.TButton", padding=(int(20*scale), int(12*scale)),
                    font=("Segoe UI Semibold", base_lg))

@dataclass
class Settings:
    models_dir: str = "./models"
    ai_py_path: str|None = None
    thr: float|None = None
    speed: float = 1.0  # ตัวคูณเวลา (1.0 = ปกติ)

class ControlGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Jetson Control System — GUI (Touch Fullscreen)")
        _apply_touch_scaling(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        setup_gpio()

        # Threads & comms
        self.worker: threading.Thread|None = None
        self.stop_event = threading.Event()
        self.log_q: queue.Queue[str] = queue.Queue()

        # Settings
        self.settings = Settings()

        # ====== ROOT GRID: ให้ทุกอย่างยืดเต็ม ======
        self.root.rowconfigure(0, weight=1)  # Notebook
        self.root.rowconfigure(1, weight=0)  # แถบ speed/emergency
        self.root.rowconfigure(2, weight=1)  # Log
        self.root.columnconfigure(0, weight=1)

        # Main layout
        self._build_menu()
        self._build_tabs()       # row 0
        self._build_log_panel()  # row 1 + 2
        self._poll_log_queue()

    # ----- UI builders -----
    def _build_menu(self):
        menubar = tk.Menu(self.root, tearoff=0)
        self.root.config(menu=menubar)

        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="ตั้งค่า Models Dir...", command=self.pick_models_dir)
        m_file.add_command(label="ตั้งค่า Ai.py...", command=self.pick_ai_py)
        m_file.add_separator()
        m_file.add_command(label="ออก", command=self.on_close)
        menubar.add_cascade(label="ไฟล์", menu=m_file)

        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="เกี่ยวกับ", command=lambda: messagebox.showinfo(
            "เกี่ยวกับ", "Jetson Control System GUI\nควบคุมอุปกรณ์ + AI Mode\n(Fullscreen Touch)"))
        menubar.add_cascade(label="ช่วยเหลือ", menu=m_help)

    def _build_tabs(self):
        nb = ttk.Notebook(self.root)
        nb.grid(row=0, column=0, sticky="nsew")
        self.nb = nb

        # แท็บสั้นใช้ Frame ปกติ
        self.tab_sunny = ttk.Frame(nb)
        self.tab_rain  = ttk.Frame(nb)
        self.tab_alt   = ttk.Frame(nb)

        # แท็บยาวใช้ ScrollFrame (สองแกน)
        self.tab_ai_sf      = ScrollFrame(nb)
        self.tab_predict_sf = ScrollFrame(nb)

        nb.add(self.tab_sunny, text="Sunny")
        nb.add(self.tab_rain,  text="Rain")
        nb.add(self.tab_alt,   text="Alternate")
        nb.add(self.tab_ai_sf, text="AI Mode")
        nb.add(self.tab_predict_sf, text="Predict Only")

        # ยืดคอลัมน์ในทุกแท็บ
        for tab in (self.tab_sunny, self.tab_rain, self.tab_alt,
                    self.tab_ai_sf.body, self.tab_predict_sf.body):
            tab.rowconfigure(999, weight=1)
            for c in range(8):
                tab.columnconfigure(c, weight=1)

        self._build_tab_sunny()
        self._build_tab_rain()
        self._build_tab_alt()
        self._build_tab_ai(parent=self.tab_ai_sf.body)            # ใช้ body ของ ScrollFrame
        self._build_tab_predict(parent=self.tab_predict_sf.body)  # ใช้ body ของ ScrollFrame

    def _build_log_panel(self):
        frm = ttk.Frame(self.root)
        frm.grid(row=1, column=0, sticky="ew", padx=8, pady=6)

        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, padx=10, pady=6)

        self.time_label = ttk.Label(left, text="พร้อมทำงาน")
        self.time_label.grid(row=0, column=0, sticky="w", padx=(0,12))

        ttk.Label(left, text="Speed x").grid(row=0, column=1, sticky="e")
        self.sv_speed = tk.StringVar(value=f"{self.settings.speed:.2f}")
        e_speed = ttk.Entry(left, textvariable=self.sv_speed, width=8)
        e_speed.grid(row=0, column=2, sticky="w", padx=(6,8))
        ttk.Button(left, text="ปรับ", command=self._apply_speed_from_ui).grid(row=0, column=3, sticky="w")

        self.btn_stop = ttk.Button(frm, text="Emergency Stop", style="Danger.TButton",
                                   command=self.emergency_stop)
        self.btn_stop.pack(side=tk.RIGHT, padx=10, pady=6)

        self.log = tk.Text(self.root, height=8, wrap=tk.WORD, font=("Consolas", 13))
        self.log.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0,10))
        self.log.config(state=tk.DISABLED)

    # ----- Sunny tab -----
    def _build_tab_sunny(self):
        f = self.tab_sunny
        pad = dict(padx=14, pady=10)

        ttk.Label(f, text="ระยะเวลาเปิด (นาที)").grid(row=0, column=0, sticky="w", **pad)
        self.sv_sunny_minutes = tk.StringVar(value="5")
        ttk.Entry(f, textvariable=self.sv_sunny_minutes, width=10).grid(row=0, column=1, sticky="w", **pad)
        ttk.Button(f, text="เริ่ม Sunny", command=self.start_sunny).grid(row=0, column=2, **pad)

    # ----- Rain tab -----
    def _build_tab_rain(self):
        f = self.tab_rain
        pad = dict(padx=14, pady=10)

        ttk.Label(f, text="จำนวนรอบ").grid(row=0, column=0, sticky="w", **pad)
        self.sv_rain_rounds = tk.StringVar(value="1")
        ttk.Entry(f, textvariable=self.sv_rain_rounds, width=10).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(f, text="เวลาเปิดต่อรอบ (นาที)").grid(row=1, column=0, sticky="w", **pad)
        self.sv_rain_on = tk.StringVar(value="5")
        ttk.Entry(f, textvariable=self.sv_rain_on, width=10).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(f, text="เวลาปิดต่อรอบ (นาที)").grid(row=2, column=0, sticky="w", **pad)
        self.sv_rain_off = tk.StringVar(value="10")
        ttk.Entry(f, textvariable=self.sv_rain_off, width=10).grid(row=2, column=1, sticky="w", **pad)

        ttk.Button(f, text="เริ่ม Rain", command=self.start_rain).grid(row=3, column=0, columnspan=2, **pad)

    # ----- Alternate tab -----
    def _build_tab_alt(self):
        f = self.tab_alt
        pad = dict(padx=14, pady=10)

        ttk.Label(f, text="จำนวนรอบ").grid(row=0, column=0, sticky="w", **pad)
        self.sv_alt_rounds = tk.StringVar(value="1")
        ttk.Entry(f, textvariable=self.sv_alt_rounds, width=10).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(f, text="เวลาเปิด (นาที)").grid(row=1, column=0, sticky="w", **pad)
        self.sv_alt_on = tk.StringVar(value="8")
        ttk.Entry(f, textvariable=self.sv_alt_on, width=10).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(f, text="เวลาปิด (นาที)").grid(row=2, column=0, sticky="w", **pad)
        self.sv_alt_off = tk.StringVar(value="4")
        ttk.Entry(f, textvariable=self.sv_alt_off, width=10).grid(row=2, column=1, sticky="w", **pad)

        ttk.Button(f, text="เริ่ม Alternate", command=self.start_alt).grid(row=3, column=0, columnspan=2, **pad)

    # ----- AI tab (Scrollable) -----
    def _build_tab_ai(self, parent):
        f = parent
        pad = dict(padx=14, pady=10)

        frm_set = ttk.LabelFrame(f, text="ตั้งค่า AI")
        frm_set.grid(row=0, column=0, columnspan=6, sticky="ew", **pad)
        frm_set.columnconfigure(1, weight=1)

        ttk.Label(frm_set, text="Models Dir:").grid(row=0, column=0, sticky="w", **pad)
        self.sv_models = tk.StringVar(value="./models")
        ttk.Entry(frm_set, textvariable=self.sv_models).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(frm_set, text="เลือก...", command=self.pick_models_dir).grid(row=0, column=2, **pad)

        ttk.Label(frm_set, text="Ai.py:").grid(row=1, column=0, sticky="w", **pad)
        self.sv_ai_py = tk.StringVar(value="")
        ttk.Entry(frm_set, textvariable=self.sv_ai_py).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frm_set, text="เลือก...", command=self.pick_ai_py).grid(row=1, column=2, **pad)

        ttk.Label(frm_set, text="Threshold (เว้นว่าง=ใช้ของโมเดล)").grid(row=2, column=0, sticky="w", **pad)
        self.sv_thr = tk.StringVar(value="")
        ttk.Entry(frm_set, textvariable=self.sv_thr, width=12).grid(row=2, column=1, sticky="w", **pad)

        frm_rainparam = ttk.LabelFrame(f, text="พารามิเตอร์ Rain (AI ใช้)")
        frm_rainparam.grid(row=1, column=0, columnspan=6, sticky="ew", **pad)

        ttk.Label(frm_rainparam, text="เวลาเปิดต่อรอบ (นาที)").grid(row=0, column=0, sticky="w", **pad)
        self.sv_ai_rain_on_min = tk.StringVar(value="5")
        ttk.Entry(frm_rainparam, textvariable=self.sv_ai_rain_on_min, width=10).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(frm_rainparam, text="เวลาปิดต่อรอบ (นาที)").grid(row=0, column=2, sticky="w", **pad)
        self.sv_ai_rain_off_min = tk.StringVar(value="10")
        ttk.Entry(frm_rainparam, textvariable=self.sv_ai_rain_off_min, width=10).grid(row=0, column=3, sticky="w", **pad)

        frm_single = ttk.LabelFrame(f, text="Single mode — ทำนาย 1 วันแล้วรันทันที")
        frm_single.grid(row=2, column=0, columnspan=6, sticky="ew", **pad)

        ttk.Label(frm_single, text="วันที่ (YYYY-MM-DD)").grid(row=0, column=0, sticky="w", **pad)
        self.sv_single_date = tk.StringVar(value=date.today().isoformat())
        ttk.Entry(frm_single, textvariable=self.sv_single_date, width=18).grid(row=0, column=1, sticky="w", **pad)
        ttk.Button(frm_single, text="Run Single", command=self.run_ai_single).grid(row=0, column=2, **pad)

        frm_multi = ttk.LabelFrame(f, text="Multi mode — สุ่มหลายวันแล้วรันอัตโนมัติ (วันละ 1 รอบ)")
        frm_multi.grid(row=3, column=0, columnspan=6, sticky="ew", **pad)

        self.multi_mode = tk.StringVar(value="M")  # M or S
        ttk.Radiobutton(frm_multi, text="Month (M)", value="M", variable=self.multi_mode).grid(row=0, column=0, **pad)
        ttk.Radiobutton(frm_multi, text="Season (S)", value="S", variable=self.multi_mode).grid(row=0, column=1, **pad)

        ttk.Label(frm_multi, text="ปี (YYYY)").grid(row=1, column=0, sticky="w", **pad)
        self.sv_m_year = tk.StringVar(value=str(date.today().year))
        ttk.Entry(frm_multi, textvariable=self.sv_m_year, width=10).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(frm_multi, text="เดือน (1-12)").grid(row=1, column=2, sticky="w", **pad)
        self.sv_m_month = tk.StringVar(value=str(date.today().month))
        ttk.Entry(frm_multi, textvariable=self.sv_m_month, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(frm_multi, text="จำนวนวันที่จะสุ่ม").grid(row=1, column=4, sticky="w", **pad)
        self.sv_m_k = tk.StringVar(value="3")
        ttk.Entry(frm_multi, textvariable=self.sv_m_k, width=8).grid(row=1, column=5, sticky="w", **pad)

        ttk.Label(frm_multi, text="ปีอ้างอิงฤดู (YYYY)").grid(row=2, column=0, sticky="w", **pad)
        self.sv_s_year = tk.StringVar(value=str(date.today().year))
        ttk.Entry(frm_multi, textvariable=self.sv_s_year, width=10).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(frm_multi, text="ฤดู 1=ร้อน 2=ฝน 3=หนาว").grid(row=2, column=2, sticky="w", **pad)
        self.sv_s_code = tk.StringVar(value="2")
        ttk.Entry(frm_multi, textvariable=self.sv_s_code, width=8).grid(row=2, column=3, sticky="w", **pad)

        ttk.Label(frm_multi, text="จำนวนวันที่จะสุ่ม").grid(row=2, column=4, sticky="w", **pad)
        self.sv_s_k = tk.StringVar(value="5")
        ttk.Entry(frm_multi, textvariable=self.sv_s_k, width=8).grid(row=2, column=5, sticky="w", **pad)

        ttk.Label(frm_multi, text="นาที Sunny ต่อวัน").grid(row=3, column=0, sticky="w", **pad)
        self.sv_multi_sunny_min = tk.StringVar(value="5")
        ttk.Entry(frm_multi, textvariable=self.sv_multi_sunny_min, width=10).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(frm_multi, text="จำนวนรอบ Rain ต่อวัน").grid(row=3, column=2, sticky="w", **pad)
        self.sv_multi_rain_rounds = tk.StringVar(value="1")
        ttk.Entry(frm_multi, textvariable=self.sv_multi_rain_rounds, width=10).grid(row=3, column=3, sticky="w", **pad)

        ttk.Button(frm_multi, text="Run Multi", command=self.run_ai_multi).grid(row=4, column=0, columnspan=2, **pad)

    # ----- Predict Only tab (Scrollable) -----
    def _build_tab_predict(self, parent):
        f = parent
        pad = dict(padx=14, pady=10)

        msg = ttk.Label(f, text="หมายเหตุ: หน้านี้ใช้ Models Dir / Ai.py / Threshold เดียวกับหน้า AI Mode")
        msg.grid(row=0, column=0, columnspan=6, sticky="w", **pad)

        frm_daily = ttk.LabelFrame(f, text="รายวัน — แสดงผลทำนายอย่างเดียว")
        frm_daily.grid(row=1, column=0, columnspan=6, sticky="ew", **pad)
        ttk.Label(frm_daily, text="วันที่ (YYYY-MM-DD)").grid(row=0, column=0, sticky="w", **pad)
        self.sv_p_daily = tk.StringVar(value=date.today().isoformat())
        ttk.Entry(frm_daily, textvariable=self.sv_p_daily, width=18).grid(row=0, column=1, sticky="w", **pad)
        ttk.Button(frm_daily, text="Predict (Daily)", command=self.predict_only_daily).grid(row=0, column=2, **pad)

        frm_range = ttk.LabelFrame(f, text="ช่วงวัน — แสดงผลทำนายอย่างเดียว")
        frm_range.grid(row=2, column=0, columnspan=6, sticky="ew", **pad)
        ttk.Label(frm_range, text="เริ่ม (YYYY-MM-DD)").grid(row=0, column=0, sticky="w", **pad)
        self.sv_p_start = tk.StringVar(value=date.today().replace(day=1).isoformat())
        ttk.Entry(frm_range, textvariable=self.sv_p_start, width=18).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(frm_range, text="สิ้นสุด (YYYY-MM-DD)").grid(row=0, column=2, sticky="w", **pad)
        self.sv_p_end = tk.StringVar(value=date.today().isoformat())
        ttk.Entry(frm_range, textvariable=self.sv_p_end, width=18).grid(row=0, column=3, sticky="w", **pad)

        ttk.Button(frm_range, text="Predict (Range)", command=self.predict_only_range).grid(row=0, column=4, **pad)
        ttk.Button(frm_range, text="ล้างผล", command=self._predict_clear).grid(row=0, column=5, **pad)

        cols = ("date", "label", "rain_probability", "temp_max", "temp_min", "humidity")
        self.pred_tree = ttk.Treeview(f, columns=cols, show="headings", height=8)
        self.pred_tree.grid(row=3, column=0, columnspan=6, sticky="nsew", padx=10, pady=(0,10))
        self.pred_tree.heading("date", text="วันที่")
        self.pred_tree.heading("label", text="ผล (SUNNY/RAIN)")
        self.pred_tree.heading("rain_probability", text="rain_probability")
        self.pred_tree.heading("temp_max", text="Temp Max (°C)")
        self.pred_tree.heading("temp_min", text="Temp Min (°C)")
        self.pred_tree.heading("humidity", text="Humidity (%)")
        self.pred_tree.column("date", width=140, anchor="center")
        self.pred_tree.column("label", width=170, anchor="center")
        self.pred_tree.column("rain_probability", width=180, anchor="center")
        self.pred_tree.column("temp_max", width=150, anchor="center")
        self.pred_tree.column("temp_min", width=150, anchor="center")
        self.pred_tree.column("humidity", width=140, anchor="center")

        self._pred_rows_raw = {}  # iid -> raw_json_str

        ttk.Button(f, text="ดูรายละเอียด (JSON)", command=self._predict_show_json_selected)\
            .grid(row=4, column=0, sticky="w", **pad)

        f.rowconfigure(3, weight=1)
        for c in range(6):
            f.columnconfigure(c, weight=1)

    # ----- Settings pickers -----
    def pick_models_dir(self):
        d = filedialog.askdirectory(title="เลือกโฟลเดอร์ models_dir")
        if d:
            self.sv_models.set(d)

    def pick_ai_py(self):
        p = filedialog.askopenfilename(title="เลือกไฟล์ Ai.py", filetypes=[("Python","*.py"), ("All","*.*")])
        if p:
            self.sv_ai_py.set(p)

    # ----- Logging helpers -----
    def log_print(self, *a):
        s = " ".join(str(x) for x in a)
        self.log_q.put(s)

    def _poll_log_queue(self):
        try:
            while True:
                s = self.log_q.get_nowait()
                self.log.config(state=tk.NORMAL)
                self.log.insert(tk.END, s + "\n")
                self.log.see(tk.END)
                self.log.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log_queue)

    def set_timer_label(self, text: str):
        self.time_label.config(text=text)

    # ----- Speed control -----
    def _apply_speed_from_ui(self):
        s = (self.sv_speed.get() or "").strip()
        try:
            spd = float(s)
            if spd <= 0:
                raise ValueError("speed ต้อง > 0")
            if spd > 1000:
                raise ValueError("speed สูงเกินไป (แนะนำ <= 1000x)")
            self.settings.speed = spd
            self.log_print(f"[Speed] ตั้งค่าเร่งเวลาเป็น {spd:.2f}x แล้ว")
        except Exception as e:
            messagebox.showerror("Speed ไม่ถูกต้อง", f"ใส่เลขทศนิยม > 0 เช่น 1, 2.5, 10\nรายละเอียด: {e}")
            self.sv_speed.set(f"{self.settings.speed:.2f}")

    # ----- Emergency stop -----
    def emergency_stop(self):
        self.stop_event.set()
        all_off()
        self.set_timer_label("หยุดฉุกเฉินแล้ว")
        self.log_print("[EMERGENCY] หยุดทุกอุปกรณ์แล้ว")

    # ----- Thread run helper -----
    def _start_worker(self, target, *args, **kwargs):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("กำลังทำงาน", "โปรดหยุดงานเดิมก่อน หรือรอให้เสร็จ")
            return False
        self.stop_event.clear()
        self.worker = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
        self.worker.start()
        return True

    # ----- Countdown (thread) -----
    def _countdown(self, total_seconds: int) -> bool:
        """นับเวลาถอยหลังโดยใช้ speed เร่ง/ผ่อนเวลา"""
        try:
            spd = float(self.settings.speed) if self.settings.speed else 1.0
            if spd <= 0:
                spd = 1.0
            sleep_tick = 0.1 / spd  # 0.1 วินาทีตรรกะต่อช่วงย่อย

            for remaining in range(total_seconds, -1, -1):
                if self.stop_event.is_set():
                    return False
                mm, ss = divmod(remaining, 60)
                self.root.after(0, self.set_timer_label, f"ทำงาน... เหลือ {mm:02d}:{ss:02d}")
                for _ in range(10):
                    if self.stop_event.is_set():
                        return False
                    if sleep_tick > 0:
                        time.sleep(sleep_tick)
            return True
        except Exception as e:
            self.log_print(f"[Countdown Error] {e}")
            return False
        finally:
            self.root.after(0, self.set_timer_label, "พร้อมทำงาน")

    # ----- Mode implementations (in threads) -----
    def _run_sunny_worker(self, minutes: int):
        sec = int(minutes) * 60
        self.log_print(f"[Sunny] เปิด UVA-340 + พัดลม {minutes} นาที (Speed {self.settings.speed:.2f}x)")
        gpio_on(UVA_PIN); gpio_on(FAN_PIN)
        try:
            finished = self._countdown(sec)
        finally:
            gpio_off(UVA_PIN); gpio_off(FAN_PIN)
        self.log_print("[Sunny] ครบเวลา" if finished else "[Sunny] ถูกหยุด")

    def _run_rain_worker(self, rounds: int, on_min: int, off_min: int):
        self.log_print(f"[Rain] {rounds} รอบ (เปิด {on_min} นาที / ปิด {off_min} นาที) (Speed {self.settings.speed:.2f}x)")
        try:
            for r in range(1, rounds + 1):
                if self.stop_event.is_set():  break
                self.log_print(f"[Rain] รอบ {r}: เปิดพัดลม+สปริงเกอร์ {on_min} นาที")
                gpio_on(FAN_PIN); gpio_on(SPRINKLER_PIN)
                finished_on = self._countdown(on_min * 60)
                gpio_off(SPRINKLER_PIN); gpio_off(FAN_PIN)
                if not finished_on:
                    self.log_print("[Rain] หยุดช่วงเปิด"); return
                if self.stop_event.is_set(): break
                self.log_print(f"[Rain] รอบ {r}: ปิดทั้งหมด {off_min} นาที")
                finished_off = self._countdown(off_min * 60)
                if not finished_off:
                    self.log_print("[Rain] หยุดช่วงปิด"); return
            self.log_print("[Rain] ครบทุกช่วง")
        finally:
            gpio_off(SPRINKLER_PIN); gpio_off(FAN_PIN)

    def _run_alt_worker(self, rounds: int, on_min: int, off_min: int):
        self.log_print(f"[Alternate] {rounds} รอบ (เปิด {on_min} นาที / ปิด {off_min} นาที) (Speed {self.settings.speed:.2f}x)")
        try:
            for r in range(1, rounds + 1):
                if self.stop_event.is_set(): break
                self.log_print(f"[Alternate] รอบ {r}: เปิด UVA-340 + พัดลม {on_min} นาที")
                gpio_on(UVA_PIN); gpio_on(FAN_PIN)
                finished_on = self._countdown(on_min * 60)
                gpio_off(UVA_PIN); gpio_off(FAN_PIN)
                if not finished_on:
                    self.log_print("[Alternate] หยุดช่วงเปิด"); return
                if self.stop_event.is_set(): break
                self.log_print(f"[Alternate] รอบ {r}: ปิด {off_min} นาที")
                finished_off = self._countdown(off_min * 60)
                if not finished_off:
                    self.log_print("[Alternate] หยุดช่วงปิด"); return
            self.log_print("[Alternate] ครบทุกช่วง")
        finally:
            gpio_off(UVA_PIN); gpio_off(FAN_PIN)

    # ----- Start buttons -----
    def start_sunny(self):
        try:
            minutes = int(self.sv_sunny_minutes.get())
            if minutes <= 0: raise ValueError
        except Exception:
            messagebox.showerror("ค่าว่างหรือไม่ถูกต้อง", "กรุณาใส่นาทีเป็นตัวเลขมากกว่า 0")
            return
        self._start_worker(self._run_sunny_worker, minutes)

    def start_rain(self):
        try:
            rounds = int(self.sv_rain_rounds.get()); onm = int(self.sv_rain_on.get()); offm = int(self.sv_rain_off.get())
            if rounds<=0 or onm<=0 or offm<=0: raise ValueError
        except Exception:
            messagebox.showerror("ค่าว่างหรือไม่ถูกต้อง", "กรุณาใส่จำนวนรอบ/นาทีให้ถูกต้อง (มากกว่า 0)")
            return
        self._start_worker(self._run_rain_worker, rounds, onm, offm)

    def start_alt(self):
        try:
            rounds = int(self.sv_alt_rounds.get()); onm = int(self.sv_alt_on.get()); offm = int(self.sv_alt_off.get())
            if rounds<=0 or onm<=0 or offm<=0: raise ValueError
        except Exception:
            messagebox.showerror("ค่าว่างหรือไม่ถูกต้อง", "กรุณาใส่จำนวนรอบ/นาทีให้ถูกต้อง (มากกว่า 0)")
            return
        self._start_worker(self._run_alt_worker, rounds, onm, offm)

    # ----- AI helpers -----
    def _read_thr(self) -> float|None:
        s = self.sv_thr.get().strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            messagebox.showerror("Threshold ไม่ถูกต้อง", "กรุณาใส่ค่าทศนิยม หรือเว้นว่าง")
            return None

    def _predict(self, date_str: str) -> dict:
        models_dir = self.sv_models.get().strip() or "./models"
        ai_py = self.sv_ai_py.get().strip() or None
        thr = self._read_thr()
        return predict_weather_with_ai(models_dir, date_str, thr, ai_py)

    def _read_ai_rain_params(self) -> tuple[int,int]:
        try:
            onm = int((self.sv_ai_rain_on_min.get() or "0").strip())
            offm = int((self.sv_ai_rain_off_min.get() or "0").strip())
            if onm <= 0 or offm <= 0:
                raise ValueError
        except Exception:
            raise ValueError("พารามิเตอร์ Rain (เปิด/ปิด นาที) ไม่ถูกต้อง ต้องเป็นจำนวนเต็ม > 0")
        return onm, offm

    # ----- AI single -----
    def run_ai_single(self):
        date_str = self.sv_single_date.get().strip()
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("รูปแบบวันที่ไม่ถูกต้อง", "ตัวอย่าง 2025-09-16")
            return

        try:
            ai_onm, ai_offm = self._read_ai_rain_params()
        except Exception as e:
            messagebox.showerror("ค่าฝน (AI) ไม่ถูกต้อง", f"{e}")
            return

        def worker():
            try:
                res = self._predict(date_str)
            except Exception as e:
                self.log_print(f"[AI] Error: {e}")
                return
            self.log_print("[AI:Single] ผลจาก AI:\n" + json.dumps(res, ensure_ascii=False, indent=2))
            label = str(res.get("rain_predicted","Sunny")).strip().lower()
            prob  = float(res.get("rain_probability", 0.0))
            self.log_print(f"[AI:Single] สรุป: {label.upper()} (rain_probability={prob:.3f})")

            if label == "rain":
                self._run_rain_worker(rounds=1, on_min=ai_onm, off_min=ai_offm)
            else:
                self._run_sunny_worker(minutes=5)
        self._start_worker(worker)

    # ----- AI multi -----
    def run_ai_multi(self):
        mode = self.multi_mode.get()  # M or S

        try:
            ai_onm, ai_offm = self._read_ai_rain_params()
        except Exception as e:
            messagebox.showerror("ค่าฝน (AI) ไม่ถูกต้อง", f"{e}")
            return

        try:
            if mode == "M":
                year = int(self.sv_m_year.get()); month = int(self.sv_m_month.get()); k = int(self.sv_m_k.get())
                dates = _pick_unique_days_in_month(year, month, k)
            else:
                by = int(self.sv_s_year.get()); sc = self.sv_s_code.get().strip(); k = int(self.sv_s_k.get())
                dates = _pick_unique_days_in_season(by, sc, k)
            sunny_min = int(self.sv_multi_sunny_min.get())
            rain_rounds = int(self.sv_multi_rain_rounds.get())
            if sunny_min<=0 or rain_rounds<=0: raise ValueError
        except Exception as e:
            messagebox.showerror("ค่าหรือช่วงไม่ถูกต้อง", f"รายละเอียด: {e}")
            return

        def worker():
            self.log_print("[AI:Multi] กำหนดการ:")
            for d in dates: self.log_print(" -", d.isoformat())
            self.log_print(f"[AI:Multi] Param: Rain rounds={rain_rounds}, on={ai_onm} นาที, off={ai_offm} นาที")
            self.log_print("[AI:Multi] เริ่มรันทันทีตามลำดับวัน")

            for d in dates:
                if self.stop_event.is_set():
                    self.log_print("[AI:Multi] ถูกหยุดฉุกเฉิน — ยุติการทำงาน")
                    break
                self.log_print(f"\n=== วันที่ {d.isoformat()} ===")
                try:
                    res = self._predict(d.isoformat())
                except Exception as e:
                    self.log_print(f"[{d}] เรียก AI ล้มเหลว: {e}")
                    continue
                self.log_print(json.dumps(res, ensure_ascii=False, indent=2))
                label = str(res.get("rain_predicted","Sunny")).strip().lower()
                prob  = float(res.get("rain_probability", 0.0))
                self.log_print(f"[AI:Multi] สรุป: {label.upper()} (rain_probability={prob:.3f})")

                if label == "rain":
                    self._run_rain_worker(rain_rounds, on_min=ai_onm, off_min=ai_offm)
                else:
                    self._run_sunny_worker(sunny_min)
            self.log_print("\n[AI:Multi] จบการทำงาน")
        self._start_worker(worker)

    # ===== PREDICT ONLY =====
    def _predict_insert_row(self, dt_str: str, res: dict):
        label = str(res.get("rain_predicted", "Sunny")).strip().upper()
        prob  = float(res.get("rain_probability", 0.0))
        tmax = res.get("t_max_C_pred", res.get("temp_max", "-"))
        tmin = res.get("t_min_C_pred", res.get("temp_min", "-"))
        hum  = res.get("humid_avg_pct", res.get("humidity", "-"))

        def fmt(x, nd=1):
            try:
                return f"{float(x):.{nd}f}"
            except Exception:
                return str(x)

        iid = self.pred_tree.insert(
            "", "end",
            values=(dt_str, label, f"{prob:.3f}", fmt(tmax,1), fmt(tmin,1), fmt(hum,1))
        )
        self._pred_rows_raw[iid] = json.dumps(res, ensure_ascii=False, indent=2)

    def _predict_clear(self):
        for iid in self.pred_tree.get_children():
            self.pred_tree.delete(iid)
        self._pred_rows_raw.clear()

    def _predict_show_json_selected(self):
        sel = self.pred_tree.selection()
        if not sel:
            messagebox.showinfo("ไม่มีรายการ", "โปรดเลือกแถวในตารางก่อน")
            return
        iid = sel[0]
        raw = self._pred_rows_raw.get(iid, "{}")
        win = tk.Toplevel(self.root)
        win.title("รายละเอียดผลทำนาย (JSON)")
        txt = tk.Text(win, wrap="word", width=80, height=24, font=("Consolas", 13))
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", raw)
        txt.config(state="disabled")

    def predict_only_daily(self):
        date_str = self.sv_p_daily.get().strip()
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("รูปแบบวันที่ไม่ถูกต้อง", "ตัวอย่าง 2025-09-16")
            return

        def worker():
            self.set_timer_label("กำลังทำนาย (รายวัน)...")
            try:
                res = self._predict(date_str)
                self._predict_insert_row(date_str, res)
            except Exception as e:
                messagebox.showerror("Predict ล้มเหลว", f"{e}")
            finally:
                self.set_timer_label("พร้อมทำงาน")
        self._start_worker(worker)

    def predict_only_range(self):
        s = self.sv_p_start.get().strip()
        e = self.sv_p_end.get().strip()
        try:
            ds = datetime.strptime(s, "%Y-%m-%d").date()
            de = datetime.strptime(e, "%Y-%m-%d").date()
            if de < ds:
                raise ValueError("วันสิ้นสุดต้องไม่ก่อนวันเริ่ม")
        except Exception as err:
            messagebox.showerror("ช่วงวันไม่ถูกต้อง", f"{err}")
            return

        dates = _dates_in_range(ds, de)
        if len(dates) > 400:
            if not messagebox.askyesno("ยืนยัน", f"ช่วงวัน {len(dates)} วัน อาจใช้เวลานาน ต้องการต่อหรือไม่?"):
                return

        def worker():
            self._predict_clear()
            self.set_timer_label(f"กำลังทำนาย {len(dates)} วัน...")
            try:
                for i, d in enumerate(dates, 1):
                    if self.stop_event.is_set():
                        break
                    try:
                        res = self._predict(d.isoformat())
                        self._predict_insert_row(d.isoformat(), res)
                    except Exception as ee:
                        self.log_print(f"[PredictOnly:{d}] ล้มเหลว: {ee}")
                    self.root.after(0, self.set_timer_label, f"กำลังทำนาย {i}/{len(dates)} วัน")
                self.set_timer_label("พร้อมทำงาน")
            finally:
                pass
        self._start_worker(worker)

    # ----- Close -----
    def on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("ออกจากโปรแกรม", "มีงานกำลังทำอยู่ ต้องการหยุดและออกหรือไม่?"):
                return
        self.stop_event.set()
        all_off()
        try:
            GPIO.cleanup()
        except Exception:
            pass
        self.root.destroy()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ControlGUI(root)
        root.mainloop()
    finally:
        all_off()
        try:
            GPIO.cleanup()
        except Exception:
            pass
