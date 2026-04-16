import h5py
import numpy as np
import glob
import os
import datetime
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui
import sys

# --- Pomocné funkce na čtení dat ---
def ProcessData(hf, x, y, z):
    frames = int(np.array(hf.get('FRAMES')))
    data_list = []
    for n in range(1, frames + 1):
        dset = hf.get(str(n))
        if dset is not None:
            data_list.append(np.array(dset[:1400]))
    if not data_list:
        return 0
    data_array = np.vstack(data_list)
    max_vals = np.max(data_array, axis=1)
    min_vals = np.min(data_array, axis=1)
    sum_vals = np.sum(data_array, axis=1)
    mask = max_vals < 256
    x.extend(max_vals[mask])
    y.extend(sum_vals[mask])
    z.extend(min_vals[mask])
    return np.sum(mask)

def read_time(hf, name):
    t = None
    if name in hf:
        t = hf[name][()]
    elif name in hf.attrs:
        t = hf.attrs[name]
    if isinstance(t, bytes):
        t = t.decode('utf-8')
    if isinstance(t, str):
        try:
            return datetime.datetime.fromisoformat(t)
        except Exception:
            pass
    if isinstance(t, (int, float)):
        return datetime.datetime.fromtimestamp(t)
    return None

def load_data(prefix, pattern, x, y, z):
    n_particles = 0
    start_time = None
    end_time = None
    for f in glob.iglob(os.path.join(prefix, pattern)):
        print(f"Načítám soubor: {f}")
        try:
            with h5py.File(f, 'r') as hf:
                n = ProcessData(hf, x, y, z)
                n_particles += n
                st = read_time(hf, 'START_TIME')
                et = read_time(hf, 'END_TIME')
                if st is not None:
                    if start_time is None or st < start_time:
                        start_time = st
                if et is not None:
                    if end_time is None or et > end_time:
                        end_time = et
        except Exception as e:
            print(f"Chyba při načítání {f}: {e}")
    return n_particles, start_time, end_time

def format_timedelta(start, end):
    if start and end:
        td = end - start
        seconds = int(td.total_seconds())
        return str(datetime.timedelta(seconds=seconds))
    return "?"

# --- Hlavní program ---
PREFIX = "/storage/experiments/2025/05_CERF/NeutronExperiment/RUN_Test2/"
PREFIX = "/home/roman/mnt/kapybara/storage/experiments/2025/05_CERF/NeutronExperiment/CERF_2025_05_27_RUN1/"

PATTERNS = {
    "B10": "data_oscB10_*.h5",
    "Li6": "data_oscLi6_*.h5",
    "Si":  "data_oscSi_*.h5",
}
HIST_BINS = (200, 128)

# --- Načtení dat před spuštěním GUI ---
x0, y0, z0 = [], [], []
n0, st0, et0 = load_data(PREFIX, PATTERNS["B10"], x0, y0, z0)
x1, y1, z1 = [], [], []
n1, st1, et1 = load_data(PREFIX, PATTERNS["Li6"], x1, y1, z1)
x2, y2, z2 = [], [], []
n2, st2, et2 = load_data(PREFIX, PATTERNS["Si"], x2, y2, z2)

x_all = x0 + x1 + x2
y_all = y0 + y1 + y2
x_min, x_max = min(x_all), max(x_all)
y_min, y_max = min(y_all), max(y_all)

data_list = [
    (y0, x0, f'B10 (n={n0}, Δt={format_timedelta(st0, et0)})'),
    (y1, x1, f'Li6 (n={n1}, Δt={format_timedelta(st1, et1)})'),
    (y2, x2, f'Si  (n={n2}, Δt={format_timedelta(st2, et2)})'),
]

# --- PyQtGraph GUI s PyQt6 ---
class HistogramWindow(QtWidgets.QMainWindow):
    def __init__(self, data_list, y_range, x_range, parent=None):
        super().__init__(parent)
        self.data_list = data_list
        self.idx = 0
        self.y_range = y_range
        self.x_range = x_range
        self.setWindowTitle("Interaktivní histogramy s PyQtGraph")

        # Hlavní widgety
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Graf
        self.plot_widget = pg.PlotWidget()
        self.img = pg.ImageItem()
        self.plot_widget.addItem(self.img)
        self.layout.addWidget(self.plot_widget)

        # Label a tlačítka
        self.label = QtWidgets.QLabel()
        self.layout.addWidget(self.label)
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("Předchozí")
        self.btn_next = QtWidgets.QPushButton("Další")
        btn_layout.addWidget(self.btn_prev)
        btn_layout.addWidget(self.btn_next)
        self.layout.addLayout(btn_layout)

        self.btn_prev.clicked.connect(self.show_prev)
        self.btn_next.clicked.connect(self.show_next)

        # Klávesové zkratky
        self.shortcut_left = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self)
        self.shortcut_left.activated.connect(self.show_prev)
        self.shortcut_right = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self)
        self.shortcut_right.activated.connect(self.show_next)

        self.show_current()

    def show_current(self):
        ydata, xdata, lbl = self.data_list[self.idx % len(self.data_list)]
        self.label.setText(lbl)
        h, yedges, xedges = np.histogram2d(
            xdata, ydata, bins=HIST_BINS,
            range=[self.y_range, self.x_range]
        )
        h = np.log1p(h)
        self.img.setImage(h.T, levels=(0, h.max()), autoLevels=False)
        self.plot_widget.setTitle(lbl)
        self.palette = pg.colormap.get('plasma')
        self.img.setLookupTable(self.palette.getLookupTable(nPts=256, alpha=True))
        self.img.setLevels((0, h.max()))
        self.plot_widget.setLabel('bottom', 'Integral of Pulse [square of the step of A/D converter]')
        self.plot_widget.setLabel('left', 'Amplitude of Pulse [step of A/D converter]')
        self.plot_widget.setLimits(xMin=0, xMax=h.shape[0], yMin=0, yMax=h.shape[1])
        self.plot_widget.setAspectLocked(False)

        # # Add color bar
        # color_bar = pg.ColorBarItem(interactive=False, values=(0, h.max()))
        # color_bar.setColorMap(self.palette)
        # color_bar.setImageItem(self.img)
        # self.plot_widget.getPlotItem().layout.addItem(color_bar, row=1, col=1)

    def show_next(self):
        self.idx = (self.idx + 1) % len(self.data_list)
        self.show_current()

    def show_prev(self):
        self.idx = (self.idx - 1) % len(self.data_list)
        self.show_current()

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = HistogramWindow(data_list, (y_min, y_max), (x_min, x_max))
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
