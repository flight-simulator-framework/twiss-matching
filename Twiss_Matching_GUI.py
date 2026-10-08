import twiss_matching
import os, sys
from datetime import datetime
import numpy as np
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QListWidgetItem, QTableWidgetItem, QScrollArea, QFrame, QHeaderView
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from machine_interfaces.Backend.LogConsole import LogConsole

TARGETS = ("beta_x", "alpha_x", "beta_y", "alpha_y", "sigma_x", "sigma_y")


class MatplotlibWidget(FigureCanvas):
    def __init__(self, parent):
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        QVBoxLayout(parent).addWidget(self)


class MainWindow(QMainWindow):
    def __init__(self, interface, dir_name):
        super().__init__()
        self.interface = interface
        self.model = getattr(interface, "tracking_interface", interface)
        uic.loadUi(os.path.join(os.path.dirname(twiss_matching.__file__), "UI files", "Twiss_Matching_GUI.ui"), self)
        self.setWindowTitle(f"Twiss Matching GUI — {interface.get_name()}")
        self._make_settings_panel_scrollable()
        self._load_logo()
        self.session_directory.setText(dir_name)
        self.momentum_in.setValue(float(getattr(self.model, "Pref", 0.0)))
        self.log_console = None
        self.canvas = MatplotlibWidget(self.plotPlaceholder)
        self.verify_canvas = MatplotlibWidget(self.verifyPlotPlaceholder)
        for table in (self.twiss_table, self.quads_table, self.iterations_table):
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.quads_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.s_positions = self._get_s_positions()
        quadrupoles = interface.get_quadrupoles()
        self.present_currents = dict(zip(quadrupoles["names"], quadrupoles["bdes"]))
        self._fill_device_list(self.quadrupoles_list, getattr(interface, "quadrupoles", []))
        self._fill_device_list(self.target_list, getattr(interface, "screens", []))
        self.target_list.setCurrentRow(self.target_list.count() - 1)
        self.log_console_button.clicked.connect(self._show_console_log)
        for signal in (self.quadrupoles_list.itemSelectionChanged, self.target_list.itemSelectionChanged,
                       self.plot_quantity_combo.currentIndexChanged,
                       *(getattr(self, f"use_{t}_target").toggled for t in TARGETS),
                       *(getattr(self, f"{t}_target").valueChanged for t in TARGETS)):
            signal.connect(lambda *_: self._refresh())
        self._refresh()
        self._draw_verification_axes()
        QTimer.singleShot(0, self.showMaximized)

    def _make_settings_panel_scrollable(self):
        scroll = QScrollArea(self.centralwidget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.centralwidget.layout().replaceWidget(self.leftGroup, scroll)
        scroll.setWidget(self.leftGroup)
        scroll.setMinimumWidth(self.leftGroup.sizeHint().width() + scroll.verticalScrollBar().sizeHint().width())

    def _load_logo(self):
        pixmap = QPixmap(os.path.join(os.path.dirname(twiss_matching.__file__), "UI files", "Assets", "CERN_logo.png"))
        if not pixmap.isNull():
            self.logo_label.setPixmap(pixmap.scaledToHeight(80, Qt.TransformationMode.SmoothTransformation))

    def _get_s_positions(self):
        try:
            positions = self.model._get_elements_positions()
        except Exception:
            return {}
        return dict(zip(map(str, positions["names"]), np.asarray(positions["S"], dtype=float)))

    def _s(self, name):
        matched = self.interface.match_screen_name(name, self.s_positions) or f"{name}_1"
        return self.s_positions.get(matched, np.nan)

    def _fill_device_list(self, list_widget, names):
        for name in sorted(map(str, names), key=lambda n: np.nan_to_num(self._s(n), nan=np.inf)):
            s = self._s(name)
            item = QListWidgetItem(f"{name}    S = {s:.3f} m" if np.isfinite(s) else name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            list_widget.addItem(item)

    def _selected(self, list_widget):
        names = [item.data(Qt.ItemDataRole.UserRole) for item in list_widget.selectedItems()]
        return sorted(names, key=lambda n: np.nan_to_num(self._s(n), nan=np.inf))

    def _refresh(self):
        quads = self._selected(self.quadrupoles_list)
        target = (self._selected(self.target_list) or ["-"])[0]
        used = [getattr(self, f"use_{t}_target").isChecked() for t in TARGETS]
        self.dof_label.setText(f"Constraints: {sum(used)} · Matching quadrupoles: {len(quads)}"
                               + (" → best compromise only" if len(quads) < sum(used) else ""))
        self.result_target_location.setText(target)
        targets = [f"{getattr(self, f'{t}_target').value():g}" if use else "-" for t, use in zip(TARGETS, used)] + ["1", "1"]
        for row, target_text in enumerate(targets):
            for column, text in enumerate((target_text, "-", "-", "-")):
                self.twiss_table.setItem(row, column, QTableWidgetItem(text))
        self.quads_table.setRowCount(len(quads))
        for row, name in enumerate(quads):
            for column, text in enumerate((name, f"{self.present_currents.get(name, np.nan):.3f}", "-", "-", "-", "-")):
                self.quads_table.setItem(row, column, QTableWidgetItem(text))
        self._draw_optics_axes(quads, target)

    def _draw_device_lines(self, ax, names, label, **style):
        for i, name in enumerate(n for n in names if np.isfinite(self._s(n))):
            ax.axvline(self._s(name), label=label if i == 0 else None, **style)

    def _draw_optics_axes(self, quads, target):
        figure = self.canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        ax.set_title(f"{self.plot_quantity_combo.currentText()} (drawn after COMPUTE MATCHING)")
        ax.set_xlabel("S [m]")
        ax.set_ylabel(("β [m]", "σ [mm]", "α")[self.plot_quantity_combo.currentIndex()])
        s_values = [s for s in self.s_positions.values() if np.isfinite(s)]
        if s_values:
            ax.set_xlim(min(s_values), max(s_values))
        self._draw_device_lines(ax, quads, "Matching quadrupoles", color="tab:blue", alpha=0.6)
        self._draw_device_lines(ax, [target], f"Target: {target}", color="red", ls="--")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def _draw_verification_axes(self):
        ax = self.verify_canvas.figure.add_subplot(111)
        ax.set_title("Beam size on the screens: measured vs predicted")
        ax.set_xlabel("S [m]")
        ax.set_ylabel("σ [mm]")
        self._draw_device_lines(ax, getattr(self.interface, "screens", []), "Screens", color="grey", ls=":")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        self.verify_canvas.draw_idle()

    def _show_console_log(self):
        if self.log_console is None:
            self.log_console = LogConsole(self)
        self.log_console.show()
        self.log_console.raise_()
        self.log_console.activateWindow()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    from machine_interfaces.Backend import SelectInterface

    interface = SelectInterface.choose_acc_and_interface()
    if interface is None:
        print("Selection cancelled.")
        sys.exit(1)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.expanduser(f"~/CERN-Flight_Simulator-Data/TM_{interface.get_name()}{time_str}_session_settings")
    w = MainWindow(interface, dir_name)
    w.show()
    sys.exit(app.exec())
