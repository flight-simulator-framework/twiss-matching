import twiss_matching
import os, sys
from datetime import datetime
import numpy as np
try:
    pyqt_version = 6
    from PyQt6 import uic
    from PyQt6.QtWidgets import QHeaderView, QApplication, QMainWindow, QMessageBox, QVBoxLayout, QScrollArea, QFrame, QFileDialog, QTableWidgetItem
    from PyQt6.QtCore import Qt, QTimer, QObject, QThread, pyqtSignal
    from PyQt6.QtGui import QPixmap
except ImportError:
    pyqt_version = 5
    from PyQt5 import uic
    from PyQt5.QtWidgets import QHeaderView, QApplication, QMainWindow, QMessageBox, QVBoxLayout, QScrollArea, QFrame, QFileDialog, QTableWidgetItem
    from PyQt5.QtCore import Qt, QTimer, QObject, QThread, pyqtSignal
    from PyQt5.QtGui import QPixmap
import matplotlib
from emittance_measurement.EM_helpers.ShowBeamline import ShowBeamline
from emittance_measurement.EM_helpers.DeviceLists import DeviceLists, SPositionDelegate, device_name
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from machine_interfaces.Backend.LogConsole import LogConsole
from datetime import datetime, timezone
from emittance_measurement.EM_helpers.QuadrupoleModelUpdate import load_model_quadrupoles, quadrupole_tracking_model, quadrupole_status_metadata, save_quadrupole_status

TARGETS = ("beta_x", "alpha_x", "beta_y", "alpha_y", "sigma_x", "sigma_y")


class MatplotlibWidget(FigureCanvas):
    def __init__(self, parent):
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        QVBoxLayout(parent).addWidget(self)


class MainWindow(QMainWindow, DeviceLists):
    def __init__(self, interface, dir_name):
        super().__init__()
        self.interface = interface
        self.model = getattr(interface, "tracking_interface", interface)
        uic.loadUi(os.path.join(os.path.dirname(twiss_matching.__file__), "UI files", "Twiss_Matching_GUI.ui"), self)
        self.setWindowTitle(f"Twiss Matching GUI — {interface.get_name()}")
        self._make_settings_panel_scrollable()
        self._load_logo()
        self.session = None
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
        self.beamline_view = None
        self.show_beamline_button.clicked.connect(self._show_beamline)
        self.quadrupoles_list.setItemDelegate(SPositionDelegate(self.quadrupoles_list))
        self.target_list.setItemDelegate(SPositionDelegate(self.target_list))
        quadrupoles = list(getattr(self.interface, "quadrupoles", []))
        screens_sorted = self._sorted_by_s(getattr(self.interface, "screens", []))
        self._show_s_values_and_device_lists(self.quadrupoles_list, quadrupoles)
        self._show_s_values_and_device_lists(self.target_list, screens_sorted)
        self.download_quads_button.clicked.connect(self._download_all_quads_status)
        self.load_machine_status_button.clicked.connect(self._load_model_quadrupoles_status)

    def _get_twiss_s_positions(self, names):
        names = list(names)
        positions = [np.nan] * len(names)

        if not hasattr(self.interface, "_get_elements_positions"):
            return positions

        try:
            pos = self.interface._get_elements_positions()
            pos_names = [str(name) for name in pos.get("names", [])]
            s_values = np.asarray(pos.get("S", []), dtype=float)
            lookup = {name: float(s_values[i]) for i, name in enumerate(pos_names) if
                      i < s_values.size and np.isfinite(s_values[i])}
            positions = []
            for requested_name in names:
                requested_name = str(requested_name)
                matched_name = self.interface.match_screen_name(requested_name, lookup)
                if matched_name is not None:
                    positions.append(lookup[matched_name])
                    continue
                quad_part_positions = []
                for lattice_name in (f"{requested_name}_1", f"{requested_name}_2"):
                    if lattice_name in lookup:
                        quad_part_positions.append(lookup[lattice_name])
                if quad_part_positions:
                    positions.append(min(quad_part_positions))
                else:
                    positions.append(np.nan)
        except Exception:
            positions = [np.nan] * len(names)

        return positions

    def log(self, text):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {text}"
        if self.log_console is None:
            self.log_console = LogConsole(self)
        self.log_console.log(line)

    def _show_beamline(self):
        selected_quadrupole, screens = self._get_selection()
        if self.beamline_view is None:
            self.beamline_view = ShowBeamline(interface=self.interface, parent=self, quad_selected=selected_quadrupole,
                                              screens=screens)
        self.beamline_view._display_beamline_view()
        self.beamline_view.show()
        self.beamline_view.raise_()
        self.beamline_view.activateWindow()

    def _download_all_quads_status(self):
        try:
            target_dir = self.session_directory.text().strip() or self.dir_name
            output_file_name = os.path.join(target_dir, "quadrupoles_status.npz")
            quadrupoles = list(getattr(self.interface, "quadrupoles", []))
            self.log(f"Reading {len(quadrupoles)} quadrupole currents from the machine.")
            data = dict(self.interface.get_quadrupoles(names=quadrupoles))
            data["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
            data["source_interface"] = self.interface.get_name()
            momentum = getattr(getattr(self.interface, "tracking_interface", self.interface), "Pref", None)
            if momentum is not None and np.isfinite(float(momentum)):
                data["reference_momentum_MeV_c"] = float(momentum)
            output_file_name = save_quadrupole_status(data, output_file_name)
            self._quadrupoles_status_data = data
            self.quadrupoles_status = quadrupole_status_metadata(data, output_file_name)
            self.quadrupoles_status["saved_path"] = output_file_name
            if self.session is not None:
                self.session["quadrupoles_status"] = dict(self.quadrupoles_status)
            self.log(f"Saved quadrupole readbacks to {output_file_name} at {data['captured_at_utc']}.")

        except Exception as e:
            QMessageBox.information(self, "Save quadrupoles status",
                                    f"An error occured while trying to save quadrupoles status. {e}")
            return

    def _load_model_quadrupoles_status(self):
        model = quadrupole_tracking_model(self.interface)
        print("Quadrupoles currents before RFTrack model update:")
        print(model.get_quadrupoles()["bact"])
        default_dir = self.session_directory.text().strip() or self.dir_name
        source_path, _ = QFileDialog.getOpenFileName(self, "Load quadrupoles status", default_dir,
                                                     "Quadrupole snapshots (*.npz)")
        if not source_path:
            return
        try:
            data, metadata = load_model_quadrupoles(self.interface, source_path)
        except Exception as e:
            QMessageBox.warning(self, "Load quadrupoles status", str(e))
            return
        self._model_quadrupoles_status_data = data
        self.model_quadrupoles_status = metadata
        if self.session is not None:
            self.session["model_quadrupoles_status"] = dict(metadata)
        print(f"Loaded {metadata['quadrupole_count']} quadrupoles into the RF-Track model")

        print("Quadrupoles currents after RFTrack model update:")
        print(model.get_quadrupoles()["bact"])

    def _filter_quadrupoles_in_gui(self):
        previously_selected = []
        for i in range(self.quadrupoles_list.count()):
            item = self.quadrupoles_list.item(i)
            if item.isSelected():
                previously_selected.append(item.data(Qt.ItemDataRole.UserRole) or item.text())

        if previously_selected:
            self._last_selected_quadrupoles = list(previously_selected)
        else:
            self._last_selected_quadrupoles = list(getattr(self, "_last_selected_quadrupoles", []))

        _, selected_screens = self._get_selection()
        if not selected_screens:
            return

        screen_position, screen_order_kind = self._get_element_order_values(selected_screens)
        finite_screen_positions = [float(s) for s in screen_position if np.isfinite(s)]
        if not finite_screen_positions:
            return

        first_screen_position = min(finite_screen_positions)
        all_quadrupoles = list(getattr(self.interface, "quadrupoles", []))
        quad_order, quad_order_kind = self._get_element_order_values(all_quadrupoles)

        if quad_order_kind != screen_order_kind:  # S [m] or index
            return

        quad_pos = {name: float(s) for name, s in zip(all_quadrupoles, quad_order) if np.isfinite(s)}
        valid_quadrupoles = [name for name in all_quadrupoles if
                             name in quad_pos and quad_pos[name] < first_screen_position]
        valid_previous = [q for q in getattr(self, "_last_selected_quadrupoles", []) if q in valid_quadrupoles]

        self.quadrupoles_list.blockSignals(True)
        self._show_s_values_and_device_lists(self.quadrupoles_list, valid_quadrupoles)
        if valid_previous:
            for i in range(self.quadrupoles_list.count()):
                item = self.quadrupoles_list.item(i)
                item_name = item.data(Qt.ItemDataRole.UserRole) or item.text()
                if item_name in valid_previous:
                    item.setSelected(True)
            self._last_selected_quadrupoles = list(valid_previous)
        elif valid_quadrupoles:
            closest_quad = max(valid_quadrupoles, key=lambda name: quad_pos[name])
            for i in range(self.quadrupoles_list.count()):
                item = self.quadrupoles_list.item(i)
                item_name = item.data(Qt.ItemDataRole.UserRole) or item.text()
                if item_name == closest_quad:
                    item.setSelected(True)
                    self._last_selected_quadrupoles = [closest_quad]
                    break
        elif self.quadrupoles_list.count() > 0:
            item = self.quadrupoles_list.item(0)
            if item is not None:
                item.setSelected(True)
                item_name = item.data(Qt.ItemDataRole.UserRole) or item.text()
                self._last_selected_quadrupoles = [item_name]
        else:
            self._last_selected_quadrupoles = []

        self.quadrupoles_list.blockSignals(False)

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
