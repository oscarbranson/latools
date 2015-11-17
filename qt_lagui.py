import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui as qt
# from pyqtgraph.Qt import QtCore as qc

pg.setConfigOption('background', 'w')

version = '0.1a'


class analyse(object):
    """
    Dummy analyse class for gui development
    """
    def __init__(self, *args):
        self.samples = np.array(['Sample A', 'Sample B'])
        self.analytes = np.array(['a', 'b'])
        self.data_dict = {'Sample A': D(m=1.),
                          'Sample B': D(m=2.)}
        self.cmap = self.data_dict['Sample A'].cmap


class D(object):
    """
    Dummy laser data class for gui development
    """
    def __init__(self, m, *args):
        self.Time = np.linspace(0, 2 * np.pi, 500)
        self.focus = {'a': np.sin(self.Time / m),
                      'b': np.cos(self.Time / m)}
        self.cmap = {'a': 'blue',
                     'b': 'red'}


class lagui(qt.QMainWindow):
    def __init__(self):
        super(lagui, self).__init__()
        self.initUI()

    def initUI(self):
        # set window title
        self.setWindowTitle('Laser Ablation TOOLS (LATOOLS) -- Version ' + version)
        self.initMenuBar()  # initialise menu bars

        # set layout
        self.mainPanel = qt.QWidget()
        self.grid = qt.QGridLayout()
        self.mainPanel.setLayout(self.grid)
        self.setCentralWidget(self.mainPanel)

        # dummy data
        self.dat = analyse()
        self.live_sample = 'Sample A'

        # checkboxes
        self.analyte_switches = {}
        for a in self.dat.analytes:
            self.analyte_switches[a] = qt.QCheckBox(checked=False,
                                                    text=a)
            self.analyte_switches[a].stateChanged.connect(self.updatePlot)

        # initialise panes
        self.p_opt = OptionsPane(self)
        self.p_plot = PlotPane(self)
        self.p_proc = ProcessingPane(self)

        self.showMaximized()

        self.makeTraces()
        # self.updatePlot()

    def initMenuBar(self):
        # make menu bar
        self.menubar = self.menuBar()

        # file menu
        self.fileMenu = self.menubar.addMenu('&File')
        # close window
        closeAction = qt.QAction('Close', self)
        closeAction.setShortcut('Ctrl+Q')
        closeAction.setShortcut('Ctrl+W')
        closeAction.setStatusTip('Close')
        closeAction.triggered.connect(self.close)

        # load data
        loadAction = qt.QAction('Load Data', self)
        loadAction.setShortcut('Ctrl+L')
        loadAction.setStatusTip('Load Dataset')
        # loadAction.triggered.connect(self.loadData)

        # populate
        self.fileMenu.addAction(loadAction)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(closeAction)

        # edit menu
        # self.editMenu = self.menubar.addMenu('&Edit')

        # processing menu
        self.editMenu = self.menubar.addMenu('&Processing')

        # help menu
        self.helpMenu = self.menubar.addMenu('&Help')

    def makeTraces(self):
        self.traces = {}
        for s in self.dat.samples:
            self.traces[s] = {}
            for a, v in self.analyte_switches.items():
                d = self.dat.data_dict[s]
                self.traces[s][a] = pg.PlotDataItem(x=d.Time,
                                                    y=d.focus[a],
                                                    pen=qt.QColor(d.cmap[a]))

    def updatePlot(self):
        self.plt.clear()
        self.plt.setLabels(title=self.live_sample)
        for k, v in self.analyte_switches.items():
            if v.isChecked():
                self.plt.addItem(self.traces[self.live_sample][k])


class ProcessingPane(qt.QFrame):
    def __init__(self, parent):
        super(ProcessingPane, self).__init__(parent)
        self.parent = parent
        self.initPane()
        self.addPane()

    def initPane(self):
        cdirButton = qt.QPushButton("Choose Directory")
        cdirButton.clicked.connect(self.chooseDir)

        self.toolbar = qt.QHBoxLayout()
        self.toolbar.addWidget(cdirButton)
        self.toolbar.addStretch(1)

        vbox = qt.QVBoxLayout()
        vbox.addLayout(self.toolbar)

        self.setLayout(vbox)

    def chooseDir(self):
        dialog = qt.QFileDialog(self)
        dialog.setFileMode(qt.QFileDialog.Directory)
        dialog.setOption(qt.QFileDialog.ShowDirsOnly)
        dialog.exec_()

        self.parent.wdir = dialog.selectedFiles()[0]
        self.toolbar.insertWidget(1, qt.QLabel(text=self.parent.wdir))

    def addPane(self):
        self.setStyleSheet("border: 1px solid green")
        self.parent.grid.addWidget(self, 0, 0, 1, 8)


class OptionsPane(qt.QFrame):
    def __init__(self, parent):
        super(OptionsPane, self).__init__(parent)
        self.parent = parent
        self.initPane()
        self.addPane()

    def initPane(self):

        checkboxes = qt.QVBoxLayout()
        checkboxes.addWidget(qt.QLabel(text='Analytes'))
        for v in self.parent.analyte_switches.values():
            checkboxes.addWidget(v)
        checkboxes.addStretch(1)

        samlist = qt.QComboBox()
        for s in self.parent.dat.samples:
            samlist.addItem(s)
        samlist.activated[str].connect(self.onSampleActivated)

        layout = qt.QVBoxLayout()
        layout.addLayout(checkboxes)
        layout.addWidget(qt.QLabel(text='Samples'))
        layout.addWidget(samlist)
        layout.addStretch(1)

        self.setLayout(layout)

    def onSampleActivated(self, sample):
        self.parent.live_sample = sample
        self.parent.updatePlot()
        return

    def addPane(self):
        self.setStyleSheet("border: 1px solid blue")
        self.parent.grid.addWidget(self, 1, 0, 5, 2)
        return


class PlotPane(pg.PlotWidget):
    def __init__(self, parent):
        super(PlotPane, self).__init__(parent)

        # self.makePlot(parent)
        self.addPane(parent)
        self.plot()
        self.enableAutoRange(False)

        parent.plt = self

    # def makePlot(self, parent):
    #     parent.plt = self.plot()

        # d = parent.dat.data_dict[parent.live_sample]

        # for a in parent.dat.analytes:
        #     if parent.analyte_switches[a].isChecked():
        #         self.plot(d.Time, d.focus[a],
        #                   pen=d.cmap[a],
        #                   antialias=True)

    def addPane(self, parent):
        parent.grid.addWidget(self, 1, 2, 5, 6)

# class PlotPane(qt.QWidget):
#     def __init__(self, parent):
#         super(PlotPane, self).__init__(parent)

# def main


if __name__ == '__main__':
    import sys
    app = qt.QApplication(sys.argv)
    lagui = lagui()
    sys.exit(app.exec())


# Useful Stuff Dump

# self.showFullScreen()  # makes the widget full screen
