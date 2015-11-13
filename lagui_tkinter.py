import tkinter as tk
import tkinter.ttk as ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from objects import *


class analysis_app(tk.Tk):
    def __init__(self, parent):
        tk.Tk.__init__(self, parent)
        self.attributes('-fullscreen', True)

        self.parent = parent

        self.initialise()

    def initialise(self):
        self.grid()

        self.analytes = ['a', 'b', 'c', 'd']

        self.options_pane()

    def options_pane(self):
        # self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        # self.grid_columnconfigure(1, weight=1)

        # define options pane
        self.opwin = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        self.opwin.grid(row=0, column=0,
                        columnspan=12, sticky='new')

        # working directory variables
        self.wdir = tk.StringVar()
        self.wdir.set('/Users/oscarbranson/UCDrive/Projects/IceSheets/Cleaning/Laser Data/151106/despiked/')
        self.wdirlab = tk.StringVar()
        self.wdirlab.set('Working Directory:')

        # Lalel with working directory
        dirlabel = ttk.Label(textvariable=self.wdirlab)
        dirlabel.grid(column=0, row=0, sticky='ew')

        # button to choose working directory
        chwdir = ttk.Button(self, text='Choose Dir...',
                            command=self.choose_dir,
                            width=10)
        chwdir.grid(column=1, row=0,
                    sticky='e')

        # button to load data
        dload = ttk.Button(self, text='Load Data',
                           command=self.data_load,
                           width=10)
        dload.grid(column=2, row=0,
                   sticky='e')

        self.opwin.add(dirlabel, weight=1)
        self.opwin.add(chwdir, weight=0)
        self.opwin.add(dload, weight=0)

    def choose_dir(self):
        self.wdir.set(tk.filedialog.askdirectory() + '/')
        self.wdirlab.set('Working Directory: ' + self.wdir.get())

    def data_load(self):
        self.dat = analyse(self.wdir.get())
        self.init_environment()

    def init_environment(self):
        self.analyte_pane()
        self.plot_pane()
        self.process_pane()

    def analyte_pane(self):
        self.anwin = ttk.PanedWindow(self,
                                     orient=tk.VERTICAL)
        self.anwin.grid(row=1, column=0,
                        columnspan=1,
                        sticky='nw')

        lab = ttk.Label(self, text='Analytes')
        lab.grid(column=0, row=1, sticky='we')
        lab.grid_columnconfigure(0, weight=1)
        self.anwin.add(lab)

        # list of analytes for plotting
        self.achecks = {}
        self.p_i = 2
        style = ttk.Style()
        for a in self.dat.analytes:
            self.achecks[a] = {'a': a,
                               'val': tk.IntVar(value=1)}
            style.configure(a + '.TCheckbutton', foreground=self.dat.cmaps[a])
            self.achecks[a]['but'] = ttk.Checkbutton(self,
                                                     variable=self.achecks[a]['val'],
                                                     text=self.achecks[a]['a'],
                                                     style=a + '.TCheckbutton',
                                                     command=self.updatePlot)
            self.achecks[a]['but'].grid(column=0, row=self.p_i, sticky='nw')
            self.p_i += 1
            self.anwin.add(self.achecks[a]['but'])

        # drop-down list of samples
        lab = ttk.Label(self, text='Samples')
        lab.grid(column=0, row=self.p_i, sticky='we')
        # lab.grid_columnconfigure(0, weight=1)
        self.anwin.add(lab)
        self.p_i += 1

        self.live_sample = tk.StringVar()
        self.live_sample.set(self.dat.data[0].sample)

        sdrop = ttk.Combobox(self, textvariable=self.live_sample)
        sdrop['values'] = tuple(self.dat.samples)
        sdrop.bind('<<ComboboxSelected>>', self.updatePlot)
        self.anwin.add(sdrop)

        # Plot options
        lab = ttk.Label(self, text='Plot Options')
        lab.grid(column=0, row=self.p_i, sticky='we')
        # lab.grid_columnconfigure(0, weight=1)
        self.anwin.add(lab)
        self.p_i += 1

        # Plot on a log scale
        self.logon = tk.IntVar(value=1)
        logswitch = ttk.Checkbutton(self, variable=self.logon,
                                    command=self.updatePlot,
                                    text='Log Scale')
        logswitch.grid(column=0, row=self.p_i,
                       sticky='we')
        self.anwin.add(logswitch)
        self.p_i += 1

        # Show Signal and Background Ranges
        self.rngon = tk.IntVar(value=0)
        rngswitch = ttk.Checkbutton(self, variable=self.rngon,
                                    command=self.updatePlot,
                                    text='Display Ranges')
        rngswitch.grid(column=0, row=self.p_i,
                       sticky='we')
        self.anwin.add(rngswitch)
        self.p_i += 1

    def process_pane(self):
        lab = ttk.Label(self, text='Data Processing')
        lab.grid(column=0, row=self.p_i, sticky='we')
        self.anwin.add(lab)
        self.p_i += 1

        # dropdown to choose discriminant analyte
        self.idan = tk.StringVar(value='Choose ID Analyte')
        iddrop = ttk.Combobox(self, textvariable=self.idan)
        iddrop['values'] = tuple(self.dat.analytes)
        self.anwin.add(iddrop)

        # background ID button
        self.sid = tk.StringVar(value='Signal ID')
        bgsel = ttk.Button(self,
                           textvariable=self.sid,
                           command=self.autorange,
                           width=10)
        bgsel.grid(column=0, row=self.p_i,
                   sticky='we')
        self.anwin.add(bgsel)
        self.p_i += 1

        # background subtract button
        self.bid = tk.StringVar(value='Background Subtract')
        bgsel = ttk.Button(self,
                           textvariable=self.bid,
                           command=self.background_subtract,
                           width=10)
        bgsel.grid(column=0, row=self.p_i,
                   sticky='we')
        self.anwin.add(bgsel)
        self.p_i += 1

        # ratio calculation

    def autorange(self, *args):
        self.dat.autorange(analyte=self.idan.get())
        self.rngon.set(1)
        self.updatePlot()
        self.sid.set(self.sid.get() + ' [Done]')

    def background_subtract(self, *args):
        self.dat.bkgcorrect()
        self.rngon.set(0)
        self.updatePlot()
        self.bid.set(self.bid.get() + ' [Done]')

    # Plotting
    def plot_pane(self):
        # self.plwin = ttk.PanedWindow(self)
        # self.plwin.grid(row=1, column=2,
        #                 rowspan=5, columnspan=4,
        #                 sticky='nsew')

        self.fig = plt.figure()
        self.plotCanvas = FigureCanvasTkAgg(self.fig, master=self)
        self.plotCanvas.show()
        self.plotCanvas.get_tk_widget().grid(column=1, row=1,
                                             sticky='nsew')
        # self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.updatePlot()

    def updatePlot(self, *args):
        self.fig.clf()  # clear figure
        self.ax = self.fig.add_axes([.1, .1, .85, .85])  # make axis
        # isolate data
        d = self.dat.data_dict[self.live_sample.get()]
        # set plot scale
        if self.logon.get() == 1:
            self.ax.set_yscale('log')
            ad = 1.
        else:
            ad = 0.
        # plot all activated traces
        for a, v in self.achecks.items():
            if v['val'].get() == 1:
                self.ax.plot(d.Time,
                             d.focus[v['a']] + ad,  # ad to remove zeros if log
                             color=d.cmap[a])
        # label plot
        self.ax.text(.02, .98, d.sample,
                     transform=self.ax.transAxes, va='top', ha='left',
                     weight='bold')
        self.ax.set_xlabel('Time (s)')

        # draw ranges, if switch is on
        if self.rngon.get() == 1:
            try:
                for b in d.bkgrng:
                    self.ax.axvspan(b[0], b[1], color=(1, 0, 0, 0.2))
                for s in d.sigrng:
                    self.ax.axvspan(s[0], s[1], color=(0, 0, 1, 0.2))
            except:
                pass

        # draw plot
        self.plotCanvas.draw()

        # ltext = tk.StringVar()
        # ltext.set('Plot Window:')

        # lab = tk.Label(self, textvariable=ltext)
        # lab.grid(column=1, row=0, sticky='nsew')
        # lab.pack(side=tk.TOP)
        # self.plwin.add(lab)




if __name__ == "__main__":
    app = analysis_app(None)
    app.title('Laser Ablations TOOLS (LATOOLS) -- Version 0.1')
    app.mainloop()

