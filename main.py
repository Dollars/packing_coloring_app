#!/usr/bin/python
from gi.repository import Gtk
import numpy as np

class MyWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Packing coloring visualization")
        self.set_border_width(10)

        grid = Gtk.Grid()
        self.add(grid)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=50)
        label = Gtk.Label("Select a graph", xalign=0)
        graph_type = Gtk.ComboBoxText()
        graph_type.insert(0, "0", "From file")
        graph_type.insert(1, "1", "Tree")

        hbox.pack_start(label, True, True, 0)
        hbox.pack_start(graph_type, False, True, 0)
        grid.add(hbox)

win = MyWindow()
win.connect("delete-event", Gtk.main_quit)
win.show_all()
Gtk.main()
