#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
from PyQt4 import QtGui

class Test(QtGui.QWidget):

    def __init__(self):
        super(Test, self).__init__()

        self.initUI()

    def initUI(self):
        pixmap = QtGui.QPixmap("/Users/lukasfischer/github/in_vivo/MTH3/Figures/LF170110_2_Day201748_1_roi_6.png")

        print(pixmap.isNull())

        lbl = QtGui.QLabel(self)
        lbl.setPixmap(pixmap)

        self.move(300, 200)
        self.setWindowTitle('Test')
        self.show()

print(os.getcwd())
app = QtGui.QApplication(sys.argv)
ex = Test()
sys.exit(app.exec_())
