"""
show plots from shuffle test for openloop and vr

author: Lukas Fischer

"""
import sys
import os
from PyQt4 import QtGui, QtCore

class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(0,0,1420,1420)

        self.lbl_VR = QtGui.QLabel(self)
        self.lbl_OL = QtGui.QLabel(self)

        self.lbl_VR.move(0,20)
        self.lbl_OL.move(720,20)

        self.lbl_VR.resize(700,700)
        self.lbl_OL.resize(700,700)

        self.lbl_VR.setFocus()

        self.selected_directory = ''
        self.selected_directory_OL = ''
        self.selected_directory_filelist = []
        self.selected_directory_filelist_OL = []
        self.current_roi_view = 0

        self.directory_box = QtGui.QLineEdit(self)
        self.directory_box.move(0,0)
        self.directory_box.resize(1420,20)
        self.directory_box.returnPressed.connect(self.directory_box_edited)

        openDirAction = QtGui.QAction("&Open ROI directory", self)
        openDirAction.setShortcut("Ctrl+D")
        openDirAction.setStatusTip('Open directory containing dfloc ROI images')
        openDirAction.triggered.connect(self.open_ROI_directory)

        quitAction = QtGui.QAction("&Close viewer", self)
        quitAction.setShortcut("Ctrl+Q")
        quitAction.setStatusTip('Leave The App')
        quitAction.triggered.connect(self.close_application)

        self.statusBar()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(openDirAction)
        fileMenu.addAction(quitAction)

        self.show()
        # self.update_roi_view()


    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Right:
            self.current_roi_view = self.current_roi_view + 1
            self.update_roi_view()
        elif event.key() == QtCore.Qt.Key_Left:
            if self.current_roi_view > 0:
                self.current_roi_view = self.current_roi_view - 1
                self.update_roi_view()
        event.accept()


    def update_roi_view(self):
        """ update displayed images for ROI """
        pixmap_VR = QtGui.QPixmap(os.path.join(self.selected_directory, self.selected_directory_filelist[self.current_roi_view]))
        pixmap_OL = QtGui.QPixmap(os.path.join(self.selected_directory_OL, self.selected_directory_filelist_OL[self.current_roi_view]))

        pixmap_VR = pixmap_VR.scaledToWidth(700)
        pixmap_OL = pixmap_OL.scaledToWidth(700)

        self.lbl_VR.setPixmap(pixmap_VR)
        self.lbl_OL.setPixmap(pixmap_OL)
        self.lbl_VR.setFocus()
        self.show()

    def close_application(self):
        print("Exit dfloc viewer")
        sys.exit()

    def open_ROI_directory(self):
        self.selected_directory = QtGui.QFileDialog.getExistingDirectory(self, 'Select directory', '/Users/lukasfischer/Work/exps/MTH3/figures')
        if not self.selected_directory == '':
            print(os.path.split(self.selected_directory))
            self.selected_directory_OL = self.selected_directory + '_openloop'
            self.update_ROI_directory()

    def directory_box_edited(self):
        self.selected_directory = self.directory_box.text()
        self.selected_directory_OL = self.selected_directory + '_openloop'
        self.update_ROI_directory()

    def update_ROI_directory(self):
        self.current_roi_view = 0
        self.selected_directory_filelist = []
        self.selected_directory_filelist_OL = []
        self.directory_box.setText(self.selected_directory)

        for files in os.listdir(self.selected_directory):
            if os.path.isfile(os.path.join(self.selected_directory, files)):
                self.selected_directory_filelist.append(files)

        for files in os.listdir(self.selected_directory_OL):
            if os.path.isfile(os.path.join(self.selected_directory_OL, files)):
                self.selected_directory_filelist_OL.append(files)

        # here we are ordering the rois in ascending order
        roi_nrs = []
        for fn in self.selected_directory_filelist:
            fn_split = fn.split('_')
            roi_nrs.append(fn_split[5].split('.')[0])
        self.selected_directory_filelist = [x for _,x in sorted(zip(roi_nrs,self.selected_directory_filelist))]

        roi_nrs = []
        for fn in self.selected_directory_filelist_OL:
            roi_nrs.append(fn.split('_')[5])
        self.selected_directory_filelist_OL = [x for _,x in sorted(zip(roi_nrs,self.selected_directory_filelist_OL))]

        # self.selected_directory_filelist = sorted(self.selected_directory_filelist)
        # self.selected_directory_filelist_OL = sorted(self.selected_directory_filelist_OL)
        self.update_roi_view()

    def on_key(self, event):
        if event.key() == QtCore.Qt.Key_Enter:
            print('hello there')

def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()
