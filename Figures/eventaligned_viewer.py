"""
little GUI to look at plots of different conditions simultaenously

author: Lukas Fischer

"""
import sys
import os
from PyQt4 import QtGui, QtCore

IMAGE_SIZE = 830

class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(0,0,3500,1500)

        self.lbl_ONSET = QtGui.QLabel(self)
        self.lbl_LMCENTER = QtGui.QLabel(self)
        # self.lbl_LMOFF = QtGui.QLabel(self)
        self.lbl_REWARD = QtGui.QLabel(self)
        self.lbl_FULL = QtGui.QLabel(self)
        self.lbl_FULLOL = QtGui.QLabel(self)

        self.lbl_ONSET.move(0,20)
        self.lbl_LMCENTER.move(830,20)
        # self.lbl_LMOFF.move(1260,20)
        self.lbl_REWARD.move(1640,20)
        self.lbl_FULL.move(0,900)
        self.lbl_FULLOL.move(0,1100)

        self.lbl_ONSET.resize(IMAGE_SIZE,IMAGE_SIZE)
        self.lbl_LMCENTER.resize(IMAGE_SIZE,IMAGE_SIZE)
        # self.lbl_LMOFF.resize(IMAGE_SIZE,IMAGE_SIZE)
        self.lbl_REWARD.resize(IMAGE_SIZE,IMAGE_SIZE)
        self.lbl_FULL.resize(2500,300)
        self.lbl_FULLOL.resize(2500,300)

        self.lbl_ONSET.setFocus()

        self.selected_directory = ''
        self.selected_directory_ONSET = ''
        self.selected_directory_LMCENTER = ''
        # self.selected_directory_LMOFF = ''
        self.selected_directory_REWARD = ''
        self.selected_directory_FULL = ''
        self.selected_directory_FULLOL = ''
        self.selected_directory_filelist_ONSET = []
        self.selected_directory_filelist_LMCENTER = []
        # self.selected_directory_filelist_LMOFF = []
        self.selected_directory_filelist_REWARD = []
        self.selected_directory_filelist_FULL = []
        self.selected_directory_filelist_FULLOL = []
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
        pixmap_ONSET = QtGui.QPixmap(os.path.join(self.selected_directory_ONSET, self.selected_directory_filelist_ONSET[self.current_roi_view]))
        pixmap_LMCENTER = QtGui.QPixmap(os.path.join(self.selected_directory_LMCENTER, self.selected_directory_filelist_LMCENTER[self.current_roi_view]))
        # pixmap_LMOFF = QtGui.QPixmap(os.path.join(self.selected_directory_LMOFF, self.selected_directory_filelist_LMOFF[self.current_roi_view]))
        pixmap_REWARD = QtGui.QPixmap(os.path.join(self.selected_directory_REWARD, self.selected_directory_filelist_REWARD[self.current_roi_view]))
        pixmap_FULL = QtGui.QPixmap(os.path.join(self.selected_directory_FULL, self.selected_directory_filelist_FULL[self.current_roi_view]))
        pixmap_FULLOL = QtGui.QPixmap(os.path.join(self.selected_directory_FULLOL, self.selected_directory_filelist_FULLOL[self.current_roi_view]))

        pixmap_ONSET = pixmap_ONSET.scaledToWidth(IMAGE_SIZE)
        pixmap_LMCENTER = pixmap_LMCENTER.scaledToWidth(IMAGE_SIZE)
        # pixmap_LMOFF = pixmap_LMOFF.scaledToWidth(IMAGE_SIZE)
        pixmap_REWARD = pixmap_REWARD.scaledToWidth(IMAGE_SIZE)
        pixmap_FULL = pixmap_FULL.scaledToWidth(2500)
        pixmap_FULLOL = pixmap_FULLOL.scaledToWidth(2500)

        self.lbl_ONSET.setPixmap(pixmap_ONSET)
        self.lbl_LMCENTER.setPixmap(pixmap_LMCENTER)
        # self.lbl_LMOFF.setPixmap(pixmap_LMOFF)
        self.lbl_REWARD.setPixmap(pixmap_REWARD)
        self.lbl_FULL.setPixmap(pixmap_FULL)
        self.lbl_FULLOL.setPixmap(pixmap_FULLOL)

        self.lbl_ONSET.setFocus()
        self.show()

    def close_application(self):
        print("Exit dfloc viewer")
        sys.exit()

    def open_ROI_directory(self):
        self.selected_directory = QtGui.QFileDialog.getExistingDirectory(self, 'Select directory', '/Users/lukasfischer/Work/exps/MTH3/figures')
        if not self.selected_directory == '':
            # when a given directory is selected, we need to strip the suffix from it, so we can then add the appropriate suffix to the respective dirs
            [pathname, dirname] = os.path.split(self.selected_directory)
            dirname_split = dirname.split('_')
            dirname_no_suffix = '_'.join(dirname_split[:-1])

            self.selected_directory = pathname + dirname_no_suffix
            self.selected_directory_ONSET = pathname + os.sep + dirname_no_suffix + '_trialonset'
            self.selected_directory_LMCENTER = pathname + os.sep + dirname_no_suffix + '_lmcenter'
            # self.selected_directory_LMOFF =  pathname + os.sep + dirname_no_suffix + '_lmoff'
            self.selected_directory_REWARD = pathname + os.sep + dirname_no_suffix + '_reward'
            self.selected_directory_FULL =  pathname + os.sep + dirname_no_suffix + '_fulltrace'
            self.selected_directory_FULLOL = pathname + os.sep + dirname_no_suffix + '_fulltraceol'

            self.update_ROI_directory()

    def directory_box_edited(self):
        self.selected_directory = self.directory_box.text()
        self.selected_directory_OL = self.selected_directory + '_openloop'
        self.update_ROI_directory()

    def update_ROI_directory(self):
        self.selected_directory_filelist_ONSET = []
        self.selected_directory_filelist_LMCENTER = []
        # self.selected_directory_filelist_LMOFF = []
        self.selected_directory_filelist_REWARD = []
        self.selected_directory_filelist_FULL = []
        self.selected_directory_filelist_FULLOL = []
        self.current_roi_view = 0

        self.directory_box.setText(self.selected_directory)

        for files in os.listdir(self.selected_directory_ONSET):
            if os.path.isfile(os.path.join(self.selected_directory_ONSET, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std' and files.split('_')[0] != 'roi':
                self.selected_directory_filelist_ONSET.append(files)

        for files in os.listdir(self.selected_directory_LMCENTER):
            if os.path.isfile(os.path.join(self.selected_directory_LMCENTER, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std' and files.split('_')[0] != 'roi':
                self.selected_directory_filelist_LMCENTER.append(files)

        # for files in os.listdir(self.selected_directory_LMOFF):
        #     if os.path.isfile(os.path.join(self.selected_directory_LMOFF, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std':
        #         self.selected_directory_filelist_LMOFF.append(files)

        for files in os.listdir(self.selected_directory_REWARD):
            if os.path.isfile(os.path.join(self.selected_directory_REWARD, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std' and files.split('_')[0] != 'roi':
                self.selected_directory_filelist_REWARD.append(files)

        for files in os.listdir(self.selected_directory_FULL):
            if os.path.isfile(os.path.join(self.selected_directory_FULL, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std' and files.split('_')[0] != 'roi':
                self.selected_directory_filelist_FULL.append(files)

        for files in os.listdir(self.selected_directory_FULLOL):
            if os.path.isfile(os.path.join(self.selected_directory_FULLOL, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std' and files.split('_')[0] != 'roi':
                self.selected_directory_filelist_FULLOL.append(files)

        # here we are ordering the rois in ascending order
        roi_nrs = []
        for fn in self.selected_directory_filelist_ONSET:
            fn_split = fn.split('_')
            if len(fn_split) > 5:
                roi_nrs.append(fn_split[5].split('.')[0])
            else:
                roi_nrs.append(fn_split[4].split('.')[0])
        self.selected_directory_filelist_ONSET = [x for _,x in sorted(zip(roi_nrs,self.selected_directory_filelist_ONSET))]

        roi_nrs = []
        for fn in self.selected_directory_filelist_LMCENTER:
            fn_split = fn.split('_')
            if len(fn_split) > 5:
                roi_nrs.append(fn_split[5].split('.')[0])
            else:
                roi_nrs.append(fn_split[4].split('.')[0])
        self.selected_directory_filelist_LMCENTER = [x for _,x in sorted(zip(roi_nrs,self.selected_directory_filelist_LMCENTER))]

        # roi_nrs = []
        # for fn in self.selected_directory_filelist_LMOFF:
        #     fn_split = fn.split('_')
        #     if fn_split[0] != 'std' and fn_split[0] != '.DS':
        #         roi_nrs.append(fn_split[5].split('.')[0])
        # self.selected_directory_filelist_LMOFF = [x for _,x in sorted(zip(roi_nrs,self.selected_directory_filelist_LMOFF))]

        roi_nrs = []
        for fn in self.selected_directory_filelist_REWARD:
            fn_split = fn.split('_')
            if len(fn_split) > 5:
                roi_nrs.append(fn_split[5].split('.')[0])
            else:
                roi_nrs.append(fn_split[4].split('.')[0])
        self.selected_directory_filelist_REWARD = [x for _,x in sorted(zip(roi_nrs,self.selected_directory_filelist_REWARD))]

        roi_nrs = []
        for fn in self.selected_directory_filelist_FULL:
            fn_split = fn.split('_')
            if len(fn_split) > 5:
                roi_nrs.append(fn_split[5].split('.')[0])
            else:
                roi_nrs.append(fn_split[4].split('.')[0])
        self.selected_directory_filelist_FULL = [x for _,x in sorted(zip(roi_nrs,self.selected_directory_filelist_FULL))]

        roi_nrs = []
        for fn in self.selected_directory_filelist_FULLOL:
            fn_split = fn.split('_')
            if len(fn_split) > 5:
                roi_nrs.append(fn_split[5].split('.')[0])
            else:
                roi_nrs.append(fn_split[4].split('.')[0])
        self.selected_directory_filelist_FULLOL = [x for _,x in sorted(zip(roi_nrs,self.selected_directory_filelist_FULLOL))]

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
