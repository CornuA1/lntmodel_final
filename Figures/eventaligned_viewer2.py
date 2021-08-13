"""
little GUI to look at plots of different conditions simultaenously

author: Lukas Fischer

"""
import sys
import os
from PyQt4 import QtGui, QtCore

IMAGE_SIZE = 500

class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        # self.setGeometry(0,0,2560,1440)
        print('number of available screens: ' + str(QtGui.QDesktopWidget().numScreens()))
        for i in range(QtGui.QDesktopWidget().numScreens()):
            print(QtGui.QDesktopWidget().availableGeometry(screen=i))

        # by default show GUI on 'last' screen in list, get x and y dimensions
        disp_screen = QtGui.QDesktopWidget().numScreens()-1
        self.screen_left = QtGui.QDesktopWidget().availableGeometry(screen=disp_screen).x()
        self.screen_top = QtGui.QDesktopWidget().availableGeometry(screen=disp_screen).y()
        self.screen_width = QtGui.QDesktopWidget().availableGeometry(screen=disp_screen).width()
        self.screen_height = QtGui.QDesktopWidget().availableGeometry(screen=disp_screen).height() - 100
        print(self.screen_left, self.screen_top, self.screen_width, self.screen_height)
        self.move(self.screen_left,self.screen_top)
        self.setGeometry(self.screen_left,self.screen_top,self.screen_width,self.screen_height)

        self.directory_box = QtGui.QLineEdit(self)
        self.directory_box.move(0,00)
        self.directory_box.resize(1420,20)
        self.directory_box.returnPressed.connect(self.directory_box_edited)

        self.lbl_ONSET = QtGui.QLabel(self)
        self.lbl_LMCENTER = QtGui.QLabel(self)
        self.lbl_REWARD = QtGui.QLabel(self)
        self.lbl_ONSET_OL = QtGui.QLabel(self)
        self.lbl_LMCENTER_OL = QtGui.QLabel(self)
        self.lbl_REWARD_OL = QtGui.QLabel(self)
        self.lbl_FULL = QtGui.QLabel(self)
        self.lbl_FULLOL = QtGui.QLabel(self)

        int(((self.screen_height/4)*1.5)+20)

        self.lbl_ONSET.move(0,20)
        self.lbl_LMCENTER.move(int(self.screen_width/3),20)
        self.lbl_REWARD.move(int((self.screen_width/3)*2),20)
        self.lbl_ONSET_OL.move(0,int(((self.screen_height/4)*1.5)+20))
        self.lbl_LMCENTER_OL.move(int(self.screen_width/3),int(((self.screen_height/4)*1.5)+20))
        self.lbl_REWARD_OL.move(int((self.screen_width/3)*2),int(((self.screen_height/4)*1.5)+20))
        self.lbl_FULL.move(0,int(((self.screen_height/4)*3)+20))
        self.lbl_FULLOL.move(0,int(((self.screen_height/4)*3)+170))

        # set the label size
        self.lbl_ONSET.resize(int(self.screen_width/3),int(self.screen_height/4)*1.5)
        self.lbl_LMCENTER.resize(int(self.screen_width/3),int(self.screen_height/4)*1.5)
        self.lbl_REWARD.resize(int(self.screen_width/3),int(self.screen_height/4)*1.5)
        self.lbl_ONSET_OL.resize(int(self.screen_width/3),int(self.screen_height/4)*1.5)
        self.lbl_LMCENTER_OL.resize(int(self.screen_width/3),int(self.screen_height/4)*1.5)
        self.lbl_REWARD_OL.resize(int(self.screen_width/3),int(self.screen_height/4)*1.5)
        self.lbl_FULL.resize(1920,150)
        self.lbl_FULLOL.resize(1920,150)

        self.lbl_ONSET.setFocus()

        self.selected_directory = ''
        self.selected_directory_ONSET = ''
        self.selected_directory_LMCENTER = ''
        self.selected_directory_REWARD = ''
        self.selected_directory_ONSET_OL = ''
        self.selected_directory_LMCENTER_OL = ''
        self.selected_directory_REWARD_OL = ''
        self.selected_directory_FULL = ''
        self.selected_directory_FULLOL = ''
        self.selected_directory_filelist_ONSET = []
        self.selected_directory_filelist_LMCENTER = []
        self.selected_directory_filelist_REWARD = []
        self.selected_directory_filelist_ONSET_OL = []
        self.selected_directory_filelist_LMCENTER_OL = []
        self.selected_directory_filelist_REWARD_OL = []
        self.selected_directory_filelist_FULL = []
        self.selected_directory_filelist_FULLOL = []
        self.current_roi_view = 0

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
        # print(self.geometry())
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
        pixmap_REWARD = QtGui.QPixmap(os.path.join(self.selected_directory_REWARD, self.selected_directory_filelist_REWARD[self.current_roi_view]))
        pixmap_ONSET_OL = QtGui.QPixmap(os.path.join(self.selected_directory_ONSET_OL, self.selected_directory_filelist_ONSET_OL[self.current_roi_view]))
        pixmap_LMCENTER_OL = QtGui.QPixmap(os.path.join(self.selected_directory_LMCENTER_OL, self.selected_directory_filelist_LMCENTER_OL[self.current_roi_view]))
        pixmap_REWARD_OL = QtGui.QPixmap(os.path.join(self.selected_directory_REWARD_OL, self.selected_directory_filelist_REWARD_OL[self.current_roi_view]))
        pixmap_FULL = QtGui.QPixmap(os.path.join(self.selected_directory_FULL, self.selected_directory_filelist_FULL[self.current_roi_view]))
        pixmap_FULLOL = QtGui.QPixmap(os.path.join(self.selected_directory_FULLOL, self.selected_directory_filelist_FULLOL[self.current_roi_view]))

        pixmap_ONSET = pixmap_ONSET.scaled(int(self.screen_width/3),int((self.screen_height/4)*1.5))
        pixmap_LMCENTER = pixmap_LMCENTER.scaled(int(self.screen_width/3),int((self.screen_height/4)*1.5))
        pixmap_REWARD = pixmap_REWARD.scaled(int(self.screen_width/3),int((self.screen_height/4)*1.5))
        pixmap_ONSET_OL = pixmap_ONSET_OL.scaled(int(self.screen_width/3),int((self.screen_height/4)*1.5)-10)
        pixmap_LMCENTER_OL = pixmap_LMCENTER_OL.scaled(int(self.screen_width/3),int((self.screen_height/4)*1.5)-10)
        pixmap_REWARD_OL = pixmap_REWARD_OL.scaled(int(self.screen_width/3),int((self.screen_height/4)*1.5)-10)
        pixmap_FULL = pixmap_FULL.scaledToWidth(2500)
        pixmap_FULLOL = pixmap_FULLOL.scaledToWidth(2500)

        self.lbl_ONSET.setPixmap(pixmap_ONSET)
        self.lbl_LMCENTER.setPixmap(pixmap_LMCENTER)
        self.lbl_REWARD.setPixmap(pixmap_REWARD)
        self.lbl_ONSET_OL.setPixmap(pixmap_ONSET_OL)
        self.lbl_LMCENTER_OL.setPixmap(pixmap_LMCENTER_OL)
        self.lbl_REWARD_OL.setPixmap(pixmap_REWARD_OL)
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
            deconvolved_figures = False
            if dirname_split[-1].find('deconvolved') != -1:
                dirname_no_suffix = '_'.join(dirname_split[:-2])
                deconvolved_figures = True
                self.selected_directory = pathname + dirname_no_suffix
                self.selected_directory_ONSET = pathname + os.sep + dirname_no_suffix + '_trialonset_deconvolved'
                self.selected_directory_LMCENTER = pathname + os.sep + dirname_no_suffix + '_lmcenter_deconvolved'
                self.selected_directory_REWARD = pathname + os.sep + dirname_no_suffix + '_reward_deconvolved'
                self.selected_directory_ONSET_OL = pathname + os.sep + dirname_no_suffix + '_trialonset_deconvolved_openloop'
                print(self.selected_directory_ONSET_OL)
                self.selected_directory_LMCENTER_OL = pathname + os.sep + dirname_no_suffix + '_lmcenter_deconvolved_openloop'
                self.selected_directory_REWARD_OL = pathname + os.sep + dirname_no_suffix + '_reward_deconvolved_openloop'
                self.selected_directory_FULL =  pathname + os.sep + dirname_no_suffix + '_fulltrace'
                self.selected_directory_FULLOL = pathname + os.sep + dirname_no_suffix + '_fulltraceol'
            else:
                dirname_no_suffix = '_'.join(dirname_split[:-1])
                self.selected_directory = pathname + dirname_no_suffix
                self.selected_directory_ONSET = pathname + os.sep + dirname_no_suffix + '_trialonset'
                self.selected_directory_LMCENTER = pathname + os.sep + dirname_no_suffix + '_lmcenter'
                self.selected_directory_REWARD = pathname + os.sep + dirname_no_suffix + '_reward'
                self.selected_directory_ONSET_OL = pathname + os.sep + dirname_no_suffix + '_trialonset_openloop'
                self.selected_directory_LMCENTER_OL = pathname + os.sep + dirname_no_suffix + '_lmcenter_openloop'
                self.selected_directory_REWARD_OL = pathname + os.sep + dirname_no_suffix + '_reward_openloop'
                self.selected_directory_FULL =  pathname + os.sep + dirname_no_suffix + '_fulltrace'
                self.selected_directory_FULLOL = pathname + os.sep + dirname_no_suffix + '_fulltraceol'

            self.update_ROI_directory()

    def directory_box_edited(self):
        self.selected_directory = self.directory_box.text()
        # self.selected_directory_OL = self.selected_directory + '_openloop'
        self.update_ROI_directory()

    def update_ROI_directory(self):
        self.selected_directory_filelist_ONSET = []
        self.selected_directory_filelist_LMCENTER = []
        self.selected_directory_filelist_REWARD = []
        self.selected_directory_filelist_ONSET_OL = []
        self.selected_directory_filelist_LMCENTER_OL = []
        self.selected_directory_filelist_REWARD_OL = []
        self.selected_directory_filelist_FULL = []
        self.selected_directory_filelist_FULLOL = []
        self.current_roi_view = 0

        self.directory_box.setText(self.selected_directory)

        for files in os.listdir(self.selected_directory_ONSET):
            if os.path.isfile(os.path.join(self.selected_directory_ONSET, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std' and files.split('_')[0] != 'roi' and files.split('_')[0] != '.svg':
                self.selected_directory_filelist_ONSET.append(files)

        for files in os.listdir(self.selected_directory_LMCENTER):
            if os.path.isfile(os.path.join(self.selected_directory_LMCENTER, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std' and files.split('_')[0] != 'roi' and files.split('_')[0] != '.svg':
                self.selected_directory_filelist_LMCENTER.append(files)

        for files in os.listdir(self.selected_directory_REWARD):
            if os.path.isfile(os.path.join(self.selected_directory_REWARD, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std' and files.split('_')[0] != 'roi' and files.split('_')[0] != '.svg':
                self.selected_directory_filelist_REWARD.append(files)


        for files in os.listdir(self.selected_directory_ONSET_OL):
            if os.path.isfile(os.path.join(self.selected_directory_ONSET_OL, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std' and files.split('_')[0] != 'roi' and files.split('_')[0] != '.svg':
                self.selected_directory_filelist_ONSET_OL.append(files)

        for files in os.listdir(self.selected_directory_LMCENTER_OL):
            if os.path.isfile(os.path.join(self.selected_directory_LMCENTER_OL, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std' and files.split('_')[0] != 'roi' and files.split('_')[0] != '.svg':
                self.selected_directory_filelist_LMCENTER_OL.append(files)

        for files in os.listdir(self.selected_directory_REWARD_OL):
            if os.path.isfile(os.path.join(self.selected_directory_REWARD_OL, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std' and files.split('_')[0] != 'roi' and files.split('_')[0] != '.svg':
                self.selected_directory_filelist_REWARD_OL.append(files)


        for files in os.listdir(self.selected_directory_FULL):
            if os.path.isfile(os.path.join(self.selected_directory_FULL, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std' and files.split('_')[0] != 'roi' and files.split('_')[0] != '.svg':
                self.selected_directory_filelist_FULL.append(files)

        for files in os.listdir(self.selected_directory_FULLOL):
            if os.path.isfile(os.path.join(self.selected_directory_FULLOL, files)) and files.split('_')[0] != '.DS' and files.split('_')[0] != 'std' and files.split('_')[0] != 'roi' and files.split('_')[0] != '.svg':
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

        roi_nrs = []
        for fn in self.selected_directory_filelist_REWARD:
            fn_split = fn.split('_')
            if len(fn_split) > 5:
                roi_nrs.append(fn_split[5].split('.')[0])
            else:
                roi_nrs.append(fn_split[4].split('.')[0])
        self.selected_directory_filelist_REWARD = [x for _,x in sorted(zip(roi_nrs,self.selected_directory_filelist_REWARD))]

        # here we are ordering the rois in ascending order
        roi_nrs = []
        for fn in self.selected_directory_filelist_ONSET_OL:
            fn_split = fn.split('_')
            if len(fn_split) > 5:
                roi_nrs.append(fn_split[5].split('.')[0])
            else:
                roi_nrs.append(fn_split[4].split('.')[0])
        self.selected_directory_filelist_ONSET_OL = [x for _,x in sorted(zip(roi_nrs,self.selected_directory_filelist_ONSET_OL))]

        roi_nrs = []
        for fn in self.selected_directory_filelist_LMCENTER_OL:
            fn_split = fn.split('_')
            if len(fn_split) > 5:
                roi_nrs.append(fn_split[5].split('.')[0])
            else:
                roi_nrs.append(fn_split[4].split('.')[0])
        self.selected_directory_filelist_LMCENTER_OL = [x for _,x in sorted(zip(roi_nrs,self.selected_directory_filelist_LMCENTER_OL))]

        roi_nrs = []
        for fn in self.selected_directory_filelist_REWARD_OL:
            fn_split = fn.split('_')
            if len(fn_split) > 5:
                roi_nrs.append(fn_split[5].split('.')[0])
            else:
                roi_nrs.append(fn_split[4].split('.')[0])
        self.selected_directory_filelist_REWARD_OL = [x for _,x in sorted(zip(roi_nrs,self.selected_directory_filelist_REWARD_OL))]

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
