import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QMessageBox
from PIL import Image, ImageQt
import cv2
import joblib
from skimage.feature import hog
import numpy as np
import warnings

from MainWindow import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.x = 0
        self.y = 0
        self.currentQRubberBand = None
        self.originQPoint = None
        self.setupUi(self)
        # khai bao nut start
        self.openBtn.clicked.connect(self.showScreen)
        self.openBtn.setIcon(QtGui.QIcon('GUI/insert.png'))
        self.openBtn.setIconSize(QtCore.QSize(300,80))
        self.fname = ''

        self.imgScreen.setStyleSheet("padding :20px;border-image: url(GUI/empty.jpg);background-position: center center")
        self.lblResult.setStyleSheet("font: 30px Times New Roman;")

   
        warnings.filterwarnings("ignore")

    def showScreen(self):
   
        self.fname, _ = QFileDialog.getOpenFileName(self, "Open File", "",
                                                    "All Files (*);;Image Files *.jpg; *.jpeg;")
        # Hien thi anh trong label
        if self.fname != '':
            image = Image.open(self.fname)
            im = image.convert("RGBA")

            label_w = self.imgScreen.width()
            label_h = self.imgScreen.height()
            self.imgScreen.setFixedSize(label_w,label_h)
            if im.width >= label_w or im.height >= label_h:
                im = im.resize((label_w, label_h), Image.LANCZOS)
            else:
                pass
            pixmap = ImageQt.toqpixmap(im)
            self.imgScreen.setPixmap(pixmap)
            self.imgScreen.setScaledContents(True)
            self.recognition()
        else:
            pass

    def recognition(self):
        categories = ['Quần áo', 'Phụ kiện', 'Giày dép', 'Đồ chăm sóc cá nhân', 'Khác']
        filename = "Completed0_model.joblib"
        model = joblib.load(filename)

        pil_image = Image.open(self.fname).convert('L')

        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.IMREAD_GRAYSCALE)
        resizeImg = cv2.resize(opencvImage, (60, 80),interpolation =cv2.INTER_LINEAR)
        blur = cv2.GaussianBlur(resizeImg,(5,5),0)

        hog_features_test = []
        try:
            fd,hog_image = hog(blur, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,transform_sqrt=True, block_norm="L2",multichannel=True)

            hog_features_test.append(fd)
            hog_features_test = np.array(hog_features_test)
            y_pred_user = model.predict(hog_features_test)

            self.lblResult.setText(categories[int(y_pred_user)])
        except:
            print("An exception occurred")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())

