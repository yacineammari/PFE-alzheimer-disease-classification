import os
import sys
import subprocess
import numpy as np
import pyqtgraph as pg
import SimpleITK as sitk
from pathlib import Path
from cv2 import equalizeHist
from scipy.stats import entropy
from nipype.interfaces import fsl
from PyQt5 import QtWidgets, uic ,QtGui,QtCore

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from time import sleep

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Dense
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Image():
    image_path = None
    image = None
    image_numpy = None

    def __init__(self,path):
        self.image_path = path
        self.image = sitk.ReadImage(self.image_path)
        self.image_numpy = sitk.GetArrayFromImage(self.image)

class Preprocessing_worker(QtCore.QObject):
    done_affine = QtCore.pyqtSignal()
    done_sks = QtCore.pyqtSignal()
    err_affine = QtCore.pyqtSignal(str)
    err_affine_sks = QtCore.pyqtSignal()
    err_sks = QtCore.pyqtSignal(str)
    change_progress = QtCore.pyqtSignal()
    thread_finish = QtCore.pyqtSignal()
    
    def __init__(self, parent):
        super().__init__()
        self.running = False
        self.parent = parent


    def run(self):
        if self.running == False:
            self.running = True
            try:
                self.parent.affin_registration()
                # sleep(1)
                self.done_affine.emit()
                try:
                    self.parent.skull_stripping_mask()
                    sleep(1)
                    self.done_sks.emit()
                except Exception as e:
                    print(e)
                    self.err_sks.emit(str(e))

            except Exception as e:
                print(e)
                self.err_affine.emit(str(e))
                self.err_affine_sks.emit()

            self.change_progress.emit()
            self.thread_finish.emit()
            self.running = False
            
    def stop(self):
            self.running = False

class Preprocessing(QtWidgets.QMainWindow):
    path_affine = QtCore.pyqtSignal(str)
    path_sks = QtCore.pyqtSignal(str)

    def __init__(self,parent):
        super(Preprocessing, self).__init__()
        uic.loadUi('Preprocessing.ui', self)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.parent = parent
        self.image_path = self.parent.images['Row image'].image_path
        self.base_name = os.path.basename(self.image_path)
        self.WORK_DIR = os.getcwd()
        self.SK_DIR = f'{self.WORK_DIR}/CACHE/SKULL_STRIPPING/'
        self.REG_DIR = f'{self.WORK_DIR}/CACHE//AFFINE_REGISTRATION/'
        self.MAT_DIR = f'{self.WORK_DIR}/CACHE//MAT/'

        self.flt = fsl.FLIRT()
        self.flt.inputs.dof = 12
        self.flt.inputs.output_type = "NIFTI"
        self.flt.inputs.reference = f'{self.WORK_DIR}/atlas/MNI_152/MNI152lin_T1_2mm.nii.gz'

        self.mask = fsl.ApplyMask()
        self.mask.inputs.output_type = "NIFTI"
        self.mask.inputs.mask_file = f'{self.WORK_DIR}/atlas/MNI_152/MNI152lin_T1_2mm_brain_mask.nii.gz'
       

        self.Preprocessing_worker = Preprocessing_worker(self)
        self.thread = QtCore.QThread()

        self.Preprocessing_worker.moveToThread(self.thread)
        self.thread.started.connect(self.Preprocessing_worker.run)
        self.Preprocessing_worker.done_affine.connect(self.done_affine)
        self.Preprocessing_worker.done_sks.connect(self.done_sks)
        self.Preprocessing_worker.err_affine.connect(self.err_affine)
        self.Preprocessing_worker.err_sks.connect(self.err_sks)
        self.Preprocessing_worker.err_affine_sks.connect(self.err_affine_sks)
        self.Preprocessing_worker.change_progress.connect(self.change_progress)
        self.Preprocessing_worker.thread_finish.connect(self.thread.quit)
        self.start.clicked.connect(self.run_pipline)
        self.cancel_done.clicked.connect(self.out)       
        self.init_files()

        self.show()

    def change_progress(self):
        self.progressBar.setMaximum(1)
    
    def done_affine(self):
        self.stat_affine_reg.setText('Done')
        self.stat_affine_reg.setStyleSheet('color: green;')
        self.path_affine.emit(f'{self.REG_DIR}/{self.base_name}')
        self.start.setEnabled(True)
    
    def done_sks(self):
        self.stat_sks.setText('Done')
        self.stat_sks.setStyleSheet('color: green;')
        self.path_sks.emit(f'{self.SK_DIR}/{self.base_name}')
        self.start.setEnabled(True)

    def err_affine(self,e):
        self.stat_affine_reg.setText('Error')
        self.start.setEnabled(True)
        self.box(e)

    def err_sks(self,e):
        self.stat_sks.setText('Error')
        self.start.setEnabled(True)
        self.box(e)

    def err_affine_sks(self):
        self.stat_sks.setText('Error')
        self.start.setEnabled(True)


    def out(self):
        if self.thread.isRunning() == True:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Message")
            msg.setText( "Are you sure you want to quit? Pipline is still running!")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setStandardButtons(QtWidgets.QMessageBox.Yes |  QtWidgets.QMessageBox.No)
            msg.setDefaultButton(QtWidgets.QMessageBox.No)
            msg.exec_()
            button = msg.clickedButton()
            sb = msg.standardButton(button)
            if sb == QtWidgets.QMessageBox.Yes:
                self.close()
            else:
                pass
        else:   self.close()
    
    def run_pipline(self):
        print(self.thread.isRunning())
        if self.thread.isRunning() == False:
            self.thread.start()
            self.stat_affine_reg.setText('Processing...')
            self.stat_sks.setText('Processing...')
            self.stat_affine_reg.setStyleSheet('color: red;')
            self.stat_sks.setStyleSheet('color: red;')
            self.progressBar.setMaximum(0)
            self.start.setEnabled(False)
            
        

    def init_files(self):
        if not os.path.isdir(self.SK_DIR):
            os.makedirs(self.SK_DIR)

        if not os.path.isdir(self.REG_DIR):
            os.makedirs(self.REG_DIR)
        
        if not os.path.isdir(self.MAT_DIR):
            os.makedirs(self.MAT_DIR)

    def affin_registration(self):
        ''' Apply Affin registration to a given image, and save the result as a new .nii.gz image.
        '''   
        self.flt.inputs.out_matrix_file = f'{self.MAT_DIR}{self.base_name.replace(".nii","")}.mat'
        self.flt.inputs.in_file = f'{self.image_path}'
        self.flt.inputs.out_file = f'{self.REG_DIR}{self.base_name}'
        self.flt.run()
        print('afiine done')
    
    def skull_stripping_mask(self):
        ''' Apply skull stripping by masking none brain tissues to a given image, and save the result as a new .nii.gz image.        
        ''' 
        self.mask.inputs.in_file = f'{self.REG_DIR}{self.base_name}'
        self.mask.inputs.out_file = f'{self.SK_DIR}{self.base_name}'
        self.mask.run()
        print('sks done')
    
    def box(self,e):
        messageBox = QtWidgets.QMessageBox()
        messageBox.setWindowTitle("oops something went wrong")
        messageBox.setText(str(e))
        messageBox.setIcon(QtWidgets.QMessageBox.Critical)
        messageBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        messageBox.exec()

class Classification(QtWidgets.QMainWindow):
   
    def __init__(self,sks_image):
        super(Classification, self).__init__()
        uic.loadUi('classification.ui', self)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.sks_image = sks_image 
        self.image_2d = self.build_2d_image()

        self.SHAPE=(436,364,3)
        self.EPOCH = 50
        self.METRICS = [
            tf.keras.metrics.CategoricalAccuracy(name='Accuracy'),
            tf.keras.metrics.Precision(name='AD_Precision',class_id=0),
            tf.keras.metrics.Precision(name='CN_Precision',class_id=1),
            tf.keras.metrics.Precision(name='MCI_Precision',class_id=2),
            tf.keras.metrics.Recall(name='AD_Recall',class_id=0),
            tf.keras.metrics.Recall(name='CN_Recall',class_id=1),
            tf.keras.metrics.Recall(name='MCI_Recall',class_id=2),
            ]
        self.LOOS = 'categorical_crossentropy'

        self.fig = plt.figure(dpi=100)
        self.ax = self.fig.add_subplot()
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        self.fig.tight_layout(pad=1)

        self.canvas = FigureCanvas(self.fig)
        self.fig.set_facecolor('black')

        self.layout = QtWidgets.QVBoxLayout()
        self.img_canves.setLayout(self.layout)
        self.layout.addWidget(self.canvas)


        self.fig_prob, self.ax_prob = plt.subplots(dpi=100, subplot_kw=dict(aspect="equal"))
        self.fig_prob.set_size_inches((1, 1), forward=True)
        self.ax_prob.set_axis_off()
        self.fig_prob.set_facecolor('white')

        self.canvas_prob = FigureCanvas(self.fig_prob)
        self.fig_prob.tight_layout(pad=2)


        self.layout_prob = QtWidgets.QVBoxLayout()
        self.prob_canves.setLayout(self.layout_prob)
        self.layout_prob.addWidget(self.canvas_prob)

        self.ax.imshow(self.image_2d,cmap='gray')
        self.canvas.draw() 
        self.canvas.flush_events()

        self.row = None

        self.classify.clicked.connect(self.predict_class)

        self.show()
    
    def build_2d_image(self):
        start = 25
        end = 70
        nb_img = 16
        
        array = self.sks_image.image_numpy
        array = np.interp(array, (array.min(), array.max()), (0, 255))

        graid_image = np.array([])
        data = np.array([])
        entpy_data = {}

        for i in range(start,end):
            value,counts = np.unique(array[i,:,:], return_counts=True)
            entpy_data[i] = entropy(counts, base=2)
        entpy_data = {k: v for k, v in sorted(entpy_data.items(),reverse=True, key=lambda item: item[1])}
        index_of_slices = list(entpy_data.keys())[0:nb_img]


        for i , max_indx in enumerate(index_of_slices):
            if (i+1) % 4 == 0:
                data = np.hstack((data,array[max_indx,:,:]))
                if graid_image.size < 1:
                    graid_image = data.copy()
                else:
                    graid_image = np.vstack((graid_image,data))
                data = np.array([])
                
            else:
                if data.size < 1:
                    data = array[max_indx,:,:]
                else:
                    data = np.hstack((data,array[max_indx,:,:]))
        return equalizeHist(np.uint8(graid_image))
    
    def resnet50_row(self):
        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
        image_input = tf.keras.applications.ResNet50(include_top=False,weights='imagenet',input_shape=self.SHAPE)

        for layer in image_input.layers:
            layer.trainable = True
            layer._name = layer._name  + str('_img')

        y = Flatten() (image_input.output)
        c = Dropout(0.4) (y)
        c = Dense(512) (c)
        c = Dense(256) (c)
        c = Dense(128) (c)
        output_layer = Dense(3, activation='softmax')(c)

        model = Model(inputs=image_input.input,outputs=output_layer)
        model.compile(optimizer=opt, loss=self.LOOS, metrics=self.METRICS)
        return model
    
    def predict_class(self):
        self.setCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.classify.setEnabled(False)
        if self.row == None:
            self.row = self.resnet50_row()
            self.row.load_weights('./Models/row.h5')
        img = np.expand_dims(np.stack((self.image_2d,)*3, axis=-1), axis=0).astype(np.float32)
        self.plot(list(self.row.predict(img).tolist()[0]))
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.classify.setEnabled(True)
          
    def plot(self,l):
        for i in range(len(l)):
            l[i] = round( l[i],4)
        mapper = {'AD': 0, 'CN': 1, 'MCI': 2}
        d = {}

        for indx, key in enumerate(mapper.keys()):
            d[key] = l[indx]

        d = {k: v for k, v in sorted(d.items(),reverse=True, key=lambda item: item[1])}

        recipe = list(d.keys())

        data = list(d.values())

        wedges, texts = self.ax_prob.pie(data, wedgeprops={'width':0.3,'edgecolor': 'white'}, startangle=40)

        self.ax_prob.set_title(f'PROBABILITY DISTRIBUTION \n FINAL DECISION Class is: {max(d,key=lambda x: d[x])}',pad=30)
        self.ax_prob.legend(wedges, [f'{recipe[i]}  {data[i]*100}%' for i in range(len(d.keys()))],bbox_to_anchor=(0.5, -0.05),
                    loc="upper center",fancybox=True, shadow=True )

        self.canvas_prob.draw() 
        self.canvas_prob.flush_events()

class Ui(QtWidgets.QMainWindow):
   
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('app.ui', self)
        
        # ============ Canves init ====================== #
        self.cmap = 'gray'
        self.cmap_list = ['gray','viridis', 'plasma', 'inferno', 'magma', 'cividis']

        self.ImageView = pg.ImageView(parent=self.canvas_frame)
        self.ImageView.ui.histogram.hide()
        self.ImageView.ui.roiBtn.hide()
        self.ImageView.ui.menuBtn.hide()
        self.ImageView.view.setMouseEnabled(x=False, y=False)
        
        self.layout = QtWidgets.QVBoxLayout()
        self.canvas_frame.setLayout(self.layout)
        self.layout.addWidget(self.ImageView)

        # ============ Tabel init ====================== #
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableWidget.setColumnWidth(0,100)
        self.tableWidget.setColumnWidth(1,215)
        self.tableWidget.resizeRowsToContents()
        self.tableWidget.verticalHeader().setDefaultSectionSize(75)

        # ============ Color palettes Button init ====================== #
        self.btn = QtWidgets.QPushButton('Color Palettes  ', self)
        self.btn.setToolTip('Color Palettes')
        self.btn.setIcon(QtGui.QIcon(':/icon/icon/color palette.png'))
        self.btn.setIconSize(QtCore.QSize(24,24))
        self.btn.setStyleSheet('''
        QPushButton{ border: none; height: 30%;}
        QPushButton:hover { border: 1px solid #8f8f91; border-radius: 2px;}
        ''')
        self.menu = QtWidgets.QMenu()
        self.group = QtWidgets.QActionGroup(self.menu)

        for elem in self.cmap_list:
            action = QtWidgets.QAction(elem, self.menu, checkable=True, checked=elem==self.cmap_list[0])
            self.menu.addAction(action)
            self.group.addAction(action)
        self.group.setExclusive(True)
        self.menu.triggered.connect(self.change_color)
        self.btn.setMenu(self.menu)
        self.toolBar.addWidget(self.btn)

        # ============ Active file Button init ====================== #
        self.btn_file = QtWidgets.QPushButton('Active File  ', self)
        self.btn_file.setToolTip('Active File')
        self.btn_file.setIcon(QtGui.QIcon(':/icon/icon/see.png'))
        self.btn_file.setIconSize(QtCore.QSize(24,24))
        self.btn_file.setStyleSheet('''
        QPushButton{ border: none; height: 30%;}
        QPushButton:hover { border: 1px solid #8f8f91; border-radius: 2px;}
        ''')
        self.toolBar.addWidget(self.btn_file)

        self.btn_file.setEnabled(False)
        self.actionImage_Preprocessing.setEnabled(False)
        self.actionClassification.setEnabled(False)
        
        # ============ Event listener bindings init ====================== #
        self.actionAdd_File.triggered.connect(self.add_file)
        self.actionClassification.triggered.connect(self.Classifiy)
        self.actionImage_Preprocessing.triggered.connect(self.Preprocessing)
        self.x_radio_Button.clicked.connect(self.XRadioevent)
        self.y_radio_Button.clicked.connect(self.YRadioevent)
        self.z_radio_Button.clicked.connect(self.ZRadioevent)
        self.x_horizontal_Slider.valueChanged.connect(self.XSliderValchange)
        self.y_horizontal_Slider.valueChanged.connect(self.YSliderValchange)
        self.z_horizontal_Slider.valueChanged.connect(self.ZSliderValchange)


        self.show()

    def Classifiy(self):
        self.classification = Classification(self.images['skull stripped'])
        self.classification.show()
        
    def add_file(self):
        file_path , _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select An MRI Image', '.', 'NIfTI Files(*.nii);; Zipped NIfTI Files(*.nii.gz)')
        if (file_path):
            self.images = {}

            self.images['Row image'] = Image(file_path)
            

            self.menu_file = QtWidgets.QMenu()
            self.group_file = QtWidgets.QActionGroup(self.menu_file)

            
            action = QtWidgets.QAction('Row image', self.menu_file, checkable=True, checked=True)
            self.menu_file.addAction(action)
            self.group_file.addAction(action)
            self.group_file.setExclusive(True)
            self.menu_file.triggered.connect(self.image_changed)
            self.btn_file.setMenu(self.menu_file)
            self.btn_file.setEnabled(True)

            self.active_image = 'Row image'

            self.view_control.setEnabled(True)    
            self.actionImage_Preprocessing.setEnabled(True)
            self.actionClassification.setEnabled(False)
            self.rest_control()
            self.LoadMetaData()
            self.XRadioevent() 

    def image_changed(self,e):
        self.active_image = e.text()
        self.rest_control()
        self.LoadMetaData()
        self.XRadioevent()  
    
    def rest_control(self):
        self.x_horizontal_Slider.setMinimum(0)
        self.y_horizontal_Slider.setMinimum(0)
        self.z_horizontal_Slider.setMinimum(0)

        self.x_horizontal_Slider.setMaximum(self.images[self.active_image].image_numpy.shape[2]-1)
        self.y_horizontal_Slider.setMaximum(self.images[self.active_image].image_numpy.shape[1]-1)
        self.z_horizontal_Slider.setMaximum(self.images[self.active_image].image_numpy.shape[0]-1)
        
        self.y_horizontal_Slider.setValue(int(self.images[self.active_image].image_numpy.shape[1]/2))
        self.z_horizontal_Slider.setValue(int(self.images[self.active_image].image_numpy.shape[0]/2))
        self.x_horizontal_Slider.setValue(int(self.images[self.active_image].image_numpy.shape[2]/2))

    def XRadioevent(self):
        self.Selectx()
        self.XSliderValchange()
    def YRadioevent(self):
        self.Selecty()
        self.YSliderValchange()
    def ZRadioevent(self):
        self.Selectz()
        self.ZSliderValchange()

    def Selectx(self):
        self.UnSelecty()
        self.UnSelectz()
        self.x_radio_Button.setChecked(True)
        self.x_horizontal_Slider.setEnabled(True)
    def Selecty(self):
        self.UnSelectx()
        self.UnSelectz()
        self.y_radio_Button.setChecked(True)
        self.y_horizontal_Slider.setEnabled(True)
    def Selectz(self):
        self.UnSelectx()
        self.UnSelecty()
        self.z_radio_Button.setChecked(True)
        self.z_horizontal_Slider.setEnabled(True)
    
    def UnSelectx(self):
        self.x_radio_Button.setChecked(False)
        self.x_horizontal_Slider.setEnabled(False)
    def UnSelecty(self):
        self.y_radio_Button.setChecked(False)
        self.y_horizontal_Slider.setEnabled(False)
    def UnSelectz(self):
        self.z_radio_Button.setChecked(False)
        self.z_horizontal_Slider.setEnabled(False)
    
    def XSliderValchange(self):
        val = self.x_horizontal_Slider.value()
        self.ImageView.setImage(np.rot90(self.images[self.active_image].image_numpy[:,:,val],k=3))
        self.ImageView.setColorMap(pg.colormap.get(self.cmap, source='matplotlib'))

    def YSliderValchange(self):
        val = self.y_horizontal_Slider.value()
        self.ImageView.setImage(np.rot90(self.images[self.active_image].image_numpy[:,val,:],k=3))
        self.ImageView.setColorMap(pg.colormap.get(self.cmap, source='matplotlib'))

    def ZSliderValchange(self):
        val = self.z_horizontal_Slider.value()
        self.ImageView.setImage(np.rot90(self.images[self.active_image].image_numpy[val,:,:],k=1))
        self.ImageView.setColorMap(pg.colormap.get(self.cmap, source='matplotlib'))


    def LoadMetaData(self):

        while (self.tableWidget.rowCount() > 0):
            self.tableWidget.removeRow(0)

        self.tableWidget.setRowCount(4)

        self.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem('Size') )
        self.tableWidget.setItem(0, 1, QtWidgets.QTableWidgetItem(str(self.images[self.active_image].image.GetSize())))

        self.tableWidget.setItem(1, 0, QtWidgets.QTableWidgetItem('Spacing') )
        self.tableWidget.setItem(1, 1, QtWidgets.QTableWidgetItem(str(self.images[self.active_image].image.GetSpacing())))

        self.tableWidget.setItem(2, 0, QtWidgets.QTableWidgetItem('Number Of Pixels') )
        self.tableWidget.setItem(2, 1, QtWidgets.QTableWidgetItem(str(self.images[self.active_image].image.GetNumberOfPixels())))

        self.tableWidget.setItem(3, 0, QtWidgets.QTableWidgetItem('Origin') )
        self.tableWidget.setItem(3, 1, QtWidgets.QTableWidgetItem(str(self.images[self.active_image].image.GetOrigin())))
        
    def change_color(self,e):
        self.cmap = e.text()
        self.ImageView.setColorMap(pg.colormap.get(self.cmap, source='matplotlib'))

    def Preprocessing(self):
        self.preprocessing_window = Preprocessing(self)
        self.preprocessing_window.show()
        self.preprocessing_window.path_affine.connect(self.path_affine)
        self.preprocessing_window.path_sks.connect(self.path_sks)

    def path_affine(self,path):
        self.images['Registered image'] = Image(path)
        names  = [name.text() for name in self.menu_file.actions()]
        if not ('Registered image' in names):
            action = QtWidgets.QAction('Registered image', self.menu_file, checkable=True, checked=False)
            self.menu_file.addAction(action)
            self.group_file.addAction(action)
    
    def path_sks(self,path):
        self.images['skull stripped'] = Image(path) 
        names  = [name.text() for name in self.menu_file.actions()]
        if not ('skull stripped' in names):
            action = QtWidgets.QAction('skull stripped', self.menu_file, checkable=True, checked=False)
            self.menu_file.addAction(action)
            self.group_file.addAction(action)
        self.actionClassification.setEnabled(True)

if __name__ == '__main__':
    output_file = Path('app_icon.py')
    resource_file = Path('app_icon.qrc')
    # cmd and sub process to generated new font resource file
    cmd = f'pyrcc5 {resource_file} -o {output_file}'
    subprocess.call(cmd, shell=True)
    import app_icon
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()