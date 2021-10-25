import random
import os
from typing import *
import sys
import torch
import numpy as np
import torchvision.transforms as T
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from torch._C import device
from torchvision.utils import make_grid

from model import latent_size, Generator
# latent_size = 10
num_columns = 6



class DoubleSlider(QSlider):

    # create our our signal that we can connect to if necessary
    doubleValueChanged = Signal(float)

    def __init__(self, decimals=3, *args, **kargs):
        super(DoubleSlider, self).__init__( *args, **kargs)
        self._multi = 10 ** decimals

        # self.valueChanged.connect(self.emitDoubleValueChanged)
        self.sliderMoved.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(value * self._multi)

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(DoubleSlider, self).setValue(int(value * self._multi))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # states
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.netG = Generator().to(self.device)
        self.netG.eval()
        self.load_weight("e_45")
        self.latent_batch = 64
        self.latents = []
        self.current_latent = None

        self.main_layout = QHBoxLayout()
        self.slider_layout = QGridLayout()

        self.sliders_scrollarea = QScrollArea()
        self.sliders: List[DoubleSlider] = []

        for i in range(latent_size):
            row = i // num_columns
            col = ((i % num_columns)+1) * 2
            s = DoubleSlider(5, Qt.Horizontal)
            s.setMaximum(8.0)
            s.setMinimum(-8.0)
            s.doubleValueChanged.connect(self.slider_val_changed_handler)
            self.sliders.append(s)
            self.slider_layout.addWidget(s, row, col)
            self.slider_layout.addWidget(QLabel(str(i)), row, col - 1)

        self.sliders_widget = QWidget()
        self.sliders_widget.setLayout(self.slider_layout)
        self.sliders_scrollarea.setWidget(self.sliders_widget)
        self.main_layout.addWidget(self.sliders_scrollarea)

        self.right_pane = QVBoxLayout()
        self.randomize_btn = QPushButton("Generate Latent")
        self.randomize_btn.clicked.connect(self.generate_latent)
        self.right_pane.addWidget(self.randomize_btn)
        self.pixmap_label = QLabel()
        self.pixmap = QPixmap()
        self.pixmap_label.setPixmap(self.pixmap)
        self.right_pane.addWidget(self.pixmap_label)
        self.main_layout.addLayout(self.right_pane)

        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        
        self.generate_latent()
        

    def set_sliders(self, data: List[float]):
        assert len(data) == len(self.sliders)
        for index, num in enumerate(data):
            self.sliders[index].setValue(num)
    
    def generate_latent(self):
        if len(self.latents) == 0 or self.current_latent is None or self.current_latent == len(self.latents) - 1:
            noise = torch.randn(64, latent_size, 1, 1)
            noise = noise.view(64, latent_size)
            self.latents = noise.tolist()
            self.current_latent = 0
        else:
            self.current_latent += 1

        self.set_sliders(self.latents[self.current_latent]) 
        # self.update_pixmap(self.get_latent())
        self.update_pixmap(self.latents[self.current_latent])

    
    def get_latent(self):
        data: List[float] = []
        for s in self.sliders:
            data.append(s.value())
        return data
    
    def slider_val_changed_handler(self):
        self.update_pixmap(self.get_latent())
    
    def update_pixmap(self, latent: List[float]):
        latent = torch.tensor(latent)
        latent = latent.view(1, latent_size, 1, 1)
        output = self.netG(latent.to(self.device))
        output = (output + 1) / 2.0
        output = output.cpu()[0].permute(1, 2, 0).contiguous()
        output = (output * 255).to(torch.uint8)
        output: np.ndarray = output.numpy()
        qimage = QImage(output.data, 128, 128, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        
        self.pixmap_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))



    def load_weight(self, name: str):
        ckpt = torch.load(os.path.join("weights", f"{name}.ckpt"))
        self.netG.load_state_dict(ckpt["net_g"])


        





app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()