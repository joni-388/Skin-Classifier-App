# code based on https://github.com/nicknochnack/FaceIDApp/blob/main/layers.py


from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.textinput import TextInput

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

from kivy.lang import Builder

import os

# Import other dependencies
import cv2
import torch
import os
import numpy as np
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms 

# from plyer import filechooser

# import own dependencies
import model
from classifier_functions import get_x_percent_boundary,mahalanobis_distance,pool_and_flatten

# class WindowManager(ScreenManager):
#     pass

class ClassifierView(GridLayout):
    def __init__(self, **kw):
        super().__init__(**kw)
    
        self.cols = 1
        # self.size_hint = (0.6, 0.7)
        # self.pos_hint = {
        #     'center_x': 0.5,
        #     'center_y': 0.5
        # }


        # Main layout components 
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="classify", on_press=self.classify, size_hint=(1,.1))
        self.classification_label = Label(text="Classification Uninitiated", size_hint=(1,.1))
        self.load_file_button = Button(text="load file from drive", on_press=self.load_file_from_drive, size_hint=(1,.1))

        # self.filechooser = FileChooser()
        # Add items to layout
        # layout = BoxLayout(orientation='vertical')
        self.add_widget(self.web_cam)
        self.add_widget(self.button)
        self.add_widget(self.classification_label)
        self.add_widget(self.load_file_button)


        layout_bottom = BoxLayout(orientation='horizontal')
        # left 
        layout_left= BoxLayout(orientation='vertical')
        layout_left1 = BoxLayout(orientation='horizontal')
        # from drive
        def on_checkbox_active_from_drive(checkbox, value):
            if value:
                app.source = 'drive'
            else:
                app.source = 'webcam'
        self.box_from_drive = CheckBox()
        self.box_from_drive.bind(active=on_checkbox_active_from_drive)
        label_box_from_drive = Label(text="Use drive")
        layout_left1.add_widget(label_box_from_drive)
        layout_left1.add_widget(self.box_from_drive)
        # monte carlo dropout
        layout_left2 = BoxLayout(orientation='horizontal')
        label_mcd = Label(text="Monte Carlo Dropout")
        self.box_mcd = CheckBox()
        def on_checkbox_active_mcd(checkbox, value):
            if value:
                app.use_mcd = True
            else:
                app.use_mcd = False
        self.box_mcd.bind(active=on_checkbox_active_mcd)
        layout_left2.add_widget(label_mcd)
        layout_left2.add_widget(self.box_mcd)
        # def on_text(instance, value):
        #     app.mc_text_value = value
        #     print( 'Text:', value)
        # textinput = TextInput()
        # textinput.bind(text=on_text)
        # layout_left2.add_widget(textinput)

        layout_left.add_widget(layout_left1)
        layout_left.add_widget(layout_left2)

        layout_right = BoxLayout()
        layout_right1 = BoxLayout(orientation="vertical")
        button_ood = Button(text="Perform OOD", on_press=self.ood)
        self.label_ood = Label(text="OOD result: not calculated")
        layout_right1.add_widget(button_ood)
        layout_right1.add_widget(self.label_ood)

        layout_right.add_widget(layout_right1)

        layout_bottom.add_widget(layout_left)
        layout_bottom.add_widget(layout_right)

        self.add_widget(layout_bottom)

        # Load model
        self.model = model.SkinLesionClassifier(num_classes=7)
        self.model
        path_final_model = r"model_epoch_5.pth" # r"windows/saved/model_checkpoints/model_epoch_5"
        path_final_model = r"mobilenetv3b64_epoch_10.pth"
        self.model.load_state_dict(torch.load(path_final_model,map_location=torch.device('cpu'))['model_state_dict'])
        # checkpoint = torch.load(path_final_model,map_location=torch.device('cpu'))
        # self.model.load_state_dict(checkpoint['model_state_dict'],map_location=torch.device('cpu'))
        self.model.eval()

        self.label_dict = {'akiec' : 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv':5, 'vasc':6}
        self.label_dict = {y: x for x, y in self.label_dict.items()}

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update_web_cam, 1.0/33.0)
        
        return 
    
    def ood(self, *args):
        # get sample
        _, feature_of_sample = self.classify()
        if feature_of_sample is None:
            print("features for ood are none")
            return
        # load mean
        feature_mean = torch.load('gaussian_mean.pt',map_location=torch.device('cpu'))
        # load sigma
        feature_covariance_matrix = torch.load('robust_covariance_matrix.pt',map_location=torch.device('cpu'))
        # load distances 
        mahalanobis_distances = torch.load('robust_mahalanobis_distances.pt',map_location=torch.device('cpu'))

        # calc threshold 
        percentile = 95
        ood_boundary_global = get_x_percent_boundary(mahalanobis_distances,percentile )
 
        # calc distanc to sample
        feature_of_sample = pool_and_flatten(feature_of_sample, 2, 2) 
        mahalanobis_distances_for_sample = mahalanobis_distance(feature_mean, feature_covariance_matrix, feature_of_sample)
        
        # in or out distribution
        is_ood_for_closest = mahalanobis_distances_for_sample > ood_boundary_global
     
        # print result to label
        if not is_ood_for_closest:
            self.label_ood.text = "OOD result: in distribution"
        else:
            self.label_ood.text = "OOD result: out of distribution"

        return 

    # Run continuously to get webcam feed
    def update_web_cam(self, *args):

        if app.source == 'webcam':

            # Read frame from opencv
            ret, frame = self.capture.read()
            frame = frame[15:15+450, 10:10+600, :] # shape: 250,250,3

            # Flip horizontall and convert image to texture
            flip = cv2.flip(frame,0)
            # flip = cv2.flip(flip,1)
            buf = flip.tostring()
            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.web_cam.texture = img_texture

        elif app.source == 'drive':
            if app.file_path_from_selection is not None:
                # ret, frame = self.capture.read()
                frame = cv2.imread(app.file_path_from_selection, 1)
                frame = frame[15:15+450, 10:10+600, :]
                flip = cv2.flip(frame,0)
                buf = flip.tostring()
                img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.web_cam.texture = img_texture
        else:
            raise ValueError('wrong source')
        

    def get_image_to_classify_as_tensor(self, *args):
# self.classification_label.text = "pressed"
        SAVE_PATH = 'input_image.jpg'

        if app.source == 'webcam':
            # Capture input image from our webcam
            
            ret, frame = self.capture.read()
            frame = frame[15:15+450, 10:10+600, :] # 250 x 250 to 600 x 450 
            flip = cv2.flip(frame,0)
            #flip = cv2.flip(flip,1)
            cv2.imwrite(SAVE_PATH, frame)

        elif app.source == 'drive':
            if app.file_path_from_selection is not None:
                # ret, frame = self.capture.read()
                frame = cv2.imread(app.file_path_from_selection, 1)
                # frame = frame[15:15+450, 10:10+600, :]
                # flip = cv2.flip(frame,0)
                flip = frame
                flip = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)

                cv2.imwrite(SAVE_PATH, frame)
            else:
                self.classification_label.text = 'no path choosen'
                return 
        else:
            raise ValueError('wrong source for classification')



        
        transform = transforms.Compose([ 
            transforms.ToTensor() 
        ]) 
        
        # Convert the image to Torch tensor 
        tensor = transform(flip).unsqueeze(0) 
        tensor_image = tensor*255

        return tensor_image


    # classification function to classify skin image
    def classify(self, *args):
        self.model.eval()
        tensor = self.get_image_to_classify_as_tensor()
        if tensor is None:
            return (None,None)
        
        # no mcd
        with torch.no_grad():
            # # classify
            logits = self.model(tensor)
            _,pred = torch.max(F.softmax(logits,dim=1),1)

        features = self.model.intermediate_features.squeeze(0,2,3)

        # with mcd
        if app.use_mcd:
            forward_runs = 5
            for m in self.model.modules():
                if isinstance(m, nn.Dropout): #for monte carlo dropout
                    m.train()

            logits = torch.zeros((1,7))

            for k in range(forward_runs):
                with torch.no_grad():
                    # # classify
                    logits += self.model(tensor)
            
            logits = logits/forward_runs
            _,pred = torch.max(F.softmax(logits,dim=1),1)
            

        label = self.label_dict[pred.item()]
        self.classification_label.text = label

        return label,features
    
    def load_file_from_drive(self, *args):
        # MyWidget()
        app.screen_manager.current = 'Files'
        return
    pass

class FileChooserView(GridLayout):
    def __init__(self, **kw):
        super().__init__(**kw)
        
        self.cols = 2
        self.size_hint = (0.6, 0.7)
        self.pos_hint = {
            'center_x': 0.5,
            'center_y': 0.5
        }

        self.filechooser = FileChooserListView()
        # filechooser.bind(on_selection=self.selected(filechooser.selection))
        self.filechooser.path =  r"C:\Users\joni-\Desktop\sample images" #r"C:\Users\joni-"
        self.add_widget(self.filechooser)
        self.button = Button(text="open", on_press=self.open, size_hint=(1,.1))
        self.add_widget(self.button)

    def open(self, *args):
        app.screen_manager.current = 'Classifier'
        path = self.filechooser.path
        filename = self.filechooser.selection

        app.file_path_from_selection = os.path.join(path, filename[0])
        print(app.file_path_from_selection)
        # app.screen_manager.transition = app.screen_manager.transition.direction[0]
        # with open(os.path.join(path, filename[0])) as f:
        #     print(f.read())
        # pass

    def selected(self, filename, *args):
        print("selected: %s" % filename)
        pass
    pass





class SkinClassifierApp(App):
    source = 'webcam'
    file_path_from_selection = None
    use_mcd = False

    def build(self):
        self.screen_manager = ScreenManager()
        self.classifier_view = ClassifierView()
        self.fileChooser_view = FileChooserView()

        classifier_screen = Screen(name='Classifier')
        classifier_screen.add_widget(self.classifier_view)
        files_screen = Screen(name='Files')
        files_screen.add_widget(self.fileChooser_view)

        self.screen_manager.add_widget(classifier_screen)
        self.screen_manager.add_widget(files_screen)

        self.screen_manager.current = 'Classifier'
        
        return self.screen_manager


if __name__ == "__main__":
    app = SkinClassifierApp()
    app.run()