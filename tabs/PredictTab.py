from tkinter import filedialog, Frame, Button, Entry, Text, Label, Scrollbar, messagebox
from tkinter.ttk import Style, Frame as Fr
import os
from common import *
import time
import threading
import numpy as np

from sa import SA    
from component import label_with_entry, label_with_link_path_file

class PredictTab(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.current_dir = os.getcwd()
        self.parent = parent
        self.initUI()
        self.isLoadingModel = False
        self.sess = False
        return
    
    def initUI(self):
        self.style = Style()
        self.style.theme_use("default")

        paddingTop = Frame(self.parent, height=30, padx=10, pady=30)
        paddingTop.grid(row=0)

        select_model_frame = Frame(self.parent, padx=10)
        select_model_frame.grid(row=1, column=0, sticky='w')

        self.btn_select_model = Button(select_model_frame, text="Select your model", command=self.select_model, width=20) 
        self.btn_select_model.grid(row=1, column=0)

        self.entry_path_model = Text(select_model_frame, width=100, height=1)
        self.entry_path_model.grid(row=1, column=1, padx=15)
        self.entry_path_model.configure(state='normal')
        self.entry_path_model.insert('insert', 'no file is selected')
        self.entry_path_model.configure(state='disabled')

        self.label_load_model = Label(select_model_frame, text='*no model is loaded', height=5, padx=15, anchor='n', justify='left')
        self.label_load_model.grid(row=2, column=1, sticky='w')

        input_frame = Frame(self.parent, padx=10, pady=15)
        input_frame.grid(row=2, column=0, sticky='w')

        self.btn_dialog_data = Button(input_frame, text="select data file", command=self.select_data, width=20) 
        self.btn_dialog_data.grid(row=1, column=0)


        self.entry_path_data = Text(input_frame, width=100, height=1)
        self.entry_path_data.grid(row=1, column=1, pady=15)
        self.entry_path_data.configure(state='normal')
        self.entry_path_data.insert('insert', 'no file is selected')
        self.entry_path_data.configure(state='disabled')

        self.btn_dialog_labels = Button(input_frame, text="select labels file", command=self.select_labels, width=20) 
        self.btn_dialog_labels.grid(row=2, column=0, pady=15)

        self.entry_path_labels = Text(input_frame, width=100, height=1)
        self.entry_path_labels.grid(row=2, column=1, padx=15)
        self.entry_path_labels.configure(state='normal')
        self.entry_path_labels.insert('insert', 'no file is selected')
        self.entry_path_labels.configure(state='disabled')

        Label(input_frame, text='*Select the labels file corresponding to the data file\nif you want to compute accuracy', anchor='n', padx=15, justify='left').grid(row=3, column=1, sticky='w')

        predict_frame = Frame(self.parent, padx=10, pady=30)
        predict_frame.grid(row=3, column=0, sticky='w')

        self.btn_predict = Button(predict_frame, text="predict", command=self.predict, width=20) 
        self.btn_predict.grid(row=0, column=0, pady=15)
        self.btn_predict.configure(state='disabled')

        self.btn_compute_accu = Button(predict_frame, text="compute accuracy", command=self.predict, width=20) 
        self.btn_compute_accu.grid(row=0, column=1, pady=15, padx=30)
        self.btn_compute_accu.configure(state='disabled')

        return
    
    def select_model(self):
        path = filedialog.askopenfilename(initialdir=self.current_dir,title = "Select file",filetypes = [("file pb", "*.pb")])
        if not path:
            return

        self.load_model_thread = threading.Thread(target=self.load_model, args=(path, 1))
        self.load_model_thread.daemon = True
        self.load_model_thread.start()

        self.current_dir = path.replace(path.split('/')[-1],"")
        # print(self.current_dir)
        self.entry_path_model.configure(state='normal')
        self.entry_path_model.delete('1.0', 'end')
        self.entry_path_model.insert('insert', path)
        self.entry_path_model.configure(state='disabled')
        self.path_model = path
        return
    
    def select_data(self):
        path = filedialog.askopenfilename(initialdir=self.current_dir,title = "Select file",filetypes = [("file npy", "*.npy")])
        if not path:
            return

        self.current_dir = path.replace(path.split('/')[-1],"")
        # print(self.current_dir)
        self.entry_path_data.configure(state='normal')
        self.entry_path_data.delete('1.0', 'end')
        self.entry_path_data.insert('insert', path)
        self.entry_path_data.configure(state='disabled')
        self.path_data = path
        self.btn_predict.configure(state='normal')
        return
    
    def select_labels(self):
        path = filedialog.askopenfilename(initialdir=self.current_dir,title = "Select file",filetypes = [("file npy", "*.npy")])
        if not path:
            return
        self.current_dir = path.replace(path.split('/')[-1],"")
        # print(self.current_dir)
        self.entry_path_labels.configure(state='normal')
        self.entry_path_labels.delete('1.0', 'end')
        self.entry_path_labels.insert('insert', path)
        self.entry_path_labels.configure(state='disabled')
        self.path_labels = path
        self.btn_compute_accu.configure(state='normal')
        return
    
    def load_model(self, path, id):
        self.isLoadingModel = True
        self.btn_select_model.config(state='disabled')
        self.label_load_model.config(text='Loading model...')
        
        graph = load_graph(path)
        output_ = graph.get_tensor_by_name('model/predict:0')
        input_ = graph.get_tensor_by_name('model/input_C:0')
        print(output_.shape[1])
        print(input_.shape[1], input_.shape[2], input_.shape[3])
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=graph)

        self.isLoadingModel = False
        self.btn_select_model.config(state='normal')
        self.label_load_model.config(text='Load model successfully.\nNumber of classes: {}.\nShape input: [{},{},{}]'.format(output_.shape[1],input_.shape[1], input_.shape[2], input_.shape[3]))
        return
    
    
    
    def predict(self):
        print('let predict')
        if not self.sess:
            print('>>> chua co model')
        
        # check size data, check sess, =>
        return
    
    def compute_accu(self):
        print('let compute_accu')
        return

