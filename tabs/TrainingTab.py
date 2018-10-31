from tkinter import filedialog, Frame, Button, Entry, Text, Label, Scrollbar, messagebox
from tkinter.ttk import Style
import os
from common import *
import time
import threading
import numpy as np

from sa import SA    
from component import label_with_entry, label_with_link_path_file


class TrainingTab(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.current_dir = os.getcwd()
        self.parent = parent
        self.initUI()
        self.text = 'clgt'
        self.path_train_set = ''
        self.path_val_data = ''
        self.path_val_labels = ''
        self.path_model = ''

    def initUI(self):
        # self.parent.title("DEMO")
        self.style = Style()
        self.style.theme_use("default")

        paddingTop = Frame(self.parent, height=30, padx=10, pady=30)
        paddingTop.grid(row=0)
        
        select_model_frame = Frame(self.parent, padx=10)
        select_model_frame.grid(row=1, column=0, sticky='w')

        self.fileDialogButton = Button(select_model_frame, text="Select your model", command=self.select_model, width=20) 
        self.fileDialogButton.grid(row=1, column=0)


        self.entry_path = Text(select_model_frame, width=80, height=1)
        self.entry_path.grid(row=1, column=1, padx=15)
        self.entry_path.configure(state='normal')
        self.entry_path.insert('insert', 'no file is selected')
        self.entry_path.configure(state='disabled')

        init_frame = Frame(self.parent, padx=10, pady=10)
        init_frame.grid(row=2, column=0, sticky='w')

        self.e_height = label_with_entry(init_frame, 'height: ', 0, 0, 10)
        self.e_width = label_with_entry(init_frame, 'width: ', 1, 0, 10)
        self.e_channel = label_with_entry(init_frame, 'channel: ', 2, 0, 10)
        self.e_no_classes = label_with_entry(init_frame, 'number of classes: ', 3, 0, 10)
        self.e_lr = label_with_entry(init_frame, 'learning rate: ', 4, 0, 10)
        self.e_no_epochs = label_with_entry(init_frame, 'number of epochs: ', 5, 0, 10)

        self.text_path_train, self.btn_dialog_train_set = label_with_link_path_file(init_frame, 'train-set: ', 0, 2, self.select_train_set)
        self.text_path_val_data, self.btn_dialog_val_data = label_with_link_path_file(init_frame, 'val-data-set: ', 1, 2, self.select_val_data)
        self.text_path_val_labels, self.btn_dialog_val_labels = label_with_link_path_file(init_frame, 'val-labels-set: ', 2, 2, self.select_val_labels)

        init_frame = Frame(self.parent, padx=10, pady=10)
        init_frame.grid(row=2, column=3, columnspan=2, sticky='w')

        log_frame = Frame(self.parent, padx=10, pady=10)
        log_frame.grid(row=3, column=0, columnspan=5, sticky='w')

        self.log = Text(log_frame, width=130, height=40)
        self.log.grid(row=0, column=0, padx=5, pady=10)
        self.log.configure(state='normal')
        self.log.insert('insert', 'Some Text')
        self.log.configure(state='disabled')
        self.scrollb = Scrollbar(log_frame, command=self.log.yview)
        self.scrollb.grid(row=0, column=1, sticky='nsew')
        self.log['yscrollcommand'] = self.scrollb.set

        Bot_frame = Frame(self.parent, padx=10, pady=10)
        Bot_frame.grid(row=4, column=0, columnspan=5, sticky='w')
        self.trainBtn = Button(Bot_frame, text="Train model", width=20, command=self.train_model)
        self.trainBtn.grid(row=0, column=0)
        self.stopBtn = Button(Bot_frame, text="Stop training and export model")
        self.stopBtn.grid(row=0, column=1, padx=20)
        self.stopBtn.configure(state='disabled', command=self.stop_training_and_export)


    def select_train_set(self):
        path = filedialog.askopenfilename(initialdir=self.current_dir,title = "Select file",filetypes = [("file npy", "*.npy")])
        if not path:
            return
        self.current_dir = path.replace(path.split('/')[-1],"")
        self.text_path_train.configure(state='normal')
        self.text_path_train.delete('1.0', 'end')
        self.text_path_train.insert('insert', path)
        self.text_path_train.configure(state='disabled')
        self.path_train_set = path
        return
    
    def select_val_data(self):
        path = filedialog.askopenfilename(initialdir=self.current_dir,title = "Select file",filetypes = [("file npy", "*.npy")])
        if not path:
            return
        self.current_dir = path.replace(path.split('/')[-1],"")
        self.text_path_val_data.configure(state='normal')
        self.text_path_val_data.delete('1.0', 'end')
        self.text_path_val_data.insert('insert', path)
        self.text_path_val_data.configure(state='disabled')
        self.path_val_data = path
        return
    
    def select_val_labels(self):
        path = filedialog.askopenfilename(initialdir=self.current_dir ,title = "Select file",filetypes = [("file npy", "*.npy")])
        if not path:
            return
        self.current_dir = path.replace(path.split('/')[-1],"")
        self.text_path_val_labels.configure(state='normal')
        self.text_path_val_labels.delete('1.0', 'end')
        self.text_path_val_labels.insert('insert', path)
        self.text_path_val_labels.configure(state='disabled')
        self.path_val_labels = path
        return
    
    def select_model(self):
        path = filedialog.askopenfilename(initialdir=self.current_dir ,title = "Select file",filetypes = [("file text", "*.txt")])
        if not path:
            return
        self.current_dir = path.replace(path.split('/')[-1],"")
        self.entry_path.configure(state='normal')
        self.entry_path.delete('1.0', 'end')
        self.entry_path.insert('insert', path)
        self.entry_path.configure(state='disabled')
        self.path_model = path
        return
        
    def train_model(self):
        w = self.e_width.get().replace(" ", "")
        h = self.e_height.get().replace(" ", "")
        channels = self.e_channel.get().replace(" ", "")
        no_classes = self.e_no_classes.get().replace(" ", "")
        lr = self.e_lr.get().replace(" ", "")
        epochs = self.e_no_epochs.get().replace(" ", "")
        # check param
        if isEmpty(self.path_model.replace('no file is selected','')):
            messagebox.showinfo("Error", "You must provide your model!")
            return
        if isEmpty(w) or isEmpty(h) or isEmpty(channels) or isEmpty(no_classes) or isEmpty(lr) or isEmpty(epochs):
            messagebox.showinfo("Error", "You must provide all elements!")
            return
        if isEmpty(self.path_train_set) or isEmpty(self.path_val_data) or isEmpty(self.path_val_labels):
            messagebox.showinfo("Error", "You must provide paths of dataset!")
            return
        # check param
        self.training_thread = threading.Thread(target=self.thread_train_model, args=(h,w,channels,no_classes,lr,epochs, self.path_model, self.path_train_set, self.path_val_data, self.path_val_labels))
        self.training_thread.daemon = True
        self.training_thread.start()
        self.disable_all_except_stopBtn()
        return
    
    def disable_all_except_stopBtn(self):
        self.e_width.configure(state='disabled')
        self.e_height.configure(state='disabled')
        self.e_channel.configure(state='disabled')
        self.e_no_classes.configure(state='disabled')
        self.e_lr.configure(state='disabled')
        self.e_no_epochs.configure(state='disabled')
        self.fileDialogButton.configure(state='disabled')
        self.btn_dialog_train_set.configure(state='disabled')
        self.btn_dialog_val_data.configure(state='disabled')
        self.btn_dialog_val_labels.configure(state='disabled')
        self.trainBtn.config(state="disabled")
        self.stopBtn.configure(state='normal')

    def enable_all_except_stopBtn(self):
        self.e_width.configure(state='normal')
        self.e_height.configure(state='normal')
        self.e_channel.configure(state='normal')
        self.e_no_classes.configure(state='normal')
        self.e_lr.configure(state='normal')
        self.e_no_epochs.configure(state='normal')
        self.fileDialogButton.configure(state='normal')
        self.btn_dialog_train_set.configure(state='normal')
        self.btn_dialog_val_data.configure(state='normal')
        self.btn_dialog_val_labels.configure(state='normal')
        self.trainBtn.config(state="normal")
        self.stopBtn.configure(state='disabled')
    
    def stop_training_and_export(self):
        self.model.stop_training = True
        self.enable_all_except_stopBtn()

    def thread_train_model(self,h,w,channels,no_classes,lr,epochs, path_model, path_train_set, path_val_data, path_val_labels):
        print(h,w,channels,no_classes,lr,epochs, path_model, path_train_set, path_val_data, path_val_labels)
        h = int(h)
        w = int(w)
        channels = int(channels)
        no_classes = int(no_classes)
        epochs = int(epochs)
        lr = float(lr)
        with open(path_model, 'rb') as f:
            classify_model = f.read()
        train_data = np.load(path_train_set)
        val_data = np.load(path_val_data)
        val_labels = np.load(path_val_labels)
        data = np.zeros((2,100,48,48,1))
        val = np.ones((20,48,48,1))
        labels_val = one_hot(np.ones((20)), 2)

        try:
            self.model = SA(shape=[h,w,channels],no_classes=no_classes, classify_model=classify_model, log_text=self.update_log, learning_rate=lr)
            self.model.build_model()
            self.model.train(data, val, labels_val, epochs=epochs)
        except ValueError:
            self.update_log("Error: {}".format(ValueError))
            self.update_log("\nPleas check your parameters and your data!")
            return

        
        self.model.export_pb()
        return

    def update_log(self, text):
        # time.sleep(1)
        self.log.configure(state='normal')
        self.log.insert('insert', text)
        self.log.see('end')
        self.log.configure(state='disabled')
        # self.log.update_idletasks()
