from tkinter import *
from tkinter import ttk
from tabs.TrainingTab import TrainingTab
from tabs.PredictTab import PredictTab

window = Tk()
window.title("Demo")
window.geometry("1000x800+300+300")
tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)

TrainingTab(tab1)
PredictTab(tab2)

tab_control.add(tab1, text='Training')
tab_control.add(tab2, text='Predict')


# lbl2 = Label(tab2, text= 'label2')
# lbl2.grid(column=0, row=0)

# lbl2 = Label(tab2, text= 'label2')
# lbl2.grid(column=0, row=0)

tab_control.pack(expand=1, fill='both')
window.mainloop()