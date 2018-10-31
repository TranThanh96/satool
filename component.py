from tkinter import Frame, Entry, Label, Button, filedialog, Text

def label_with_entry(parent, label_name, row, col, width_text_entry):
    Label(parent, text=label_name).grid(row=row, column=col, sticky='w')
    vcmd = (parent.register(validate),
                '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
    text_entry = Entry(parent, validate = 'key',  validatecommand = vcmd)

    text_entry.config(width=10)
    text_entry.grid(row=row, column=col+1)
    return text_entry

    
        
def validate(action, index, value_if_allowed,
                       prior_value, text, validation_type, trigger_type, widget_name):
        if text in '0123456789.':
            try:
                float(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False

def label_with_link_path_file(parent, label_name, row, col, openFileDialog, path_default='/', padx=(30,0)):
    Label(parent, text=label_name).grid(row=row, column=col, sticky='w', padx=padx)

    entry_path = Text(parent, width=80, height=1)
    entry_path.grid(row=row, column=col+1)
    entry_path.configure(state='normal')
    entry_path.insert('insert', 'no file is selected')
    entry_path.configure(state='disabled')
    btn = Button(parent, text=" ", command=openFileDialog)
    btn.grid(row=row, column=col+2)
    return entry_path, btn
