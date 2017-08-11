import tkinter as tk


class TheProject(object):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Medical image feature detection system v1.0")
        # frames
        self.frame_1 = tk.LabelFrame(self.root, text="process_1")
        self.frame_1.grid(row=0, column=0, padx=5, pady=5)
        self.frame_2 = tk.LabelFrame(self.root, text="process_2")
        self.frame_2.grid(row=0, column=1, padx=5, pady=5)
        self.frame_3 = tk.LabelFrame(self.root, text="process_3")
        self.frame_3.grid(row=1, column=0, padx=5, pady=5)
        self.frame_4 = tk.LabelFrame(self.root, text="process_4")
        self.frame_4.grid(row=1, column=1, padx=5, pady=5)
        # frame_1
        self.frame_11 = tk.Frame(self.frame_1, width=300, height=100)
        self.frame_11.grid(row=0, column=0, rowspan=2, columnspan=3)
        self.label_f1_1 = tk.Label(self.frame_1, text="label_1")
        self.label_f1_1.grid(row=0, column=0)
        self.btn_f1_1 = tk.Button(self.frame_1, text="button_1")
        self.btn_f1_1.grid(row=0, column=2)
        self.label_f1_2 = tk.Label(self.frame_1, text="label_2")
        self.label_f1_2.grid(row=1, column=0)
        self.btn_f1_2 = tk.Button(self.frame_1, text="button_2")
        self.btn_f1_2.grid(row=1, column=2)
        # frame_2
        self.frame_21 = tk.Frame(self.frame_2, width=700, height=100)
        self.frame_21.grid(row=0, column=0, rowspan=2, columnspan=5)
        self.label_f2_1 = tk.Label(self.frame_2, text="label_2_1")
        self.label_f2_1.grid(row=0, column=0)
        self.var_entry_f2_1 = tk.StringVar()
        self.entry_f2_1 = tk.Entry(self.frame_2, width=60, textvariable=self.var_entry_f2_1)
        self.entry_f2_1.grid(row=0, column=1, columnspan=4)
        self.label_f2_2 = tk.Label(self.frame_2, text="label_2_2")
        self.label_f2_2.grid(row=1, column=0)
        self.var_entry_f2_2 = tk.StringVar()
        self.entry_f2_2 = tk.Entry(self.frame_2, width=60, textvariable=self.var_entry_f2_2)
        self.entry_f2_2.grid(row=1, column=1, columnspan=4)
        # frame_3
        self.frame_31 = tk.Frame(self.frame_3, width=300, height=600)
        self.frame_31.grid(row=0, column=0)
        self.var_text_f3_1 = tk.StringVar()
        self.text_f3_1 = tk.Text(self.frame_3, width=40, height=45)
        self.text_f3_1.grid(row=0, column=0)
        # frame_4
        self.frame_41 = tk.Frame(self.frame_4, width=700, height=600)
        self.frame_41.grid(row=0, column=0, rowspan=6, columnspan=8)

        self.var_radiobtn_1 = tk.IntVar(value=1)
        self.radiobtn_f4_1 = tk.Radiobutton(self.frame_4, text="radiobtn_type_1", variable=self.var_radiobtn_1, value=1)
        self.radiobtn_f4_1.grid(row=0, column=1)
        self.radiobtn_f4_2 = tk.Radiobutton(self.frame_4, text="radiobtn_type_2", variable=self.var_radiobtn_1, value=2)
        self.radiobtn_f4_2.grid(row=0, column=2)
        self.radiobtn_f4_3 = tk.Radiobutton(self.frame_4, text="radiobtn_type_3", variable=self.var_radiobtn_1, value=3)
        self.radiobtn_f4_3.grid(row=0, column=3)

        tk.Label(self.frame_4, text="current type_1 num:").grid(row=1, column=1)
        self.var_label_f4_2 = tk.IntVar(value=234)
        self.label_f4_2 = tk.Label(self.frame_4, textvariable=self.var_label_f4_2).grid(row=1, column=2, sticky=tk.W)
        tk.Label(self.frame_4, text="current type_2 num:").grid(row=2, column=1)

    def mainloop(self):
        self.root.mainloop()


project = TheProject()
project.mainloop()
