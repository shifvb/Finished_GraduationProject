import time
import tkinter as tk


class Window:
    def __init__(self, title='nms', width=300, height=120, staFunc=bool, stoFunc=bool):
        self.w = width
        self.h = height
        self.stat = True
        self.staFunc = staFunc
        self.stoFunc = stoFunc
        self.staIco = None
        self.stoIco = None

        self.root = tk.Tk(className=title)

    def center(self):
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = int((ws / 2) - (self.w / 2))
        y = int((hs / 2) - (self.h / 2))
        self.root.geometry('{}x{}+{}+{}'.format(self.w, self.h, x, y))

    def packBtn(self):
        self.btnSer = tk.Button(self.root, command=self.event, )
        self.btnSer.pack(side=tk.LEFT)
        btnQuit = tk.Button(self.root, text='关闭窗口', command=self.root.quit)
        btnQuit.pack(side=tk.RIGHT)

    def event(self):
        self.btnSer['state'] = 'disabled'
        if self.stat:
            if self.stoFunc():
                self.btnSer['text'] = '启动服务'
                self.stat = False
                self.root.iconbitmap(self.stoIco)
        else:
            if self.staFunc():
                self.btnSer['text'] = '停止服务'
                self.stat = True
                self.root.iconbitmap(self.staIco)
        self.btnSer['state'] = 'active'

    def loop(self):
        self.root.resizable(False, False)  # 禁止修改窗口大小
        self.packBtn()
        self.center()  # 窗口居中
        self.event()
        self.root.mainloop()


########################################################################
def sta():
    print('start.')
    return True


def sto():
    print('stop.')
    return True


if __name__ == '__main__':
    import sys, os

    w = Window(staFunc=sta, stoFunc=sto)
    w.staIco = "idle.ico"
    w.stoIco = "pyc.ico"
    w.loop()
