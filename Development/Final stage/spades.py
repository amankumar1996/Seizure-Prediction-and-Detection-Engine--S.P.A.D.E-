from tkinter import *
from random import *
from spade import *
import os
path=""
source=""


import tkinter.messagebox
def team1():
    tan=Tk()

    ab=Label(tan,text="Aman Kumar\nKeshav Garg\nKunal Kishore\nSophie Kakker",font="Helvetica 18")
    ab.grid(row=1,column=0)
    khali1=Label(tan,text="         ",font="Helvetica 24")
    khali2=Label(tan,text="         ",font="Helvetica 24")
    khali3=Label(tan,text="         ",font="Helvetica 24")
    khali4=Label(tan,text="         ",font="Helvetica 18")
    khali5=Label(tan,text="         ",font="Helvetica 18")
    khali6=Label(tan,text="         ",font="Helvetica 18")
    khali1.grid(row=0,column=1)
    khali2.grid(row=0,column=3)
    khali3.grid(row=0,column=5)
    khali4.grid(row=1,column=1)
    khali5.grid(row=1,column=3)
    khali6.grid(row=1,column=5)

    roll=Label(tan,text="401503006\n401503013\n401503014\n401503029",font="Helvetica 18")
    roll.grid(row=1,column=2)
    phone=Label(tan,text="9465001010\n9811967439\n8195905598\n8168617743",font="Helvetica 18")
    phone.grid(row=1,column=4)
    email=Label(tan,text="amankumar0343@gmail.com\nkeshavgarg139@gmail.com\nkunalkk477@gmail.com\nsophie.kakker@gmail.com",font="Helvetica 18")
    email.grid(row=1,column=6)
    tan.mainloop()
def help1():
    window3=Tk()
    window3.wm_title("help")

    label1=Label(window3,text="PATH:- Path where you want your epilepsy \n     files and details to be saved!!",font="helvetica 18")
    label1.grid(row=0,column=0,rowspan=1)

    label2=Label(window3,text="SOURCE:- Path from where the datasets are!!",font="helvetica 18")
    label2.grid(row=2,column=0)

    window3.mainloop()
def help2():
    Help2=Tk()
    label1=Label(Help2,text="Please enter the name of the files for intericatl info",font="helvetica 18")
    label1.grid(row=0,column=0,rowspan=1)

    label2=Label(Help2,text="Extension for the files must be in .edf form",font="helvetica 18")
    label2.grid(row=2,column=0)
    Help2.mainloop()
def help3():
    Help3=Tk()
    label1=Label(Help3,text="For the first Entry- Please enter the number of channels that your device have",font="helvetica 18")
    label1.grid(row=1,column=0)


    label2=Label(Help3,text="It must be a number",font="helvetica 12")
    label2.grid(row=3,column=0)

    label3=Label(Help3,text=" For the second Entry- Please enter the number of the channels that have to be ignored",font="helvetica 18")
    label3.grid(row=4,column=0,rowspan=1)

    label4=Label(Help3,text="It should start from 1 and must be seperated by commas in b/w.",font="helvetica 12")
    label4.grid(row=6,column=0)

    label5=Label(Help3,text="For the third Entry- Enter the sampling frequency of the headset",font="helvetica 18")
    label5.grid(row=8,column=0,rowspan=1)

    label6=Label(Help3,text="It should be a power of 2",font="helvetica 12")
    label6.grid(row=10,column=0)
    Help3.mainloop()
def help4():
    Help3=Tk()
    label1=Label(Help3,text="For the first Entry- Please enter the name of the file for previous encountered seizures. ",font="helvetica 18")
    label1.grid(row=1,column=0)


    label2=Label(Help3,text="It must be a in .edf format",font="helvetica 12")
    label2.grid(row=3,column=0)

    label3=Label(Help3,text=" For the second Entry- Please enter the name of the current seizure file name",font="helvetica 18")
    label3.grid(row=4,column=0,rowspan=1)

    label4=Label(Help3,text="It must be in .edf format.",font="helvetica 12")
    label4.grid(row=6,column=0)

    label5=Label(Help3,text="For the third Entry- Enter the starting time of the seizure in the seizure file",font="helvetica 18")
    label5.grid(row=8,column=0,rowspan=1)

    label6=Label(Help3,text="it cannot be negative",font="helvetica 12")
    label6.grid(row=10,column=0)

    label5=Label(Help3,text="For the fourth Entry- How many minutes of preictal seizure data that we can extract",font="helvetica 18")
    label5.grid(row=12,column=0,rowspan=1)

    label6=Label(Help3,text="it cannot be negative, must be a multiple of 5 and range b/w 10-60",font="helvetica 12")
    label6.grid(row=14,column=0)

    Help3.mainloop()
def eightwindow():
    window6=Tk()
    window6.wm_title("close app")
    def Home():
        window6.destroy()
        mainwindow()
    def Back():
        window6.destroy()
        sixthhwindow()
    def destroy1():
        window6.destroy()
    label1=Label(window6,text="Your Model is training!! \n Please don't shut down your system\n Please close this application",font="helvetica 18 bold",justify="center")
    label1.grid(row=0,column=0,rowspan=1,columnspan=5,padx=10,pady=10,ipadx=5,ipady=5)

    gif1=PhotoImage(file="shutdown.png")
    label2=Label(window6,image=gif1,justify="center")
    label2.grid(row=2,column=2,padx=5,pady=5,ipadx=5,ipady=5)

    image3=PhotoImage(file="home1.png")
    button3=Button(window6,image=image3,command=Home)
    button3.grid(row=4,column=2)

    image4= PhotoImage(file="back.png")
    button4=Button(window6,image=image4,command=Back)
    button4.grid(row=4,column=0)

    image2=PhotoImage(file="close.png")
    button3=Button(window6,image=image2,command=destroy1)
    button3.grid(row=4,column=4)
    window6.mainloop()
def sixthwindow():
    window6=Tk()
    window6.wm_title("start and end time")
    def create():
        global starttime
        global endtime
        a=Entry1.get()
        b=Entry2.get()
        a=int(a)
        b=int(b)
        if a<5 or a>55 or a%5!=0:
            tkinter.messagebox.showinfo("wrong value","check the time")
            Entry1.delete(0,END)
        elif a>b or b%5!=0 or b>60:
            tkinter.messagebox.showinfo("wrong value","check the time")
            Entry2.delete(0,END)
        else:
            Entry1.delete(0,END)
            Entry2.delete(0,END)
            starttime=a
            endtime=b
            window6.destroy()
            eightwindow()
    def testVal(inStr,i,acttyp):
        ind=int(i)
        if acttyp == '1': #insert
            if not inStr[ind].isdigit():
                return False
        return True
    def Home():
        window6.destroy()
        mainwindow()
    def Back():
        window6.destroy()
        fifthwindow()

    label1=Label(window6,text="Preprocessing the seizure data enter the range of the time!!",font="helvetica 18 bold")
    label1.grid(row=0,column=0,rowspan=1,columnspan=4,padx=10,pady=10,ipadx=5,ipady=5)
    start=StringVar()
    end=StringVar()
    Label2=Label(window6,text="Start Time",font="helvetica 12 bold")
    Label2.grid(row=2,column=0,padx=2,pady=2,ipadx=2,ipady=2)

    Label3=Label(window6,text="End Time",font="helvetica 12 bold")
    Label3.grid(row=3,column=0,padx=2,pady=2,ipadx=2,ipady=2)

    Entry1=Entry(window6,textvariable=start,validate="key",width=50,selectbackground="red")
    Entry1['validatecommand'] = (Entry1.register(testVal),'%P','%i','%d')
    Entry1.grid(row=2,column=2,ipadx=2,ipady=2,padx=2,pady=2)

    Entry2=Entry(window6,textvariable=end,validate="key",width=50,selectbackground="red")
    Entry2['validatecommand'] = (Entry2.register(testVal),'%P','%i','%d')
    Entry2.grid(row=3,column=2,ipadx=2,ipady=2,padx=2,pady=2)

    image6=PhotoImage(file="create.png")
    button1=Button(window6,image=image6,command=create)
    button1.grid(row=4,column=4,padx=2,pady=2,ipadx=2,ipady=2)
    # gif1=PhotoImage(file="shutdown.png")
    # label2=Label(window6,image=gif1,justify="center")
    # label2.grid(row=2,column=2,padx=5,pady=5,ipadx=5,ipady=5)

    image3=PhotoImage(file="home1.png")
    button3=Button(window6,image=image3,command=Home)
    button3.grid(row=4,column=2)

    image4= PhotoImage(file="back.png")
    button4=Button(window6,image=image4,command=Back)
    button4.grid(row=4,column=0)

    # image2=PhotoImage(file="next.png")
    # button3=Button(window6,image=image2,command=destroy1)
    # button3.grid(row=5,column=4)
    window6.mainloop()
def fifthwindow():
    window5=Tk()
    window5.wm_title("intericatl info")
    def Next():
        window5.destroy()
        sixthwindow()

    def ADD():
        a= Entry1.get()
        if(a[-4:]!=".edf"):
            tkinter.messagebox.showinfo("wrong extension","check the file extension for file name.")
            Entry1.delete(0,END)
        elif(len(a)<5):
            tkinter.messagebox.showinfo("no complete file name","check file name it is not complete.")
            Entry1.delete(0,END)
        else:
            f=open(path+"\\SPADE\\info_files\\interictal_info.txt","a+")
            f.write(a+"\n")
            f.close()
            Entry1.delete(0,END)

    def Home():
        window5.destroy()
        mainwindow()
    # def Back():
    #     window5.destroy()
    #     fourthwindow()
    def Help():
        help2()
    filename=StringVar()

    label1=Label(window5,text="Enter the name of the files",font="Helvetica 24",justify="center",pady=4)
    label1.grid(row=0,column=0,columnspan=4,rowspan=2)

    label2=Label(window5,text="file name",font="Helvetica 16 bold")
    label2.grid(row=2,column=0,padx=2,pady=2,ipadx=2,ipady=2)

    Entry1=Entry(window5,textvariable=filename,width=50,selectbackground="red")
    Entry1.grid(row=2,column=2,padx=2,pady=2,ipadx=2,ipady=2)

    image1=PhotoImage(file="next.png")
    image2=PhotoImage(file="ADD.png")
    button1=Button(window5,image=image2,command=ADD)
    button1.grid(row=4,column=4)

    button2=Button(window5,image=image1,command=Next)
    button2.grid(row=6,column=4)

    image3=PhotoImage(file="home1.png")
    button3=Button(window5,image=image3,command=Home)
    button3.grid(row=6,column=2)

    # image4= PhotoImage(file="back.png")
    # button4=Button(window5,image=image4,command=Back)
    # button4.grid(row=6,column=0)
    image5=PhotoImage(file="help.png")
    button5=Button(window5,image=image5,command=Help)
    button5.grid(row=2,column=4,padx=4,pady=4,ipadx=3,ipady=3)
    window5.mainloop()
def fourthwindow():
    window4=Tk()
    window4.wm_title("preictal info")
    def Help41():
        help4()
    def Next():
        window4.destroy()
        fifthwindow()
    def testVal(inStr,i,acttyp):
        ind=int(i)
        if acttyp == '1': #insert
            if not inStr[ind].isdigit():
                return False
        return True
    def ADD():
        a=Entry1.get()
        b=Entry2.get()
        c=Entry3.get()
        d=Entry4.get()
        c=int(c)
        d=int(d)
        if (a[0]!="-"):
            if(a[-4:]!=".edf" or len(a)<5):
                tkinter.messagebox.showinfo("wrong extension","check the file extension for prev seizure file name.")
                Entry1.delete(0,END)


        elif(b[-4:]!=".edf" or len(b)<5):
            tkinter.messagebox.showinfo("wrong extension","check the file extension for the seizure file name.")
            Entry2.delete(0,END)
        elif(c<0):
            tkinter.messagebox.showinfo("wrong input","negative value for time start.")
            Entry3.delete(0,END)
        elif(d<10 or d>60 or d%5!=0):
            tkinter.messagebox.showinfo("wrong input","extraction is not a multiple of 5 or it is out of range")
            Entry4.delete(0,END)
        else:
            c=str(c)
            d=str(d)
            f=open(path+"\\SPADE\\info_files\\preictal_info.txt","a+")
            f.write(a+"\t"+b+"\t"+c+"\t"+d+"\n")
            f.close()
            Entry1.delete(0,END)
            Entry2.delete(0,END)
            Entry3.delete(0,END)
            Entry4.delete(0,END)

    def Home():
        window4.destroy()
        mainwindow()
    def Back():
        window4.destroy()
        thirdwindow()


    label1=Label(window4,text="Enter the preictal file info",font="Helvetica 24",justify="center",pady=4)
    label1.grid(row=0,column=0,columnspan=4,rowspan=2)

    prefilename=StringVar()
    seizfilename= StringVar()
    timeofstart=StringVar()
    minutestoextract=StringVar()

    label2=Label(window4,text="Prev seizure file",font="Helvetica 16 bold")
    label2.grid(row=2,column=0,padx=2,pady=2,ipadx=2,ipady=2)

    label3=Label(window4,text="Seizure file",font="Helvetica 16 bold")
    label3.grid(row=3,column=0,padx=2,pady=2,ipadx=2,ipady=2)

    label4=Label(window4,text="time of start",font="Helvetica 16 bold")
    label4.grid(row=4,column=0,padx=2,pady=2,ipadx=2,ipady=2)

    label5=Label(window4,text="minutes to extract",font="Helvetica 16 bold")
    label5.grid(row=5,column=0,padx=2,pady=2,ipadx=2,ipady=2)

    Entry1=Entry(window4,textvariable=prefilename,width=50,selectbackground="red")
    Entry1.grid(row=2,column=2,padx=2,pady=2,ipadx=2,ipady=2)

    Entry2=Entry(window4,textvariable=seizfilename,width=50,selectbackground="red")
    Entry2.grid(row=3,column=2,padx=2,pady=2,ipadx=2,ipady=2)

    Entry3=Entry(window4,textvariable=timeofstart,validate="key",width=50,selectbackground="red")
    Entry3['validatecommand'] = (Entry3.register(testVal),'%P','%i','%d')
    Entry3.grid(row=4,column=2,padx=2,pady=2,ipadx=2,ipady=2)

    Entry4=Entry(window4,textvariable=minutestoextract,validate="key",width=50,selectbackground="red")
    Entry4['validatecommand'] = (Entry4.register(testVal),'%P','%i','%d')
    Entry4.grid(row=5,column=2,padx=2,pady=2,ipadx=2,ipady=2)

    image1=PhotoImage(file="next.png")
    image2=PhotoImage(file="ADD.png")
    button1=Button(window4,image=image2,command=ADD)
    button1.grid(row=6,column=4)

    button2=Button(window4,image=image1,command=Next)
    button2.grid(row=8,column=4)
    image3=PhotoImage(file="home1.png")
    button3=Button(window4,image=image3,command=Home)
    button3.grid(row=8,column=2)

    image4= PhotoImage(file="back.png")
    button4=Button(window4,image=image4,command=Back)
    button4.grid(row=8,column=0)
    image5=PhotoImage(file="help.png")
    button5=Button(window4,image=image5,command=Help41)
    button5.grid(row=6,column=0,padx=4,pady=4,ipadx=3,ipady=3)
    window4.mainloop()
def thirdwindow():
    window3=Tk()
    create_dir(path)
    window3.wm_title("system_info")



    def testVal(inStr,i,acttyp):
        ind=int(i)
        if acttyp == '1': #insert
            if not inStr[ind].isdigit():
                return False
        return True
    def Home():
        window3.destroy()
        mainwindow()
    def Back():
        window3.destroy()
        nextwindow()
    def Help31():
        help3()
    def Next():
        a=Entry1.get()
        b=Entry2.get()
        c=Entry3.get()
        if(len(a)==0):
            tkinter.messagebox.showinfo("no input","Pleased enter the no of channels")
            Entry1.delete(0,END)

        elif(len(c)==0):
            tkinter.messagebox.showinfo("no input","Pleased enter the frequency")
            Entry3.delete(0,END)
        elif(len(b)==0):
            s={0}
            for i in range(int(a)):
                s.add(i)
            print(s)
            for i in s:
                s=str(s)
            f=open(path+"\\SPADE\\info_files\\system_info.txt","w+")
            f.write("number_of_channels\t")
            f.write(a)
            f.write("\n")
            f.write("channel_names\t")
            l=len(s)-1
            f.write(s[1:l])
            f.write("\n")
            f.write("sampling_frequency\t")
            f.write(c)
            f.write("\n")
            f.close()
            f=open(path+"\\SPADE\\info_files\\preictal_info.txt","w")
            f.write("prev_seizure_file_name\tseizure_file_name\ttime_of_start\tminutes_to_extract(multiple_of_5)\n")
            f.close()

            f=open(path+"\\SPADE\\info_files\\interictal_info.txt","w")
            f.write("File Names \n")
            f.close()
            window3.destroy()
            fourthwindow()
        else:
            b=b.split(",")

            for i in range(len(b)):
                b[i]=int(b[i])
                b[i]=b[i]-1
            b.sort()
            b=set(b)
            s={0}
            for i in range(int(a)):
                s.add(i)
            s=s.difference(b)
            print(s)
            for i in s:
                s=str(s)
            f=open(path+"\\SPADE\\info_files\\system_info.txt","w+")
            f.write("number_of_channels\t")
            f.write(a)
            f.write("\n")
            f.write("channel_names\t")
            l=len(s)-1
            f.write(s[1:l])
            f.write("\n")
            f.write("sampling_frequency\t")
            f.write(c)
            f.write("\n")
            f.close()
            f=open(path+"\\SPADE\\info_files\\preictal_info.txt","w")
            f.write("prev_seizure_file_name\tseizure_file_name\ttime_of_start\tminutes_to_extract(multiple_of_5)\n")
            f.close()

            f=open(path+"\\SPADE\\info_files\\interictal_info.txt","w")
            f.write("File Names \n")
            f.close()
            window3.destroy()
            fourthwindow()

    label1=Label(window3,text="Please provide the systems info!",font="Helvetica 24",justify="center",pady=4)
    label1.grid(row=0,column=0,columnspan=4,rowspan=2)

    numberchannel=StringVar()
    channelname= StringVar()
    samplingfrequency=StringVar()

    label2=Label(window3,text="No of channels",font="Helvetica 16 bold")
    label2.grid(row=2,column=0,padx=2,pady=2,ipadx=2,ipady=2)

    label3=Label(window3,text="Channels to be ignored",font="Helvetica 16 bold")
    label3.grid(row=3,column=0,padx=2,pady=2,ipadx=2,ipady=2)

    label4=Label(window3,text="Channel Frequency",font="Helvetica 16 bold")
    label4.grid(row=4,column=0,padx=2,pady=2,ipadx=2,ipady=2)

    Entry1=Entry(window3,textvariable=numberchannel,validate="key",width=50,selectbackground="red")
    Entry1['validatecommand'] = (Entry1.register(testVal),'%P','%i','%d')
    Entry1.grid(row=2,column=2,padx=2,pady=2,ipadx=2,ipady=2)

    Entry2=Entry(window3,textvariable=channelname,width=50,selectbackground="red")
    Entry2.grid(row=3,column=2,padx=2,pady=2,ipadx=2,ipady=2)

    Entry3=Entry(window3,textvariable=samplingfrequency,validate="key",width=50,selectbackground="red")
    Entry3['validatecommand'] = (Entry3.register(testVal),'%P','%i','%d')
    Entry3.grid(row=4,column=2,padx=2,pady=2,ipadx=2,ipady=2)

    image1=PhotoImage(file="next.png")
    button1=Button(window3,image=image1,command=Next)
    button1.grid(row=6,column=4)

    image3=PhotoImage(file="home1.png")
    button3=Button(window3,image=image3,command=Home)
    button3.grid(row=6,column=2)

    image4= PhotoImage(file="back.png")
    button4=Button(window3,image=image4,command=Back)
    button4.grid(row=6,column=0)
    image5=PhotoImage(file="help.png")
    button5=Button(window3,image=image5,command=Help31)
    button5.grid(row=5,column=0,padx=4,pady=4,ipadx=3,ipady=3)


    window3.mainloop()
def nextwindow():

    window2= Tk()
    window2.wm_title("enter directory")

    def Next():
        global path
        global source
        path=Entry1.get()
        source=Entry2.get()
        # create_dir(path)
        if(len(path)==0):
            tkinter.messagebox.showinfo("No input","Enter the path")
            Entry1.delete(0,END)
        elif(len(source)==0):
            tkinter.messagebox.showinfo("No input","Enter the source")
            Entry2.delete(0,END)
        else:
            print(path)
            print(source)
            window2.destroy()
            thirdwindow()

    def Help():
        help1()
    def Home():
        window2.destroy()
        mainwindow()
    label1=Label(window2,text="Enter the path for the files.",font="Helvetica 24",justify="center",pady=4)
    label1.grid(row=0,column=0,columnspan=4,rowspan=2)

    label2=Label(window2,text="Path",font="Helvetica 16 bold")
    label2.grid(row=2,column=0,columnspan=2,padx=2,pady=2,ipadx=2,ipady=2)

    label3=Label(window2,text="source",font="helvetica 16 bold")
    label3.grid(row=3,column=0,columnspan=2,padx=2,pady=2,ipadx=2,ipady=2)


    Path=StringVar()
    cwd=os.getcwd()
    Path2=StringVar()


    Entry1=Entry(window2,textvariable=Path,width=75,selectbackground="red")
    Entry1.grid(row=2,column=3,columnspan=2,padx=2,pady=2,ipadx=2,ipady=2)

    Entry2=Entry(window2,textvariable=Path2,width=75,selectbackground="red")
    Entry2.grid(row=3,column=3,columnspan=2,padx=2,pady=2,ipadx=2,ipady=2)


    image1=PhotoImage(file="next.png")
    button1=Button(window2,image=image1,command=Next)
    button1.grid(row=5,column=4)

    image2=PhotoImage(file="help.png")
    button2=Button(window2,image=image2,command=Help)
    button2.grid(row=4,column=0,padx=4,pady=4,ipadx=3,ipady=3)

    contact=Label(window2,text="For any query:\nReach us @\n kunalkk477@gmail.com")
    contact.grid(row=7,column=4)

    image3=PhotoImage(file="home1.png")
    button3=Button(window2,image=image3,command=Home)
    button3.grid(row=5,column=1)

    window2.mainloop()
def mainwindow():
    window=Tk()
    frame=Frame(window)
    frame.grid()
    frame=Frame(window,width=200,height=500)

    window.wm_title("welcome")

    def destroy1():
        window.destroy()

    def team():
        team1()

    def Next():
        username=Entry1.get()
        print(username)
        Entry1.delete(0,END)
        window.destroy()
        nextwindow()

    name=StringVar()

    label1=Label(window,text="Welcome to\n SPADE",font="Times 32",justify="center",pady=4)
    label1.grid(row=0,column=0,columnspan=4)

    khalilabel1=Label(window,text="         ")
    khalilabel1.grid(row=1,column=0,columnspan=4)

    image1=PhotoImage(file='spades.png')
    label2=Label(window,image=image1,justify="center",pady=4)
    label2.grid(row=2,column=0,columnspan=4)

    khalilabel2=Label(window,text="         ")
    khalilabel2.grid(row=3,column=0,columnspan=4)

    label3=Label(window,text="Seizure Prediction and Detection Engine",font="Helvetica 20",justify="center",padx=4,pady=4)
    label3.grid(row=4,column=0,columnspan=4)

    label4=Label(window,text="Name",font="Helvetica 24 ")
    label4.grid(row=5,column=0,columnspan=2,rowspan=2)

    Entry1=Entry(window,textvariable=name,width=50)
    Entry1.grid(row=5,column=2,columnspan=2,rowspan=2)

    image3=PhotoImage(file="next.png")
    button1=Button(window,image=image3,command=Next)
    button1.grid(row=7,column=3)

    khalilabel3=Label(window,text="         ")
    khalilabel3.grid(row=8,column=0,columnspan=4)

    contact1=Label(window,text="\nContact us:\n+91-8195905598\n+91-9811967439",font="times 10 bold",anchor=W,justify=LEFT)
    contact1.grid(row=10,column=0)

    contact2=Label(window,text="\nEmail:\n Kunalkk477@gmail.com",font="times 10 bold",anchor=W,justify=LEFT)
    contact2.grid(row=10,column=3)

    button2=Button(window,text="Developer Team",font="Helvetica 10",command=team)
    button2.grid(row=11,column=3)

    image2=PhotoImage(file="close.png")
    button3=Button(window,image=image2,command=destroy1)
    button3.grid(row=12,column=2)


    window.mainloop()
mainwindow()

print(starttime,endtime,type(starttime),type(endtime))
# print(starttime1,endtime1,type(starttime1),type(endtime1))
#
spade_path = path + '\\SPADE'
#
num_of_channels, channels, sampling_freq = fetch_system_info(spade_path)
preictal_info = fetch_preictal_info(spade_path)
interictal_info = fetch_interictal_info(spade_path)

extract_seizure_data(src = source, des = spade_path, preictal_info = preictal_info, interictal_info = interictal_info, sampling_freq = sampling_freq)

createDataset(addr = spade_path, start_time= starttime , end_time = endtime, num_of_channels = num_of_channels, sampling_freq = sampling_freq, channels = channels)

trainModel(addr = spade_path,start_time = starttime, end_time = endtime)

print(path,source)
# print(5)
# for i in range(100):
#     print(i**2)
