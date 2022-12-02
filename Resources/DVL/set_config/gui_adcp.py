# -*- coding: utf-8 -*-
"""
Use this script for writing configuration files for RDI ADCP.
Compatible models :
    - Workhorse
    - Pathfinder
    
CHANGELOG:
    - modify TE + TP commands format (for PATHFINDER)
"""

import tkinter as tk
#from tkinter.filedialog import *
from tkinter.messagebox import showinfo # boîte de dialogue
from ToolTip import CreateToolTip

class VarADCP():
    """
     btype :
        1 : Entry
        2 : Radiobutton
        3 : Checkbutton
    """
    def __init__(self, parent, grow, blabel, binfo, btype, vdef, vlims, prefix):
        self.parent = parent
        self.row = grow
        self.blabel = blabel
        self.binfo = binfo
        self.btype = btype
        self.vdef = vdef
        # vlims : pour les boutons à choix type Radiobutton ou Checkbutton,
        # cela représente les différents choix possibles :
        self.vlims = vlims 
        self.prefix = prefix
        
        self.init_value(btype)
        
        self.plot_label()
        self.plot_entry()
        
    def init_value(self, btype):
        if btype == 1:
            vv = tk.StringVar()
        elif btype == 2:
            vv = tk.IntVar()
        elif btype == 3:
            vv = [tk.IntVar() for _ in self.vlims]
        else:
            print("Unknown btype.")
            vv = None
        self.value = vv
    
    def plot_label(self):
        lab = tk.Label(self.parent, text=self.blabel)
        lab.grid(row=self.row, column=1)
        CreateToolTip(lab, text = self.binfo)
        
    def plot_entry(self):
        if self.btype == 1: # Entry
            entry = tk.Entry(self.parent, textvariable=self.value, bg ='bisque', fg='maroon')
            entry.grid(row=self.row, column=2)
            self.value.set(self.vdef)
        elif self.btype == 2: # Radiobutton
            for ii, vv in enumerate(self.vlims):
                bb = tk.Radiobutton(self.parent, text=vv, variable=self.value, value=ii)
                if "Wide" in vv: # "Wide" option not bought by ENSTA
                    bb.configure(state = tk.DISABLED)
                bb.grid(row=self.row, column=ii+2)
                if ii == self.vdef:
                    bb.select()
        elif self.btype == 3: # Checkbutton
            for ii, vv in enumerate(self.vlims):
                bb = tk.Checkbutton(self.parent, text=vv, variable=self.value[ii])
                bb.grid(row=self.row, column=ii+2)
                if self.vdef[ii] == 1:
                    bb.select()
               
def write_file():
    file = open(fn.value.get(), "w")
    # Header
    file.write(";-----------------------------------------------------------------------------\n")
    file.write("; DVL Command File for use with VmDas software.\n")
    file.write(";\n")
    print(dt.vlims)
    print(dt.vlims[dt.value.get()], type(dt.vlims[dt.value.get()]))
    print(st.value.get(), type(st.value.get()))
    file.write("; DVL type:    "+ dt.vlims[dt.value.get()] +"\n")
    file.write("; Setup type:    "+ st.value.get() +"\n")
    file.write(";\n")
    file.write("; NOTE:  Any line beginning with a semicolon in the first\n")
    file.write(";        column is treated as a comment and is ignored by\n")
    file.write(";        the VmDas software.\n")
    file.write(";-----------------------------------------------------------------------------\n")
    file.write(";Restore factory default settings in the ADCP\n")
    file.write("CR1\n")
    for vv in varlist:
        file.write(";" + vv.blabel+"\n")
        if isinstance(vv.value, list):
            file.write(vv.prefix)
            for vvi in vv.value:
                file.write(str(vvi.get()))
            if vv.prefix in "WD": # add 4 unused bits for WD command
                file.write("0000")
            file.write("\n")
        else:
            file.write(vv.prefix + str(vv.value.get()) + "\n")
    # Footer
    file.write(";Save this setup to non-volatile memory in the ADCP\n")
    file.write("CK\n")
    file.write(";Start pinging (GO)\n")
    file.write("CS\n")
    file.close()
    showinfo('Confirmation','Fichier de configuration créé.')
    Mafenetre.destroy()
    
# Création de la fenêtre principale (main window)
Mafenetre = tk.Tk()
Mafenetre.title('Set ADCP parameters (Workhorse/Pathfinder compliant)')

# Création du Frame 'File'
F0 = tk.Frame(Mafenetre,borderwidth=2,relief=tk.GROOVE)
F0.pack(side=tk.TOP,padx=10,pady=10)
tk.Label(F0,text="Configuration file",font='Helvetica 13 bold').grid(row=1, column=1)

# création du Frame 'Environment'
F1 = tk.Frame(Mafenetre,borderwidth=2,relief=tk.GROOVE)
F1.pack(side=tk.TOP,padx=10,pady=10)
tk.Label(F1,text="Environment commands",font='Helvetica 13 bold').grid(row=1, column=1)

# création du Frame 'Time'
F2 = tk.Frame(Mafenetre,borderwidth=2,relief=tk.GROOVE)
F2.pack(side=tk.TOP,padx=10,pady=10)
tk.Label(F2,text="Time commands",font='Helvetica 13 bold').grid(row=1, column=1)

# création du Frame 'Bottom tracking'
F3 = tk.Frame(Mafenetre,borderwidth=2,relief=tk.GROOVE)
F3.pack(side=tk.TOP,padx=10,pady=10)
tk.Label(F3,text="Bottom tracking commands",font='Helvetica 13 bold').grid(row=1, column=1)

# création du Frame 'Water profiling'
F4 = tk.Frame(Mafenetre,borderwidth=2,relief=tk.GROOVE)
F4.pack(side=tk.TOP,padx=10,pady=10)
tk.Label(F4,text="Water profiling commands",font='Helvetica 13 bold').grid(row=1, column=1)

fn = VarADCP(F0,2,"ADCP setup file name", "Choose the name of your setup file.", 1, "default.txt", [],"")
dt = VarADCP(F0,3,"DVL/ADCP type", "Select the sensor model.", 2, 0, ["Pathfinder 600","Workhorse Mariner 1200"],"")
st = VarADCP(F0,4,"Setup description", "Describe your setup purpose here\n(e.g. High resolution, low range profile)", 1, "", [],"")

# Création des boutons :
es = VarADCP(F1,2,"Salinity", "Set the water's salinity value.\nUsed for speed of sound calculation.\n0=fresh water   35=salt water.", 1, 35, [0,40],"ES")
et = VarADCP(F1,3,"Temperature", "Set the water's temperature value.\nUsed for speed of sound calculation.\nUnit : 10°C = 1000.", 1, '1200', [-500,4000],"ET")
ed = VarADCP(F1,4,"Transducer depth (decimeters)", "Set the ADCP transducer depth.\nUsed for speed of sound calculation.\nUnit is dm: for 1m write '10'.", 1, "00005", [0,65535],"ED")
ea = VarADCP(F1,5,"Heading alignment (100*deg)", "Correct for physical misalignment between Beam 3 and the heading reference.\n Pathfinder/Workhorse ADCP are mounted on a ship with beam 3 aligned at\na +45° angle (ie. clockwise) from the forward axis of the ship.", 1, 4500, [-17999,18000],"EA")
ex = VarADCP(F1,6,"Coordinate Transformation", "Sets the coordinate transformation processing flags.\nBETA: KEEP DEFAULT VALUES PLEASE", 1, "00010", [00000,11111],"EX")
ez = VarADCP(F1,7,"Sensor sources", "Select the sources of environmental sensor data.\nSpeed of sound: 0 means manual value, 1 means computed with depth, salinity and temperature values.\nOthers: 0 means manual value, 1 means internal sensor.\nBETA: KEEP DEFAULT VALUES PLEASE", 3, [1,0,0,0,0,0,0,0], 
             ["Speed of sound","Depth","Heading","Pitch","Roll","Salinity","Temp","Up/Down orientation"],"EZ")

te = VarADCP(F2,2,"Time between ensemble", "Set the minimum interval between data collection cycles.\nFormat : hhmmssff\nhh = 00 to 23 hours\nmm = 00 to 59 mins\nss = 00 to 59 secs\nff = 00 to 99 hundredths of secs", 1, "00:00:00.20", [0,23595999],"TE")
tp = VarADCP(F2,3,"Minimum time between pings", "Set the minimum time between pings.\nFormat : mmssff\nmm = 00 to 59 mins\nss = 00 to 59 secs\nff = 00 to 99 hundredths of secs", 1, "00:00:00", [0,595999],"TP")

bp = VarADCP(F3,2,"Number of bottom-track pings to average", "Set the number of bottom-track pings to average together.\nIf 0, the ADCP will not collect bottom track data.", 1, "001", [0,999],"BP")
bx = VarADCP(F3,3,"Maximum bottom search depth (decimeters)", "Set the maximum tracking depth in bottom-track mode.\nUnit is dm: for 20m write '200'.", 1, 500, [3,1100],"BX")

wb = VarADCP(F4,2,"Bandwidth profile mode", "Set the profiling bandwidth (sampling rate).\nSmaller bandwidths allow the ADCP to profile farther,\nbut the standard deviation is increased.\nWide option not bought by ENSTA.", 2, 1, ["Wide","Narrow"],"WB")
ws = VarADCP(F4,3,"Depth cells size (cm)", "Set the size of each cell in vertical centimeters.", 1, "050", [10,400],"WS")
wn = VarADCP(F4,4,"Depth cells number", "Set the number of depth cells over which the ADCP collect data.", 1, "050", [1,255],"WN")
wf = VarADCP(F4,5,"Blank after transmit (cm)", "Moves the location of first depth cell away from\nthe transducer head to allow the transmit circuit\nto recover before the receive cycle begins.", 1, "088", [0,9999],"WF")
wv = VarADCP(F4,6,"Radial ambiguity velocity (m/s)", "Set the radial ambiguity velocity for profile and water-mass mode.", 1, 330, [20,700],"WV")
wd = VarADCP(F4,7,"Data out", "Select the data types collected by the ADCP.", 3, [1,1,1,1,1], 
             ["Velocity","Correlation","Echo intensity","Percent good","Status"],"WD")
wp = VarADCP(F4,8,"Number of water data pings to average", "Set the number of water column data pings to average together.\nIf 0, the ADCP will not collect water column data.", 1, "00001", [0,16384], "WP")

# Variables qui sont enregistrées dans le fichier de config :
varlist = [wb,wp,wn,ws,wf,wv,wd,bp,bx,tp,te,es,ed,ea,ex,ez]

Bouton = tk.Button(Mafenetre, text ='Save file', command = write_file)
Bouton.pack(side = tk.BOTTOM, padx = 5, pady = 5)

Mafenetre.mainloop()

