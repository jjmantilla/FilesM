from db_marche import Database
from db_marche.process.Feature import Feature
from db_marche.process.pattern import Pattern
from db_marche.process.step_detection import StepDetection
from db_marche.features.step_feat import *
import time
import pickle
import scipy
from os.path import exists, join
from operator import attrgetter,itemgetter
from flask import Flask, render_template
import numpy as np
from bokeh.models.widgets import Panel, Tabs,TextInput,Slider,RadioButtonGroup
from bokeh.plotting import figure, gridplot,hplot,vplot
from bokeh.models import ColumnDataSource,BoxAnnotation
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.resources import CDN
from bokeh.embed import file_html
import glob
import os
from os.path import join as j
from bokeh.io import output_file, show, vform
import collections

from csv import DictReader
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'raw'))
#DATA_DIR='./raw/'
SENSORS = ['Pied Droit', 'Pied Gauche', 'Ceinture', 'Tete']
N_SENSORS = ['2AC', '2B5', '2BB', '2BC']
CAPTOR_ID = '_00B41'
G = 9.80665

app = Flask(__name__)

# fname='FEB-Mic-YO-1-TCon-PARTX.csv'
# fname='ILI-Pie-YO-2-Xsens-PARTX.csv'
# fname='BUI-Jea-YO-1-Xsens-PARTX.csv'
# fname='BOU-Odi-YO-1-Xsens-PARTX.csv'
# fname='FEB-Mic-YO-2-TCon-PARTX.csv'
# fname='HAR-Ber-YO-2-TCon-PARTX.csv'
# fname='MAZ-Ber-YO-1-TCon-PARTX.csv'
# fname='ILI-Pie-YO-1-Xsens-PARTX.csv'
# fname='RAG-Jan-YO-1-TCon-PARTX.csv'
# fname='RAG-Jan-YO-2-TCon-PARTX.csv'
# fname='DON-Hen-YO-4-TCon-PARTX.csv'
# fname='JAL-Mar-YO-2-TCon-PARTX.csv'




db=Database(debug=1)
DATA_FOLDER=db.data_folder
urls3={}
for pattern in [j(DATA_FOLDER, 'Data', '*Tete.csv'),
                        j(DATA_FOLDER, 'Raw', '*672.txt'),
                        j(DATA_FOLDER, 'Raw', '*datas-1.txt')]:
            for fname in glob.glob(pattern):
                fname = fname.split(os.sep)[-1][0:13]
                urls3[fname] = '/list2/'+fname

sorted(urls3.items(), key=itemgetter(1))

pname=DATA_DIR
l_dirs = glob.glob(j(pname, '*.txt'))
dirname=[]
for ldir in l_dirs:
    e=ldir.split(os.sep)[-1][0:11]
    dirname.append(e)

l=[item for item, count in collections.Counter(dirname).items() if count > 1]
urls2 = {}
for t in l:
    urls2[t]='/course/'+t


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/my-link/<s1>')
def my_link(s1):
    #err=toe_off_heel_strike()

    ex = db.get_data(fname=s1)[0]
    seg = ex.seg_annotation
    print("segmentation",seg)
    steps1=ex.steps

    ####################################################################################################################

    # T = len(ex.CRX[0][seg[0]:seg[1]])
    # t = np.arange(T)/100
    # plot = figure(width=350, plot_height=250, title="Aller")
    # plot.line(t,ex.rCRX[0][seg[0]:seg[1]])
    #
    # T = len(ex.CRX[0][seg[1]:seg[2]])
    # t = np.arange(T)/100
    # plot1 = figure(width=350, plot_height=250, title="u-Turn")
    # plot1.line(t,ex.rCRX[0][seg[1]:seg[2]])
    #
    # T = len(ex.CRX[0][seg[2]:seg[3]])
    # t = np.arange(T)/100
    # plot2 = figure(width=350, plot_height=250, title="Return")
    # plot2.line(t,ex.rCRX[0][seg[2]:seg[3]])
    #
    # p = hplot(plot, plot1, plot2)
    # tab1 = Panel(child=p, title="Segmentation")

    ####################################################################################################################

    dictio2={}
    dictio2['value']=[  str(ex.seg_annotation),
    str(ex.steps), str(ex.fname), str(ex.g_sensor), str(ex.data_sensor),
                        str(ex.data_earth), str(ex.steps_meta), str(ex.g_earth), str(ex.meta)]
    dictio2['feature']=['seg_annotation', 'steps', 'fname', 'g_sensor', 'data_sensor', 'data_earth',
                        'steps_meta', 'g_earth', 'meta']
    print(dictio2)
    source = ColumnDataSource(dictio2)
    columns = [
        TableColumn(field="feature", title="Field"),
        TableColumn(field="value", title="Value"),
    ]
    data_table = DataTable(source=source, columns=columns, width=1000, height=680)
    tab2 = Panel(child=data_table, title="Exercise")

    ####################################################################################################################

    # dictio2={}
    # l=list(ex.feats_desc.items())
    # dictio2['feature']=[str(l[t][1][0]) for t in range(len(l))]
    # dictio2['desc']=[str(l[t][1][1]) for t in range(len(l))]
    # source = ColumnDataSource(dictio2)
    # columns = [
    #     TableColumn(field="desc", title="Description"),
    #     TableColumn(field="feature", title="Value"),
    # ]
    # data_table = DataTable(source=source, columns=columns, width=1024)
    # tab3 = Panel(child=data_table, title="Features")

    ####################################################################################################################

    foot=0
    sigRY = ex.data_sensor[6*foot+4]
    s3=np.sqrt((ex.get_signal("DAX")[0]**2)+ (ex.get_signal("DAY")[0]**2) + (ex.get_signal("DAZ")[0]**2) )
    sigRY = sigRY - sigRY.mean()
    s=sigRY
    sA=sigRY

    s2=cleansignal(s)
    plotP = figure(width=1020, plot_height=450, title="Pied droit")
    plotP.line(np.arange(len(s3))/100,s3*100, line_color="red" ,legend="AV")
    plotP.line(np.arange(len(sA))/100,sA,line_color="blue",legend="RY")
    for n1 in range(len(steps1[foot])):
        plotP.add_layout(BoxAnnotation(left=(steps1[foot][n1][0])/100, right=(steps1[foot][n1][1])/100, fill_alpha=0.1, fill_color='yellow'))
        r=peaks(-s2[steps1[foot][n1][0]:steps1[foot][n1][1]])
        r=r+steps1[foot][n1][0]
        tr=len(r)
        if tr<2 and tr>0:
            plotP.circle(r[0]/100,sA[r[0]],size=8,color='red',legend="Toe off")
        if tr==2:
            plotP.circle(r[0]/100,sA[r[0]],size=8,color='red',legend="Toe off")
            plotP.circle(r[1]/100,sA[r[1]],size=8,color='blue',legend="Heel strike")
        if tr<2 and tr>0:
            plotP.add_layout(BoxAnnotation(left=(r[0])/100, right=(r[0])/100, fill_alpha=0.1, fill_color='green'))
        if tr==2:
            plotP.add_layout(BoxAnnotation(left=(r[0])/100, right=(r[1])/100, fill_alpha=0.1, fill_color='green'))
    #plotP.add_layout(BoxAnnotation(left=(seg[1])/100, right=(seg[2])/100, fill_alpha=0.1, fill_color='cyan'))

    foot=1
    sigRY = ex.data_sensor[6*foot+4]
    sigRY = sigRY - sigRY.mean()
    s=sigRY
    sA=sigRY
    s2=cleansignal(s)
    s3=np.sqrt((ex.get_signal("GAX")[0]**2)+ (ex.get_signal("GAY")[0]**2) + (ex.get_signal("GAZ")[0]**2) )
    plotP2 = figure(width=1020, plot_height=450, title="Pied Gauche")
    plotP2.line(np.arange(len(s3))/100,s3*100, line_color="red" ,legend="AV")
    plotP2.line(np.arange(len(sA))/100,sA, line_color="blue", legend="RY")
    for n1 in range(len(steps1[foot])):
        plotP2.add_layout(BoxAnnotation(left=(steps1[foot][n1][0])/100, right=(steps1[foot][n1][1])/100, fill_alpha=0.1, fill_color='yellow'))
        r=peaks(-s2[steps1[foot][n1][0]:steps1[foot][n1][1]])
        r=r+steps1[foot][n1][0]
        tr=len(r)
        if tr<2 and tr>0:
            plotP2.circle(r[0]/100,sA[r[0]],size=8,color='red',legend="Toe off")
        if tr==2:
            plotP2.circle(r[0]/100,sA[r[0]],size=8,color='red',legend="Toe off")
            plotP2.circle(r[1]/100,sA[r[1]],size=8,color='blue',legend="Heel strike")
        if tr<2 and tr>0:
            plotP2.add_layout(BoxAnnotation(left=(r[0])/100, right=(r[0])/100, fill_alpha=0.1, fill_color='green'))
        if tr==2:
            plotP2.add_layout(BoxAnnotation(left=(r[0])/100, right=(r[1])/100, fill_alpha=0.1, fill_color='green'))
    #plotP2.add_layout(BoxAnnotation(left=(seg[1])/100, right=(seg[2])/100, fill_alpha=0.1, fill_color='red'))

    vp=vplot(plotP,plotP2)
    tab4 = Panel(child=vp, title="Steps")

    ####################################################################################################################


    tabs = Tabs(tabs=[ tab2,tab4])
    text_input = TextInput(value=ex.fname, title="Enregistrement: ")
    layout = vform(text_input, tabs)
    html = file_html(layout, CDN, "home2")
    return html





@app.route('/list1/<s1>')
def list1(s1):
    s=int(s1)
    exercise = list_exercises[s]
    T = exercise.X.shape[1]
    t = np.arange(T)/100
    plot = figure(width=350, plot_height=250, title="Droit Acceleration X")
    plot.line(t,exercise.get_signal("DAX")[0])
    plot2=figure(width=350, plot_height=250, title="Droit Acceleration Y")
    plot2.line(t,exercise.get_signal("DAY")[0])
    plot3=figure(width=350, plot_height=250, title="Droit Acceleration Z")
    plot3.line(t,exercise.get_signal("DAZ")[0])
    plot4=figure(width=350, plot_height=250, title="Droit Rotation X")
    plot4.line(t,exercise.get_signal("DRX")[0])
    plot5=figure(width=350, plot_height=250, title="Droit Rotation X")
    plot5.line(t,exercise.get_signal("DRY")[0])
    plot6=figure(width=350, plot_height=250, title="Droit Rotation X")
    plot6.line(t,exercise.get_signal("DRZ")[0])
    #p = hplot(plot, plot2)
    p = gridplot([[plot, plot2, plot3], [plot4, plot5, plot6]])
    html = file_html(p, CDN, "home")





    return html


@app.route('/list2/<s1>')
def list2(s1):
    #output_file("home.html")
    #s=int(s1)
    print("Exercise n: ", s1)
    exercise = db.get_data(fname=s1)[0]
    #exercise = list_exercises[s]

    T = exercise.X.shape[1]
    t = np.arange(T)/100

    plot = figure(width=350, plot_height=250, title="Acceleration X")
    plot.line(t,exercise.get_signal("DAX")[0])
    plot2=figure(width=350, plot_height=250, title=" Acceleration Y")
    plot2.line(t,exercise.get_signal("DAY")[0])
    plot3=figure(width=350, plot_height=250, title=" Acceleration Z")
    plot3.line(t,exercise.get_signal("DAZ")[0])
    plot4=figure(width=350, plot_height=250, title=" Rotation X")
    plot4.line(t,exercise.get_signal("DRX")[0])
    plot5=figure(width=350, plot_height=250, title=" Rotation Y")
    plot5.line(t,exercise.get_signal("DRY")[0])
    plot6=figure(width=350, plot_height=250, title=" Rotation Z")
    plot6.line(t,exercise.get_signal("DRZ")[0])
    plot7=figure(width=350, plot_height=250, title=" Norm AV")
    plot7.line(t,np.sqrt((exercise.get_signal("DAX")[0]**2)+ (exercise.get_signal("DAY")[0]**2) + (exercise.get_signal("DAZ")[0]**2) ))
    p1 = gridplot([[plot, plot2, plot3], [plot4, plot5, plot6],[plot7]])
    tab1 = Panel(child=p1, title="Pied Droite")


    plot = figure(width=350, plot_height=250, title="Acceleration X")
    plot.line(t,exercise.get_signal("GAX")[0])
    plot2=figure(width=350, plot_height=250, title=" Acceleration Y")
    plot2.line(t,exercise.get_signal("GAY")[0])
    plot3=figure(width=350, plot_height=250, title=" Acceleration Z")
    plot3.line(t,exercise.get_signal("GAZ")[0])
    plot4=figure(width=350, plot_height=250, title=" Rotation X")
    plot4.line(t,exercise.get_signal("GRX")[0])
    plot5=figure(width=350, plot_height=250, title=" Rotation Y")
    plot5.line(t,exercise.get_signal("GRY")[0])
    plot6=figure(width=350, plot_height=250, title=" Rotation Z")
    plot6.line(t,exercise.get_signal("GRZ")[0])
    plot7=figure(width=350, plot_height=250, title=" Norm AV")
    plot7.line(t,np.sqrt((exercise.get_signal("GAX")[0]**2)+ (exercise.get_signal("GAY")[0]**2) + (exercise.get_signal("GAZ")[0]**2) ))
    p2 = gridplot([[plot, plot2, plot3], [plot4, plot5, plot6],[plot7]])
    tab2 = Panel(child=p2, title="Pied Gauche")

    plot = figure(width=350, plot_height=250, title="Acceleration X")
    plot.line(t,exercise.get_signal("rCAX")[0])
    plot2=figure(width=350, plot_height=250, title=" Acceleration Y")
    plot2.line(t,exercise.get_signal("rCAY")[0])
    plot3=figure(width=350, plot_height=250, title=" Acceleration Z")
    plot3.line(t,exercise.get_signal("rCAZ")[0])
    plot4=figure(width=350, plot_height=250, title=" Rotation X")
    plot4.line(t,exercise.get_signal("rCRX")[0])
    plot5=figure(width=350, plot_height=250, title=" Rotation Y")
    plot5.line(t,exercise.get_signal("rCRY")[0])
    plot6=figure(width=350, plot_height=250, title=" Rotation Z")
    plot6.line(t,exercise.get_signal("rCRZ")[0])
    plot7=figure(width=350, plot_height=250, title=" Norm AV")
    plot7.line(t,np.sqrt((exercise.get_signal("CAX")[0]**2)+ (exercise.get_signal("CAY")[0]**2) + (exercise.get_signal("CAZ")[0]**2) ))
    p = gridplot([[plot, plot2, plot3], [plot4, plot5, plot6],[plot7]])
    tab3 = Panel(child=p, title="Ceinture")


    plot = figure(width=350, plot_height=250, title="Acceleration X")
    plot.line(t,exercise.get_signal("TAX")[0])
    plot2=figure(width=350, plot_height=250, title=" Acceleration Y")
    plot2.line(t,exercise.get_signal("TAY")[0])
    plot3=figure(width=350, plot_height=250, title=" Acceleration Z")
    plot3.line(t,exercise.get_signal("TAZ")[0])
    plot4=figure(width=350, plot_height=250, title=" Rotation X")
    plot4.line(t,exercise.get_signal("TRX")[0])
    plot5=figure(width=350, plot_height=250, title=" Rotation Y")
    plot5.line(t,exercise.get_signal("TRY")[0])
    plot6=figure(width=350, plot_height=250, title=" Rotation Z")
    plot6.line(t,exercise.get_signal("TRZ")[0])
    plot7=figure(width=350, plot_height=250, title=" Norm AV")
    plot7.line(t,np.sqrt((exercise.get_signal("TAX")[0]**2)+ (exercise.get_signal("TAY")[0]**2) + (exercise.get_signal("TAZ")[0]**2) ))
    p = gridplot([[plot, plot2, plot3], [plot4, plot5, plot6],[plot7]])
    tab4 = Panel(child=p, title="Tête")
    tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4])
    text_input = TextInput(value=exercise.fname, title="Enregistrement: ")
    layout = vform(text_input, tabs)

    #show(layout)
    html = file_html(layout, CDN, "home")

    return html




def load_raw_course(fn):

    scale_A=1/G
    scale_R = 180 / np.pi
    delimiter='\t'
    X=[]
    fps = []
    for ns in N_SENSORS:
        fname = fn+ CAPTOR_ID+ns+'.txt'
        res = []
        with open(j(DATA_DIR, fname)) as f:
            t=f.readline()
            l_spr = f.readline()
            f.readline()
            f.readline()

            # Parse categorie name (except sampling rate)
            for row in DictReader(f, delimiter=delimiter):
                res.append([float(row['Acc_X'])*scale_A,
                            float(row['Acc_Y'])*scale_A,
                            float(row['Acc_Z'])*scale_A,
                            float(row['Gyr_X'])*scale_R,
                            float(row['Gyr_Y'])*scale_R,
                            float(row['Gyr_Z'])*scale_R])
        X += [np.transpose(res)]
    return X


@app.route('/course/<files>')
def course(files):
    start = time.clock()
    i=3
    for ir in range(3,5):
        r=files + str(i+1)
        Xr=load_raw_course(r)
        T = len(Xr[0][0][:])
        t = np.arange(T)
        plot1 = figure(width=350, plot_height=250, title="Acceleration  X ")
        plot1.line(t,Xr[0][0][:])
        plot2 = figure(width=350, plot_height=250, title="Acceleration Y " )
        plot2.line(t,Xr[0][1][:])
        plot3 = figure(width=350, plot_height=250, title="Acceleration Z")
        plot3.line(t,Xr[0][2][:])
        plot4 = figure(width=350, plot_height=250, title="Rotation X")
        plot4.line(t,Xr[0][3][:])
        plot5 = figure(width=350, plot_height=250, title="Rotation Y")
        plot5.line(t,Xr[0][4][:])
        plot6 = figure(width=350, plot_height=250, title="Rotation Z")
        plot6.line(t,Xr[0][5][:])
        p = gridplot([[plot1, plot2, plot3], [plot4, plot5, plot6]])
        tab1 = Panel(child=p, title="Tête")
        T = len(Xr[1][0][:])
        t = np.arange(T)/100
        plot1 = figure(width=350, plot_height=250, title="Acceleration X" )
        plot1.line(t,Xr[1][0][:])
        plot2 = figure(width=350, plot_height=250, title="Acceleration Y")
        plot2.line(t,Xr[1][1][:])
        plot3 = figure(width=350, plot_height=250, title="Acceleration Z")
        plot3.line(t,Xr[1][2][:])
        plot4 = figure(width=350, plot_height=250, title="Rotation X")
        plot4.line(t,Xr[1][3][:])
        plot5 = figure(width=350, plot_height=250, title="Rotation Y")
        plot5.line(t,Xr[1][4][:])
        plot6 = figure(width=350, plot_height=250, title="Rotation Z")
        plot6.line(t,Xr[1][5][:])
        p2 = gridplot([[plot1, plot2, plot3], [plot4, plot5, plot6]])
        tab2 = Panel(child=p2, title="Ceinture")
        T = len(Xr[2][0][:])
        t = np.arange(T)
        plot1 = figure(width=350, plot_height=250, title="Acceleration X")
        plot1.line(t,Xr[2][0][:])
        plot2 = figure(width=350, plot_height=250, title="Acceleration Y")
        plot2.line(t,Xr[2][1][:])
        plot3 = figure(width=350, plot_height=250, title="Acceleration Z")
        plot3.line(t,Xr[2][2][:])
        plot4 = figure(width=350, plot_height=250, title="Rotation X")
        plot4.line(t,Xr[2][3][:])
        plot5 = figure(width=350, plot_height=250, title="Rotation Y")
        plot5.line(t,Xr[2][4][:])
        plot6 = figure(width=350, plot_height=250, title="Rotation Z")
        plot6.line(t,Xr[2][5][:])
        p3 = gridplot([[plot1, plot2, plot3], [plot4, plot5, plot6]])
        tab3 = Panel(child=p3, title="Pied Gauche")
        T = len(Xr[3][0][:])
        t = np.arange(T)
        plot1 = figure(width=350, plot_height=250, title="Acceleration X")
        plot1.line(t,Xr[3][0][:])
        plot2 = figure(width=350, plot_height=250, title="Acceleration Y")
        plot2.line(t,Xr[3][1][:])
        plot3 = figure(width=350, plot_height=250, title="Acceleration Z")
        plot3.line(t,Xr[3][2][:])
        plot4 = figure(width=350, plot_height=250, title="Rotation X")
        plot4.line(t,Xr[3][3][:])
        plot5 = figure(width=350, plot_height=250, title="Rotation Y")
        plot5.line(t,Xr[3][4][:])
        plot6 = figure(width=350, plot_height=250, title="Rotation Z")
        plot6.line(t,Xr[3][5][:])
        p4 = gridplot([[plot1, plot2, plot3], [plot4, plot5, plot6]])
        tab4 = Panel(child=p4, title="Pied Droite")
        tabs = Tabs(tabs=[tab1, tab2 , tab3, tab4])
        i=i+1
        if i==1:
            l1=vform(tabs)
        if i==2:
            l2=vform(tabs)
        if i==3:
            l3=vform(tabs)
        if i==4:
            l4=vform(tabs)
        if i==5:
            l5=vform(tabs)

    #ex1 = Panel(child=l1, title="Marche V1:confortable  2min")
    #ex2 = Panel(child=l2, title="Marche V2:4km/h 2min")
    #ex3 = Panel(child=l3, title="Marche V3:limit 1min")
    ex4 = Panel(child=l4, title="Course V3:marche limit 1min")
    ex5 = Panel(child=l5, title="Course V4:limit 2min")
    tabsE = Tabs(tabs=[ ex4, ex5])#ex1, ex2, ex3,
    stop = time.clock()
    t1=stop - start
    text_input = TextInput(value=files+" - time: "+str(t1), title="Enregistrement" )
    l=vform(text_input,tabsE)
    html = file_html(l, CDN, "home")
    return html

def stepdet(ex1):
    patterns = []
    for foot in range(2):
        for st in ex1.steps_annotation[foot]:
            if st[1]-st[0] < 30:
                continue
            patterns += [Pattern(dict(coord='RY', l_pat=st[1]-st[0],
                                    foot='right' if foot else 'left'),
                                 ex1.data_sensor[6*foot+4, st[0]:st[1]])]
            patterns += [Pattern(dict(coord='AZ', l_pat=st[1]-st[0],
                                          foot='right' if foot else 'left'),
                                     ex1.data_sensor[6*foot+2, st[0]:st[1]])]
            patterns += [Pattern(dict(coord='AV', l_pat=st[1]-st[0],
                                          foot='right' if foot else 'left'),
                                     ex1.data_earth[6*foot+2, st[0]:st[1]])]
    stepDet = StepDetection(patterns=patterns, lmbd=.8, mu=.1)
    steps1, steps_label1 = stepDet.compute_steps(ex1)
    return steps1

def stepdetM(ex1):
    list_step = ex1.steps_annotation
    return list_step



import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array


def peaks(vector):

    from peakutils.peak import indexes
    vector=cleansignal(vector)
    thres=1/max(abs(vector))
    indexes2 = indexes(vector, thres, min_dist=len(vector)/3)
    if len(indexes2)==2:

        s=vector[indexes2[0]:indexes2[1]]
        #print(s)
        zero_crossings = np.where(np.diff(np.sign(s)))[0]
        if len(zero_crossings)==2:
            indexes2[1]=zero_crossings[1]+indexes2[0]
    return indexes2

@app.route('/Course/')
def patients(): # why was there urls=urls here before?
    return render_template('patientsc.html',urls=urls2)


@app.route('/Marche/')
def patients2(): # why was there urls=urls here before?
    return render_template('patients.html',urls=urls3)


def toe_off_heel_strike():

    from pymongo import MongoClient
    client = MongoClient('localhost', 27017)
    db2 = client['marche']
    coll = db2.marchecollection
    import pickle
    f= '/Users/jjmantilla/Documents/CMLA/Marche/marche/Exo/ADD-Dan-1.pkl'
    ex1 = db.get_data() #sensor='XSens'
    er=0
    er2=[0,0]
    pasos=0
    print(len(ex1))
    for ex in ex1:
        #tobd(ex,coll)
        #print(ex.fname)
        if ex.fname=="ALV-Gen-YO-4-Xsens-PARTX.csv":
            print("NO")
        else:
            seg = ex.seg_annotation
            #steps1=stepdetM(ex)
            steps1=ex.steps

            for foot in range(2):
                er2[foot]=0
                sigAV = ex.data_earth[6*foot+2]
                sigAZ = ex.data_sensor[6*foot+2]
                sigRY = ex.data_sensor[6*foot+4]
                sigAV = sigAV - sigAV.mean()
                sigAZ = sigAZ - sigAZ.mean()
                sigRY = sigRY - sigRY.mean()
                s=cleansignal(sigRY)

                for n1 in range(len(steps1[foot])):
                    pasos=pasos+1
                    r=peaks(-s[steps1[foot][n1][0]:steps1[foot][n1][1]])
                    #r=r+steps1[foot][n1][0]
                    tr=len(r)
                    if tr<2:
                        er=er+1
                        er2[foot]=er2[foot]+1
            if abs(er2[0]/len(steps1[0])-er2[1]/len(steps1[1]))>0.5:
                print(ex.fname)

            print(ex.seg_from_labels())
            print(ex.seg_annotation)
            uturn=ex.seg_from_labels()
            print("Total pasos: ", pasos)
    return er

def cleansignal(s):
    w = scipy.fftpack.rfft(s)
    spectrum = w**2
    cutoff_idx = spectrum < (spectrum.max()/1000)
    w2 = w.copy()
    w2[cutoff_idx] = 0
    return scipy.fftpack.irfft(w2)


def tobd(ex,coll):
    file_code = ex.fname.replace('-Xsens-PARTX.csv', '')
    file_code = file_code.replace('-TCon-PARTX.csv', '').replace('-YO', '')
    DATA_DIR2 ="/Users/jjmantilla/Documents/CMLA/Marche/marche/"
    f = join(DATA_DIR2, 'Exo', file_code) + '.pkl'
    steps = pickle.load( open( f,'rb') )
    steps2={}
    test_boundaries = []
    for point in steps['data_sensor']:
        test_boundaries.append(point.tolist())
    steps2['data_sensor']=test_boundaries
    test_boundaries = []
    for point in steps['data_earth']:
        test_boundaries.append(point.tolist())
    steps2['data_earth']=test_boundaries
    test_boundaries = []
    for point in steps['g_earth']:
        test_boundaries.append(point.tolist())
    steps2['g_earth']=test_boundaries
    steps2['fname']=steps['fname']
    steps2['_id']=steps['fname']
    post_id = coll.insert_one(steps2)
    return post_id

if __name__ == '__main__':
    app.run(debug=True)


