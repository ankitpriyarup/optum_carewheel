import os
import csv
from os import listdir
from collections import OrderedDict
from math import log, sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from six.moves import cStringIO as StringIO
from bokeh.plotting import figure, show, output_file, save
from sklearn import svm
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

ROOT = ''

# region HELPER_FUNCTIONS
def timeconvert(time):
	year = 2012
	month = 1
	day = 1
	hour = int(time.split(":")[0])
	minute = int(time.split(":")[1])

	if (hour>23)&(hour<48):
		day = 2
		hour = int(hour - 24)
	elif hour == 48:
		day = 3
		hour = 0

	newtime = datetime.datetime(year, month, day, hour, minute)
	return newtime


def load_file(filename):
	with open(filename) as f:
		reader = csv.reader(f, delimiter=',')
		header = next(reader)
		d = {}
		for var in variables:
			d[var] = []
		_, _, idx = next(reader)
		d['RecordID'] = idx
		for row in reader:
			time, variable, value = row
			value = float(value)
			d[variable].append(value)
	return d


def load_results(filename):
	with open(filename) as f:
		reader = csv.reader(f, delimiter=',')
		header = next(reader)
		d = {}
		for row in reader:
			idx, saps, sofa, len_stay, survival, in_hos_death = row
			d[idx] = int(in_hos_death)
	return d


def to_vec(x):
	vec = []
	for var in variables[:-1]:
		vec.append(x[var])
	if x['ICUType'] == 1.0:
		vec += [0.0, 0.0, 0.0]
	elif x['ICUType'] == 2.0:
		vec += [1.0, 0.0, 0.0]
	elif x['ICUType'] == 3.0:
		vec += [0.0, 1.0, 0.0]
	else:
		vec += [0.0, 0.0, 1.0]
	return vec


def to_disc_vec(x, disc_vars):
	vec = []
	for var in disc_vars:
		if var != 'ICUType':
			vec.append(x[var])
	if x['ICUType'] == 1.0:
		vec += [1]
	elif x['ICUType'] == 2.0:
		vec += [2]
	elif x['ICUType'] == 3.0:
		vec += [3]
	else:
		vec += [4]
	return vec


def to_cont_vec(x, cont_vars):
	vec = []
	for var in cont_vars:
		vec.append(x[var])
	return vec


def mytype(list):

	if list[0]>list[1]:
		return 0
	else:
		return 1


def leftcompare(list1, list2):
	n = 0
	for i in range(0, len(list1)):
		if(list1[i]==list2[i])&(list1[i]==1):
			n = n + 1

	return float(n)/float(len(list1))


variables = ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
			 "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose",
			 "HCO3", "HCT", "HR", "K", "Lactate", "Mg",
			 "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP",
			 "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
			 "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
			 "Weight", "Height", "Gender", "Age", "ICUType"]
# endregion


class patient(object):

	def __init__(self, filename):
		self.filename = filename
		self.f = open(os.path.join(ROOT, 'data/set-a/' + self.filename), 'r')
		self.row = []
		for row in self.f.readlines():
			self.row.append(row)

	def convert_to_time_value(self):
		var_list = ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
				"Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose",
				"HCO3", "HCT", "HR", "K", "Lactate", "Mg",
				"MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP",
				"PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
				"SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
				"Weight", "ICUType", "Height", "Gender", "Age", "RecordID"]

		pa_dic = {}
		pa_dic["time"] = []
		for var in var_list:
			pa_dic[var] = []

		for row in self.row[1:]:
			pa_dic["time"].append(row.split(',')[0])
			pa_dic[row.split(',')[1]].append([row.split(',')[0],row.rstrip().split(',')[2]])

		return pa_dic


def patient_level_timer_var(patientID):
    patientID += '.txt'
    new = patient(patientID)
    result = new.convert_to_time_value()
    return result
    #print result


def show_scatter_plot(setName, observe_marker, pick_count):
    # READ DATA AND OUTCOMES
    outcomes = []
    with open('data/Outcomes-' + setName + '.txt', 'r') as f:
        outcomes = f.readlines()
    RecordIDs = []
    for f in listdir("./data/set-" + setName):
        RecordIDs.append(f)

    # RANDOMLY PICK *pick_count* SURVIVAL PATIENTS AND 1 DEAD PATIENT
    dead_list = []
    survival_list = []
    for n in range(1,(len(RecordIDs))):
        if(outcomes[n].rstrip().split(',')[5] == '1'):
            dead_list.append(n)
        else:
            survival_list.append(n)
    random_list =[]
    random_list.append(random.sample(dead_list,1)[0])
    for j in range(0,pick_count):
        random_list.append(random.sample(survival_list,pick_count)[j])

    # PLOT THE LINE GRAPH SHOWING TIME AND TEMPERATURE
    output_file(os.path.join(ROOT, "data", "scatter.html"), title="patient_" + observe_marker + "_time.py example")
    p = figure(width=800, height=250, x_axis_type="datetime", title="Plot")
    p.xaxis.axis_label = "Time"
    p.yaxis.axis_label = observe_marker

    for i in random_list:
        test_patient = patient(RecordIDs[i])
        observation = test_patient.convert_to_time_value()
        data=[]

        # DEAD IN HOSPITAL IS MARKED BY RED LINE
        # SURVIVAL IS MARKED BY NAVY LINE
        if (outcomes[i].rstrip().split(',')[5]=='1'):
            linecolor = 'red'
            linelegend = 'dead'
        else:
            linecolor = 'navy'
            linelegend = 'survival'
        for n in range(0,len(observation[observe_marker])):
            observe_time = timeconvert(observation[observe_marker][n][0])
            observe_field = float(observation[observe_marker][n][1])
            data.append((observe_time, observe_field))
        data_array = np.array(data)
        p.line(data_array[:,0], data_array[:,1], color=linecolor, alpha=0.5, legend = linelegend)
    p.legend.label_text_font = "times"
    save(p)
    # return open(os.path.join(ROOT, "scatter.html"),"r").read().replace('\n', '').replace('\"', "'")


def get_patient_avg_val(ID, var):
    test_patient = patient(ID)
    observation = test_patient.convert_to_time_value()
    if len(observation[var]) == 0:
        return 0

    average_list = []

    for n in range(0,len(observation[var])):
        value = float(observation[var][n][1])
        if(value < 0):
            continue
        else:
            average_list.append(value)

    if len(average_list) == 0:
        return 0

    average = sum(average_list) / float(len(average_list))
    return average


def get_avg_value_of_each_variable(setName):
    # READ DATA AND OUTCOMES
    outcomes = []
    with open(os.path.join(ROOT, 'data/Outcomes-' + setName +'.txt'), 'r') as f:
        outcomes = f.readlines()

    RecordIDs = []
    for f in listdir(os.path.join(ROOT, "./data/set-" + setName)):
        RecordIDs.append(f)

    # GET THE LIST OF DEAD AND SURVIVAL PATIENTS
    dead_list = []
    survival_list = []

    for n in range(1,(len(RecordIDs))):

        if(outcomes[n].rstrip().split(',')[5] == '1'):
            dead_list.append(n)
        else:
            survival_list.append(n)

    var_list = ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
                    "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose",
                    "HCO3", "HCT", "HR", "K", "Lactate", "Mg",
                    "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP",
                    "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
                    "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
                    "Weight", "Height", "Age"]

    # average value of each variable of dead patients
    for var in var_list:
        var_ave = []

        for id in dead_list:
            ID = RecordIDs[id]
            var_ave.append(get_patient_avg_val(ID, var))

        #print "variable is:", var, "and the average is:", sum(var_ave) / float(len(var_ave))

    # average value of each variable of survival patients
    for var in var_list:
        var_ave = []

        for id in survival_list:
            ID = RecordIDs[id]
            var_ave.append(get_patient_avg_val(ID, var))

        #print "variable is:", var, "and the average is:", sum(var_ave) / float(len(var_ave))


def show_patient_plot(patient_data):
    patient_color = OrderedDict([
        ("Survival",   "#c64737"),
        ("Dead",     "#0d3362"  ),
    ])
    df = pd.read_csv(StringIO(patient_data),
                    skiprows=1,
                    skipinitialspace=True,
                    engine='python')

    inner_radius = 90
    outer_radius = 300 - 10

    minr = sqrt(log(.001 * 1E4))
    maxr = sqrt(log(1000 * 1E4))
    a = (outer_radius - inner_radius) / (minr - maxr)
    b = inner_radius - a * maxr

    def rad(mic):
        return a * np.sqrt(np.log(mic * 1E4)) + b

    big_angle = 2.0 * np.pi / (len(df) + 1)
    small_angle = big_angle / 7

    x = np.zeros(len(df))
    y = np.zeros(len(df))

    output_file(os.path.join(ROOT, "data", "report.html"), title="average variables of survival and dead patients")

    p = figure(plot_width=570, plot_height=500, title="",
        x_axis_type=None, y_axis_type=None,
        x_range=[-420, 420], y_range=[-420, 420],
        min_border=0, outline_line_color="black")

    p.line(x+1, y+1, alpha=0)
    angles = np.pi/2 - big_angle/2 - df.index.to_series()*big_angle
    p.annular_wedge(
        x, y, inner_radius, outer_radius, -big_angle+angles, angles, color='#e69584'
    )
    p.annular_wedge(x, y, inner_radius, rad(df.survival),
        -big_angle+angles+5*small_angle, -big_angle+angles+6*small_angle,
        color=patient_color['Survival'])

    p.annular_wedge(x, y, inner_radius, rad(df.dead),
        -big_angle+angles+1*small_angle, -big_angle+angles+2*small_angle,
        color=patient_color['Dead'])
    labels = np.power(10.0, np.arange(-3, 4))
    radii = a * np.sqrt(np.log(labels * 1E4)) + b
    p.circle(x, y, radius=radii, fill_color=None, line_color="white")
    p.text(x[:-1], radii[:-1], [str(r) for r in labels[:-1]],
        text_font_size="8pt", text_align="center", text_baseline="middle")
    p.annular_wedge(x, y, inner_radius-10, outer_radius+10,
        -big_angle+angles, -big_angle+angles, color="black")
    xr = radii[0]*np.cos(np.array(-big_angle/2 + angles))
    yr = radii[0]*np.sin(np.array(-big_angle/2 + angles))
    label_angle=np.array(-big_angle/2+angles)
    label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
    p.text(xr, yr, df.variables, angle=label_angle,
        text_font_size="9pt", text_align="center", text_baseline="middle")

    p.rect([-40, -40, ], [18, 0], width=30, height=13,
        color=list(patient_color.values()))
    p.text([-15, -15], [18, 0], text=list(patient_color.keys()),
        text_font_size="9pt", text_align="left", text_baseline="middle")

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    save(p)
    # return open(os.path.join(ROOT, "report.html"),"r").read().replace("\n", "").replace('\"', "'")


def mortality_plot(type):
    typALL = """
    variables, survival, dead
    Albumin, 1.1518906784, 1.49846613375
    ALP, 38.9586900847, 76.859215088
    ALT, 60.9184730984, 154.58952209
    AST, 76.8053950813, 270.996248782
    Bilirubin, 0.592279039881, 2.21461456364
    BUN, 23.277702843, 36.7815788103
    Cholesterol,   11.8743470691, 12.3393501805
    Creatinine, 1.28057115689, 1.76910688564
    DiasABP, 41.4940156671, 42.7322274708
    FiO2,  0.360537675575, 0.450833056257
    GCS,  11.7345551978, 9.45314906374
    Glucose,    131.434187701, 145.995310115
    HCO3, 23.3801901005, 22.109043158
    HCT, 30.9088795786, 31.0957918405
    HR, 85.268220221, 89.0124949418
    K, 4.0239520255, 4.10960690633
    Lactate, 1.12826875867, 2.12307788422
    Mg, 1.95452307462, 2.03935531827
    MAP,  56.2342735863, 58.704807735
    MechVent, 0.619268717353, 0.712996389892
    Na, 136.0427718, 137.578854689
    NIDiasABP, 50.2164518916, 48.2115363125
    NIMAP, 66.6528324997, 64.8306405855
    NISysABP,  102.659746579, 100.894294256
    PaCO2, 30.0883484763, 33.1956542309
    PaO2, 111.170123841, 116.016003702
    pH, 5.57261049715, 6.6080495173
    Platelets,  202.366719846, 198.46288881
    RespRate, 5.74687351483, 3.32674242862
    SaO2, 42.8388292682, 45.88767696
    SysABP, 82.5726623659, 85.5441223263
    Temp, 36.3556487346, 36.5423427493
    TroponinI, 0.287830403781, 0.788402527076
    TroponinT,  0.209555996601, 0.356216305656
    Urine, 135.808379279, 94.5775743826
    WBC, 12.0448733094, 13.9684298764
    Weight, 77.0327811579, 76.0586113727
    Height, 90.9883052815, 79.4696750903
    Age, 63.3273360418, 69.9711191336
    """

    # cardiological
    type1 = """
    variables, survival, dead
    BUN, 23.277702843, 36.7815788103
    Cholesterol,   11.8743470691, 12.3393501805
    FiO2,  0.360537675575, 0.450833056257
    GCS,  11.7345551978, 9.45314906374
    Glucose,    131.434187701, 145.995310115
    HCO3, 23.3801901005, 22.109043158
    HCT, 30.9088795786, 31.0957918405
    HR, 85.268220221, 89.0124949418
    K, 4.0239520255, 4.10960690633
    Lactate, 1.12826875867, 2.12307788422
    Mg, 1.95452307462, 2.03935531827
    MAP,  56.2342735863, 58.704807735
    MechVent, 0.619268717353, 0.712996389892
    Platelets,  202.366719846, 198.46288881
    Temp, 36.3556487346, 36.5423427493
    Urine, 135.808379279, 94.5775743826
    WBC, 12.0448733094, 13.9684298764
    Weight, 77.0327811579, 76.0586113727
    Height, 90.9883052815, 79.4696750903
    Age, 63.3273360418, 69.9711191336
    """

    # infection
    type2 = """
    variables, survival, dead
    Albumin, 1.1518906784, 1.49846613375
    ALP, 38.9586900847, 76.859215088
    ALT, 60.9184730984, 154.58952209
    AST, 76.8053950813, 270.996248782
    Bilirubin, 0.592279039881, 2.21461456364
    BUN, 23.277702843, 36.7815788103
    FiO2,  0.360537675575, 0.450833056257
    GCS,  11.7345551978, 9.45314906374
    Glucose,    131.434187701, 145.995310115
    HCO3, 23.3801901005, 22.109043158
    HCT, 30.9088795786, 31.0957918405
    HR, 85.268220221, 89.0124949418
    K, 4.0239520255, 4.10960690633
    Lactate, 1.12826875867, 2.12307788422
    Mg, 1.95452307462, 2.03935531827
    Weight, 77.0327811579, 76.0586113727
    Height, 90.9883052815, 79.4696750903
    Age, 63.3273360418, 69.9711191336
    """

    # extensive care
    type3 = """
    variables, survival, dead
    Bilirubin, 0.592279039881, 2.21461456364
    BUN, 23.277702843, 36.7815788103
    DiasABP, 41.4940156671, 42.7322274708
    FiO2,  0.360537675575, 0.450833056257
    GCS,  11.7345551978, 9.45314906374
    Glucose,    131.434187701, 145.995310115
    HCO3, 23.3801901005, 22.109043158
    HCT, 30.9088795786, 31.0957918405
    HR, 85.268220221, 89.0124949418
    K, 4.0239520255, 4.10960690633
    Lactate, 1.12826875867, 2.12307788422
    Mg, 1.95452307462, 2.03935531827
    MAP,  56.2342735863, 58.704807735
    MechVent, 0.619268717353, 0.712996389892
    RespRate, 5.74687351483, 3.32674242862
    SaO2, 42.8388292682, 45.88767696
    SysABP, 82.5726623659, 85.5441223263
    Temp, 36.3556487346, 36.5423427493
    TroponinI, 0.287830403781, 0.788402527076
    TroponinT,  0.209555996601, 0.356216305656
    Urine, 135.808379279, 94.5775743826
    WBC, 12.0448733094, 13.9684298764
    Weight, 77.0327811579, 76.0586113727
    Height, 90.9883052815, 79.4696750903
    Age, 63.3273360418, 69.9711191336
    """

    # lung
    type4 = """
    variables, survival, dead
    Albumin, 1.1518906784, 1.49846613375
    ALP, 38.9586900847, 76.859215088
    ALT, 60.9184730984, 154.58952209
    AST, 76.8053950813, 270.996248782
    Bilirubin, 0.592279039881, 2.21461456364
    BUN, 23.277702843, 36.7815788103
    K, 4.0239520255, 4.10960690633
    Lactate, 1.12826875867, 2.12307788422
    Mg, 1.95452307462, 2.03935531827
    MAP,  56.2342735863, 58.704807735
    MechVent, 0.619268717353, 0.712996389892
    Na, 136.0427718, 137.578854689
    NIDiasABP, 50.2164518916, 48.2115363125
    NIMAP, 66.6528324997, 64.8306405855
    NISysABP,  102.659746579, 100.894294256
    PaCO2, 30.0883484763, 33.1956542309
    PaO2, 111.170123841, 116.016003702
    RespRate, 5.74687351483, 3.32674242862
    Temp, 36.3556487346, 36.5423427493
    TroponinI, 0.287830403781, 0.788402527076
    TroponinT,  0.209555996601, 0.356216305656
    Weight, 77.0327811579, 76.0586113727
    Height, 90.9883052815, 79.4696750903
    Age, 63.3273360418, 69.9711191336
    """

    if type == '0':
        return show_patient_plot(typALL)
    if type == '1':
        return show_patient_plot(type1)
    if type == '2':
        return show_patient_plot(type2)
    if type == '3':
        return show_patient_plot(type3)
    if type == '4':
        return show_patient_plot(type4)


def in_range (lowerlimit, upperlimit, value):
    return lowerlimit <= value <= upperlimit

def calculate_single_scores (value, selector):
    # APACHEII Ranges
    if selector is "abp":
        value_ranges = ([160,300,4],[130,159,3],[110,129,2],[70,109,0],[50,69,2],[0,49,4])
    elif selector is "temperature":
        value_ranges = ([41,50,4],[39,40.9,3],[38.5,38.9,1],[36,38.9,0],[34,35.9,1],[32,33.9,2],[30,31.9,3],[19,29.9,4])
    elif selector is "heart_bpm":
        value_ranges = ([180,300,4],[140,179,3],[110,139,2],[70,109,0],[55,69,2],[40,45,4],[0,49,4])
    elif selector is "respiratory_rate":
        value_ranges = ([50,100,4],[35,49,3],[25,34,2],[12,24,0],[10,11,1],[6,9,2],[0,5,4])
    elif selector is "oxygenation": # see explanation!
        value_ranges = ([500,1000,4],[350,499,3],[200,349,2],[70,200,0],[61,70,1],[55,60,3],[0,55,4])
    elif selector is "ph":
        value_ranges = ([7.7,9.0,4],[7.6,7.69,3],[7.5,7.59,1],[7.33,7.49,0],[7.25,7.32,2],[7.15,7.24,3],[5,7.15,4])
    elif selector is "sodium":
        value_ranges = ([180,300,4],[160,179,3],[155,159,2],[150,154,1],[130,149,0],[120,129,2],[111,119,3],[50,110,4])
    elif selector is "potassium":
        value_ranges = ([7,9,4],[6,6.9],[5.5,5.9],[3.5,5.4,0],[3,3.4,1],[2.5,2.9,2],[2.5,2,4])
    #elif selector is "creatinine":
     #   value_ranges = ([3.5,30,4],[2,3.4,3],[1.5,1.9,2],[0.6,1.4,0],[0,0.6,2])
    elif selector is "hematocrit":
        value_ranges = ([60,100,4],[50,59.9,2],[46,49.9,1],[30,45.9,0],[20,29.9,2],[0,20,4])
    elif selector is "wbc":
        value_ranges = ([40,100,4],[20,39.9,2],[15,19.9,1],[3,14.9,0],[1,2.9,2],[0,1,4]) 
    elif selector is "age":
        value_ranges = ([18,44,0],[45,54,2],[55,64,3],[65,74,5],[75,120,6]) 
    else:
        value_ranges = 0

    for value_range in value_ranges:
        if in_range(value_range[0], value_range[1], value):
            return int(value_range[2])
    return 0

def calculate_apache2_physiology(abp,temperature,heart_bpm,respiratory_rate,oxygenation,ph,sodium,potassium,hematocrit,wbc,age):
    args = list(locals().values())
    #print(args)
    list_of_scores = []
    selectors = list(reversed(["abp","temperature","heart_bpm","respiratory_rate","oxygenation","ph","sodium","potassium","hematocrit","wbc", "age"]))
    for arg, selector in zip(args, selectors):
        #print(selector, arg)
        result = calculate_single_scores(arg, selector)
        list_of_scores.append(int(result))
    return(sum(list_of_scores))

#This function accepts a total of 6 values for adressing Chronic Health Points and surgery status
#value="RO NSCD">elective postoperative, no SCD (Severe Chronic Diseases, see list)
#value="RO SCD">elective postoperative, SCD present
#value="NO NSCD">no surgery, no SCD
#value="NO SCD">no surgery, SCD present
#value="EO NSCD">emergency postoperative, no SCD
#value="EO SCD">emergency postoperative, SCD present

def chronic_health_score(value):

    if value == "RO NSCD":
        result = "0"
    elif value == "RO SCD":
        result = "2"
    elif value == "EO NSCD":
        result = "0"
    elif value == "EO SCD":
        result = "5"
    elif value == "NO NSCD":
        result = "0"
    elif value == "NO SCD":
        result = "5"
    return(result)

def calculate_apache(abp,temperature,heart_bpm,respiratory_rate,oxygenation,ph,sodium,potassium,hematocrit,wbc,age,scd,aki,creatinine,gcs):
    args = list(locals().values())
    physiology = calculate_apache2_physiology(abp,temperature,heart_bpm,respiratory_rate,oxygenation,ph,sodium,potassium,hematocrit,wbc,age)
    health_points = int(chronic_health_score(scd))
    # Double points for acute kidney injury
    if aki == 0:
        value_ranges = ([3.5,30,4],[2,3.4,3],[1.5,1.9,2],[0.6,1.4,0],[0,0.6,2])
    else:
        value_ranges = ([3.5,30,8],[2,3.4,6],[1.5,1.9,4],[0.6,1.4,0],[0,0.6,4])

    for value_range in value_ranges:
        if in_range(value_range[0], value_range[1], creatinine):
            creatinine_points = int(value_range[2])
    gcs_final = 15 - gcs
    final_score = int(physiology)+int(health_points)+int(creatinine_points)+gcs_final
    return(final_score)


# ---- USAGE ----

# print(calculate_apache(60,37,90,20,300,7.7,190,7,50,22,44,"NO NSCD",1,7,15))
# print('\n')
# print('\n')
# print('Performing Patient Timer Analysis of ID:132539')
# patient_level_timer_var('132539.txt')

# print('\n')
# print('\n')
# print('Plotting Platelets plot with time showing dead and survival')
# # -> Types: [BUN, Creatinine, GCS, Glucose, HCO, HCT, HR, K, Mg, Na, PaO2, Platelets, Temp, Urine, WBC, Weight, Height, Gender, Age, ICUType]
# show_scatter_plot('a', 'Platelets', 7)

# print('\n')
# print('\n')
# print('Showing average value of each variable in set-a')
# get_avg_value_of_each_variable('a')