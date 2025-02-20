# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:31:51 2024

@author: kernke
"""

import h5py
import numpy as np
from numba import njit
import numba
from datetime import datetime

# use clock inside numba njit functions
#https://github.com/numba/numba/issues/4003
import ctypes
import platform
if platform.system() == "Windows":
    from ctypes.util import find_msvcrt
    __LIB = find_msvcrt()
    if __LIB is None:
        __LIB = "msvcrt.dll"
else:
    from ctypes.util import find_library
    __LIB = find_library("c")
clock = ctypes.CDLL(__LIB).clock
clock.argtypes = []



#%% single line simple

@njit
def single_line_exponential(Ek,Ekw,T,duration,frequency,sampling_dt,
                activation=None,t_extra_in=None,timelim_sec=30,
                fixed_prob=2*10**-6):#Emax=100.

    kB=8.617*10**-5
    kBT=kB*T
    Nequisteps=int(duration//sampling_dt)+1
    length=np.empty(Nequisteps,dtype=numba.int64)
    E0=np.empty(Nequisteps)
    t_end=duration*frequency
    sample_dt=sampling_dt*frequency
    glidel=np.int64(0)
    length[0]=glidel
    t=0
    counter=1
    oldindex=1
    timelim=timelim_sec*1000
    time0=clock()

    if activation is None:
        activation=Ek+np.random.exponential(Ekw)

    E0[0]=activation
    
    if activation == -1:
        dt=2*t_end
    else:
        if t_extra_in is None:
            dt=np.random.exponential(np.exp(activation/kBT))+1
        else:
            dt=t_extra_in[0]*frequency*np.exp(activation/kBT)/np.exp(activation/(kB*t_extra_in[1]))

    while True:
        time1=clock()
        if time1-time0>timelim:
            print("computation time above threshold")
            break

        newt=t+dt
        if newt > t_end:
            length[oldindex:]=glidel
            E0[oldindex:]=activation
            dt=newt-t_end
            break
            
        index=int(newt//sample_dt)+1
        if index>counter:
            length[oldindex:index]=glidel
            E0[oldindex:index]=activation
            counter+=1
            oldindex=index
                    
        t+=dt
        glidel+=1
        
        activation=Ek+np.random.exponential(Ekw)
        if fixed_prob>np.random.uniform():
            activation=-1
            dt=2*t_end

        else:
            dt=np.random.exponential(np.exp(activation/kBT))+1

    return length,E0,(dt/frequency,T)



#%% simulation
def simulation(simulation_parameters,printing=True,distribution="lognormal",initial_energy=None):
    
    N_lines=simulation_parameters["number_of_lines"]
    
    lines=dict()
    lines["parameters"]=simulation_parameters
    lines["parameters"]["distribution"]=distribution
    initial_energy=simulation_parameters["initial_energy"]
    
    old_energies=[initial_energy for i in range(N_lines)]
    t_extras=[None for i in range(N_lines)]

    for i in range(len(simulation_parameters["temperatures"])):
        
        T=simulation_parameters["temperatures"][i]+273.15
        duration=simulation_parameters["durations"][i]
        if printing:
            print(simulation_parameters["temperatures"][i])

        Nequisteps=int(duration//simulation_parameters["sampling_time_intervals"][i])+1
        
        istring=str(i)
        lines[istring]=dict()
        lines[istring]["temperature_K"]=T
        lines[istring]["temperature_C"]=simulation_parameters["temperatures"][i]
        lines[istring]["times"]=np.arange(Nequisteps,dtype=np.float64)*simulation_parameters["sampling_time_intervals"][i]
        lines[istring]["lengths"]=np.empty((N_lines,Nequisteps))
        lines[istring]["energies"]=np.empty((N_lines,Nequisteps))

        for j in range(N_lines):

            glide_length,E0,t_extra = single_line_exponential(
                simulation_parameters["location_parameter"],
                simulation_parameters["scale_parameter"],
                T,duration,
                simulation_parameters["frequency"],
                simulation_parameters["sampling_time_intervals"][i],
                activation=old_energies[j],
                t_extra_in=t_extras[j],
                timelim_sec=simulation_parameters["single_line_computation_time_threshold"],
                fixed_prob=simulation_parameters["fixed_probability"])
                #Emax=simulation_parameters["maximum_energy"])

            t_extras[j]=(t_extra[0],t_extra[1])
            old_energies[j]=E0[-1]

            lines[istring]["lengths"][j]=glide_length*simulation_parameters["length_conversion"]
            lines[istring]["energies"][j]=E0

    return lines
#%%

def counting(lines,dts,minv=0):

    keylist=[i for i in lines.keys() if i.isnumeric()]
    for c,i in enumerate(keylist): 
        speeds_abs=np.diff(lines[i]["lengths"],axis=-1)
        speeds=speeds_abs/dts[c]
        speeds_binary=speeds>minv
        lines[i]["velocities"]=speeds[speeds_binary]
        lines[i]["all_events"]=len(lines[i]["velocities"])
        lines[i]["event_dist"]=np.sum(speeds_binary,axis=0)
        lines[i]["length_dist"]=np.sum(speeds_abs,axis=0)
        lines[i]["total_length"]=np.cumsum(lines[i]["length_dist"])

#%% multiple sims
def multi_sim_save_step(lines,multi=None,
                       velocity_bins=None):
    if multi is None:
        multi=dict()
    if "cumulative_events" not in multi:
        multi["cumulative_events"]=[]
    if "cumulative_distance" not in multi:
        multi["cumulative_distance"]=[]
    if "velocity_hist" not in multi:
        multi["velocity_hist"]=[]
    if "energy_dist" not in multi:
        multi["energy_dist"]=[]
        
    keylist=[i for i in lines.keys() if i.isnumeric()]
    
    cumu_events=[]
    for c,i in enumerate(keylist):
        #print(i)
        cumu_events.append(
            np.cumsum(lines[i]["event_dist"])#/simulation_parameters["number_of_lines"]*30
            )
    multi["cumulative_events"].append(cumu_events)
    
    energies=[]
    for c,i in enumerate(keylist):
        energies.append(lines[i]["energies"])
    multi["energy_dist"].append(energies)
    
    cumu_dist=[]
    for c,i in enumerate(keylist):
        #print(i)
        cumu_dist.append(
            np.cumsum(lines[i]["length_dist"])#/simulation_parameters["number_of_lines"]*30
            )
    multi["cumulative_distance"].append(cumu_dist)


    if velocity_bins is None:
        velmax=0
        velmin=np.inf
        for c,i in enumerate(keylist):
            velmaxi=np.max(lines[i]["velocities"] *10**6)
            velmax=max(velmax,velmaxi)
            velmini=np.min(lines[i]["velocities"] *10**6)
            velmin=min(velmin,velmini)
        
        velocity_bins=np.linspace(velmin, velmax)
            
            
    velos=[]
    for c,i in enumerate(keylist):
        simhist,bins=np.histogram(lines[i]["velocities"] *10**6,
                                  velocity_bins)
        velos.append(simhist)
    multi["velocity_hist"].append(velos)
    
    return multi


#%%

def prepare_multisim_for_saving(multi_sim,lines,xbins,
                                timeoffset=0,timedelta=0):

    sim_data=dict()
    sim_data["parameters"]=lines["parameters"]
        
    eventcurves=[]
    distancecurves=[]
    velocitycurves=[]
    energycurves=[]
    newtimes=[]
    
    temperatures=lines["parameters"]["temperatures"]
    stis=lines["parameters"]["sampling_time_intervals"]
    durations=lines["parameters"]["durations"]
    number_of_lines=lines["parameters"]["number_of_lines"]
    number_of_linesets=lines["parameters"]["number_of_line_sets"]
    tempnum=len(temperatures)
    for i in range(tempnum):#
        Ntimesteps=int(durations[i]//stis[i])
        eventcurves.append(np.zeros([number_of_linesets,Ntimesteps]))
        distancecurves.append(np.zeros([number_of_linesets,Ntimesteps]))
        velocitycurves.append(np.zeros([number_of_linesets,len(xbins)-1]))
        energycurves.append(np.zeros([number_of_linesets*number_of_lines,Ntimesteps+1]))
        
        newtimes.append(np.arange(Ntimesteps)*stis[i]+timeoffset)
        timeoffset=np.max(newtimes[-1])+timedelta
    
    for i in range(tempnum):#
        for j in range(number_of_linesets):
            eventcurves[i][j]=multi_sim["cumulative_events"][j][i]
            distancecurves[i][j]=multi_sim["cumulative_distance"][j][i]
            velocitycurves[i][j]=multi_sim["velocity_hist"][j][i]
            for k in range(number_of_lines):
                energycurves[i][j*number_of_lines+k]=multi_sim["energy_dist"][j][i][k]
    
        valid=np.ma.array(energycurves[i],
                    mask=energycurves[i] == -1)
        
        sim_data[str(i)]=dict()
        sim_data[str(i)]["energies"]=dict()
        sim_data[str(i)]["distances"]=dict()
        sim_data[str(i)]["events"]=dict()
        sim_data[str(i)]["velocities"]=dict()
        sim_data[str(i)]["blocked"]=dict()
        
        sim_data[str(i)]["time"]=newtimes[i]
        
        nblock=np.sum(np.ma.getmask(valid),axis=0)
        sim_data[str(i)]["blocked"]["number_total"]=nblock
        sim_data[str(i)]["blocked"]["ratio"]=nblock/(number_of_lines*number_of_linesets)

        sim_data[str(i)]["energies"]["mean"]=valid.mean(axis=0)
        sim_data[str(i)]["energies"]["min"]=valid.min(axis=0)
        sim_data[str(i)]["energies"]["max"]=valid.max(axis=0)
        sim_data[str(i)]["energies"]["std"]=valid.std(axis=0)
        
        sim_data[str(i)]["velocities"]["mean"]=np.mean(velocitycurves[i],axis=0)
        sim_data[str(i)]["velocities"]["min"]=np.min(velocitycurves[i],axis=0)
        sim_data[str(i)]["velocities"]["max"]=np.max(velocitycurves[i],axis=0)
        sim_data[str(i)]["velocities"]["std"]=np.std(velocitycurves[i],axis=0)

        sim_data[str(i)]["events"]["mean"]=np.mean(eventcurves[i],axis=0)
        sim_data[str(i)]["events"]["min"]=np.min(eventcurves[i],axis=0)
        sim_data[str(i)]["events"]["max"]=np.max(eventcurves[i],axis=0)
        sim_data[str(i)]["events"]["std"]=np.std(eventcurves[i],axis=0)
        
        sim_data[str(i)]["distances"]["mean"]=np.mean(distancecurves[i],axis=0)
        sim_data[str(i)]["distances"]["min"]=np.min(distancecurves[i],axis=0)
        sim_data[str(i)]["distances"]["max"]=np.max(distancecurves[i],axis=0)
        sim_data[str(i)]["distances"]["std"]=np.std(distancecurves[i],axis=0)

    return sim_data
        
#%%

def save_h5(lines,hdf,keypath="/"):
    keypaths=[]
    linekeys=[]
    if "ignore_list" in lines:
        ignore_list=lines["ignore_list"]
    else:
        ignore_list=[]
        
    for key, value in lines.items():
        if key in ignore_list:
            #print(key)
            pass
        else:
            if isinstance(value, np.ndarray):
                hdf.create_dataset(keypath+"/"+key,
                                   data=value,
                                   dtype='float32')
            elif isinstance(value, dict):
                linekeys.append(key)
                if keypath=="/":
                    keypaths.append(keypath+key)
                else:
                    keypaths.append(keypath+"/"+key)
            elif value is None:
                hdf[keypath].attrs[key] = "None"
            else:
                hdf[keypath].attrs[key] = value

    for i in range(len(keypaths)):
        hdf.create_group(keypaths[i])
        save_h5(lines[linekeys[i]],hdf,keypath=keypaths[i])


def lines_to_h5(lines,name=None,pref=None):
    if name is None:
        if pref is None:
            name="simulation_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            name=pref+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name+=".h5"
    with h5py.File(name, 'w') as hdf:
        save_h5(lines, hdf)
#%%
def sim_loop(simulation_parameters,xbins,timeoffset,timedelta,pref,ignorelist=[]):
    multi_sim=None
    for knum in range(simulation_parameters["number_of_line_sets"]):
        #t1=time.time()
        lines=simulation(simulation_parameters,printing=False,distribution="exponential")    
        #t2=time.time()
        #print(t2-t1)
        
        counting(lines,simulation_parameters["sampling_time_intervals"],minv=2*10**-7)
        multi_sim=multi_sim_save_step(lines,multi_sim,xbins)
        
        #print(knum)
    
    #print(i)
    #print(len(sps))
    sim_data=prepare_multisim_for_saving(multi_sim,lines,xbins,
                                    timeoffset=timeoffset,timedelta=timedelta)
    
    sim_data["ignore_list"]=ignorelist#["0","1","2","3","4","5","6","7","8","9"]
    lines_to_h5(sim_data,pref=pref)#"multilines.h5"
    return sim_data




#%% deprecated
import json
def _arrays_to_lists(indic):
    for k, v in indic.items():
        if isinstance(v, np.ndarray):
            indic[k]=v.tolist()
        elif isinstance(v, dict):
            _arrays_to_lists(v)
        elif isinstance(v, list):
            print("list encountered")
        elif isinstance(v, np.int32):
            print(k)
    return indic

def _save_sim(filename,lines):
    with open(filename+".json","w") as f:
        json.dump(lines,f)