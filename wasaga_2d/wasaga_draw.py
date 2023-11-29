# %%
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import matplotlib.pyplot as plt
import flopy
import pandas as pd
import shutil
import os

import flopy.utils.binaryfile as bf

# %%
topo_mask = np.loadtxt(Path('_output/_imports/ibound'))
topt_mask=topo_mask[topo_mask!=0]=1
topt_mask=topo_mask[topo_mask==0]=-1
# %%

delv = np.array([0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ,
       0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ,
       0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.25, 0.25, 0.25,
       0.25, 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ,
       0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ,
       0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ,
       0.1 , 0.1 , 0.1 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 ])

delr = np.concatenate([np.repeat(2,25),np.repeat(1,70),np.repeat(2,15)])

#--------------------

def draw_all_hds(workspace,name,model):
    headobj = bf.HeadFile(Path(workspace) / f"{name}.hds")

    times = headobj.get_times()
    mydir = Path(workspace) /'_output/hd/'
    os.makedirs(mydir,exist_ok=True)

    fig,ax = plt.subplots()

    p = 0 
    ax.set_aspect("auto")
    pmv = flopy.plot.PlotCrossSection(model=model, ax=ax, line={"row": 0},extent=[0, 150, 172.2, 182.2])
    topo = pmv.plot_array(topo_mask, cmap='Greys_r',masked_values=[-1],alpha=0.2)

    for p in range(len(times)):
        
        head = headobj.get_data(totim=times[p])
        arr = pmv.plot_array(head, cmap='winter')

        title = ax.set_title(" Simulated Heads a t: {} days".format(times[p]))
        cb = fig.colorbar(arr,ax=ax,location='bottom')
        contours = pmv.contour_array(head, colors="white",levels=np.arange(175.8,177.6,0.2))
        ax.clabel(contours, fmt="%2.2f")

        fig.savefig(Path(mydir) /"hd_{0:0=3}.png".format(p))
        
        cb.remove()
        arr.remove()
        contours.remove()
    plt.close()

    





# %%------------------------------

def draw_all_sp1a(workspace,name,model):
    ucnobj = bf.UcnFile(Path(workspace) / "MT3D001.UCN", model=model)
    times = ucnobj.get_times()

    d =  ucnobj.get_alldata().copy()
    vmax = d[d!=1E30].max()
    vmin = d[d!=1E30].min()

    mydir = Path(workspace) /'_output/sp{}/'.format(1)
    os.makedirs(mydir,exist_ok=True)
    fig,ax = plt.subplots()


    p = 0 
    #concentration = ucnobj.get_data(totim=times[p])
    ax.set_aspect("auto")
    pmv = flopy.plot.PlotCrossSection(model=model, ax=ax, line={"row": 0},extent=[0, 150, 172.2, 182.2])
    topo = pmv.plot_array(topo_mask, cmap='Greys_r',masked_values=[-1],alpha=0.2)
    plt.scatter(90,176.3,s=200,c='dimgrey',marker='|',zorder=10)
    plt.scatter(50,177.2,s=200,c='dimgrey',marker='|',zorder=10)
    plt.scatter(70,176.7,s=200,c='dimgrey',marker='|',zorder=10)


    for p in range(len(times)):
        
        concentration = ucnobj.get_data(totim=times[p])
        arr = pmv.plot_array(concentration,masked_values=[1E30],cmap='coolwarm',vmax=vmax,vmin=vmin)

        title = ax.set_title("{} at: {} days. dT={}".format( 'SRP',times[p],times[p]-times[p-1]))
        cb = fig.colorbar(arr,ax=ax,location='bottom')
        fig.savefig(Path(mydir) /"sp1_{0:0=3}.png".format(p))

        
        cb.remove()
        arr.remove()
    plt.close()

        




def draw_all_sp1b(workspace,name,model):
    ucnobj = bf.UcnFile(Path(workspace) / "MT3D001.UCN", model=model)
    times = ucnobj.get_times()

    d =  ucnobj.get_alldata().copy()
    vmax = d[d!=1E30].max()
    vmin = d[d!=1E30].min()

    mydir = Path(workspace) /'_output/sp{}b/'.format(1)
    os.makedirs(mydir,exist_ok=True)
    fig,ax = plt.subplots()


    p = 0 
    #concentration = ucnobj.get_data(totim=times[p])
    ax.set_aspect("auto")
    pmv = flopy.plot.PlotCrossSection(model=model, ax=ax, line={"row": 0},extent=[0, 150, 172.2, 182.2])
    topo = pmv.plot_array(topo_mask, cmap='Greys_r',masked_values=[-1],alpha=0.2)
    plt.scatter(90,176.3,s=200,c='dimgrey',marker='|',zorder=10)
    plt.scatter(50,177.2,s=200,c='dimgrey',marker='|',zorder=10)
    plt.scatter(70,176.7,s=200,c='dimgrey',marker='|',zorder=10)


    for p in range(len(times)):
        
        concentration = ucnobj.get_data(totim=times[p])
        arr = pmv.plot_array(concentration,masked_values=[1E30],cmap='coolwarm',vmax=500,vmin=vmin)

        title = ax.set_title("{} at: {} days. dT={}".format( 'SRP',times[p],times[p]-times[p-1]))
        cb = fig.colorbar(arr,ax=ax,location='bottom')
        fig.savefig(Path(mydir) /"sp1_{0:0=3}.png".format(p))

        
        cb.remove()
        arr.remove()
    plt.close()
    

def draw_all_sp1c(workspace,name,model):
    ucnobj = bf.UcnFile(Path(workspace) / "MT3D001.UCN", model=model)
    times = ucnobj.get_times()

    d =  ucnobj.get_alldata().copy()


    mydir = Path(workspace) /'_output/sp{}c/'.format(1)
    os.makedirs(mydir,exist_ok=True)
    fig,ax = plt.subplots()


    p = 0 
    #concentration = ucnobj.get_data(totim=times[p])
    ax.set_aspect("auto")
    pmv = flopy.plot.PlotCrossSection(model=model, ax=ax, line={"row": 0},extent=[0, 150, 172.2, 182.2])
    topo = pmv.plot_array(topo_mask, cmap='Greys_r',masked_values=[-1],alpha=0.2)
    plt.scatter(90,176.3,s=200,c='dimgrey',marker='|',zorder=10)
    plt.scatter(50,177.2,s=200,c='dimgrey',marker='|',zorder=10)
    plt.scatter(70,176.7,s=200,c='dimgrey',marker='|',zorder=10)


    for p in range(len(times)):
        
        concentration = ucnobj.get_data(totim=times[p])
        arr = pmv.plot_array(concentration,masked_values=[1E30],cmap='coolwarm')

        title = ax.set_title("{} at: {} days. dT={}".format( 'SRP',times[p],times[p]-times[p-1]))
        cb = fig.colorbar(arr,ax=ax,location='bottom')
        fig.savefig(Path(mydir) /"sp1_{0:0=3}.png".format(p))

        
        cb.remove()
        arr.remove()
    plt.close()


# %%------------------------------


# %%
def plot_distance_mass(workspace,name,model):


    ucnobj = bf.UcnFile(Path(workspace) / "MT3D001.UCN", model=model)
    times = ucnobj.get_times()

    for per in [-1,-2,-3]:
        concentration = ucnobj.get_data(totim=times[per])
        concentration[concentration==1e30]=0

        distances = [45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120]
        distances = np.arange(30,140,5)
        find_closest_value = lambda target_value: min(delr.cumsum(), key=lambda x: abs(target_value- x))
        distances  = np.unique([find_closest_value(n) for n in distances])

        #distances = [40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135]
        #distances = np.arange(40,140,5)
        vsums = []
        for dis in distances:
            col = delr.cumsum().tolist().index(dis)
            vsum = (concentration[:,0,col]*delv).sum()*60
            vsums.append(vsum)


        plt.plot(np.array(distances)-30,vsums,marker='o',alpha=0.8,label='{} days'.format(times[per]))
    plt.ylabel('SRP Mass')
    plt.xlabel('Distance from Septic Bed (m)')
    plt.ylim(0)
    plt.xlim(0)
    plt.axvline(x=20,c='k',linestyle='--')
    plt.axvline(x=40,c='k',linestyle='--')
    plt.axvline(x=60,c='k',linestyle='--')

    sp1 = model.my_params['sp1']
    k = model.my_params['hk']
    la = model.my_params['la']
    plt.title('hk:{:.2g}'.format(k))

    plt.legend()
    plt.ylim(0,25000)
    plt.gcf().set_size_inches(4,3)
    plt.tight_layout()

    plt.savefig(Path(workspace) /'_output/SRP_mass.png')
    plt.close()

    

# %%------------------------------

# %%
def plot_distance_conc(workspace,name,model):

    ucnobj = bf.UcnFile(Path(workspace) / "MT3D001.UCN", model=model)
    times = ucnobj.get_times()

    for per in [-1,-2,-3]:
        concentration = ucnobj.get_data(totim=times[per])
        concentration[concentration==1e30]=0

        distances = [50,70,90]
        vsums = []
        for dis in distances:
            col = delr.cumsum().tolist().index(dis)
            mask = concentration[:,0,col]>0
            v_height = (delv[mask]).sum()
            vsum = (concentration[:,0,col]*delv).sum()
            conc = vsum/v_height
            vsums.append(conc)


        plt.plot(np.array(distances)-30,vsums,marker='o',alpha=0.8,label='{} days'.format(times[per]))
    plt.ylabel('SRP Conc')
    plt.xlabel('Distance to Tile (m)')
    plt.ylim(0)
    plt.xlim(0)

    plt.legend()

    plt.gcf().set_size_inches(4,3)
    plt.tight_layout()
    plt.savefig(Path(workspace) /'_output/SRP_Conc.png')








if (__name__ == '__main__'):
    import pickle
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    workspace= '.'
    name = [f for f in files if f.endswith('.nam')][0].split('.')[0]
    print(name)
    model = pickle.load( open( Path('_output') / 'model.pickle', "rb"  ))


    #draw_all_sp1a(workspace,name,model)
    
    plot_distance_mass(workspace,name,model)
    plot_distance_conc(workspace,name,model)
    #draw_all_sp1b(workspace,name,model)
    #draw_all_sp1c(workspace,name,model)
    #draw_all_hds(workspace,name,model)
# %%------------------------------


# %%


# %%



