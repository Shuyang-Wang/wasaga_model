

# %%
# ## Getting Started
from pathlib import Path
from tempfile import TemporaryDirectory
from tempfile import mkdtemp

import numpy as np
import matplotlib.pyplot as plt
import flopy
import pandas as pd
import shutil
import pickle
import subprocess
import os
# %% [markdown]

# %%

def run_wasaga(my_params):
    # ## Import

    mytop =np.loadtxt('_imports/top')
    mybot = np.loadtxt('_imports/bot')
    myibound = np.loadtxt('_imports/ibound')
    mystrt = np.loadtxt('_imports/strt')
    myicbund = np.loadtxt('_imports/icbund')

    mybot.min() -mytop.max()

    Lx = 150
    Lz = 10
    nlay = 74
    nrow = 1
    ncol = 110
    delr = np.concatenate([np.repeat(2,25),np.repeat(1,70),np.repeat(2,15)])
    delc = 1.0
    delv = mytop[:,0] - mybot[:,0]
 

    # ## Workspace

    
    import os
    from datetime import datetime


    myt = datetime.now().strftime('%m%d%H%M%S')
    name = 'wasaga'
    #name =my_params['name']
    temp_dir = mkdtemp(prefix='_T{}_{}_'.format(myt,name))
    workspace = temp_dir
    os.makedirs(Path(workspace) / '_output')
    swt = flopy.seawat.Seawat(name, exe_name="swtv4", model_ws=workspace)

    
   ##

    import os
    import platform
    import subprocess

    def open_file(path):
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    open_file(workspace)


    
    # %%
    def create_time_table():
        time_table = pd.DataFrame()

        perlen = pd.Series(my_params['perlen'])
        perlen.index = perlen.index+1

        time_table['perlen'] = perlen
        time_table['Time'] = time_table['perlen'].cumsum()

        #time_table = pd.DataFrame()
        for key,value in wel_data.items():
            key
            time_table.loc[key,'wel_data'] = value[0][3]

        #time_table = pd.DataFrame()
        for key,value in ssm_data.items():
            key
            time_table.loc[key,'ssm'] = value[0][3]
        return time_table


    # ## DIS

    # %%
    ipakcb = 53

    # %%

    perlen = my_params['perlen']
    nper = len(perlen)
    nstp = np.repeat(1,nper)
    steady = np.append(np.array([True]),np.repeat(False,nper-1))



    # %%
    dis = flopy.modflow.ModflowDis(
        swt,
        nlay,
        nrow,
        ncol,
        nper=nper,
        delr=delr,
        delc=delc,
        laycbd=0,
        top=182.2,
        botm=mybot.reshape([74,1,110]),
        perlen=perlen,
        nstp=nstp, #Number of time steps in each stress period
        steady = steady
    )


    bas = flopy.modflow.ModflowBas(swt, myibound.reshape([74,1,110]), mystrt.reshape([74,1,110]))


    lpf = flopy.modflow.ModflowLpf(swt, hk=my_params['hk'], vka=my_params['vk'],
                                ss= my_params['ss'],sy=my_params['sy'],
                                ipakcb=ipakcb,laytyp=1)

    pcg = flopy.modflow.ModflowPcg(swt, hclose=1e-4)



    stress_period_data = {}
    for kper in range(nper):
        for kstp in range(nstp[kper]):
            stress_period_data[(kper, kstp)] = [
                "save head",
                "save drawdown",
                "save budget",
                "print head",
                "print budget",
            ]

    # OC - Output Control Option
    oc = flopy.modflow.ModflowOc(
        swt,
        stress_period_data= stress_period_data,
        compact=True,
    )

    # ## Well

    wel_data = my_params['wel_data']
    wel = flopy.modflow.ModflowWel(swt, stress_period_data=wel_data, ipakcb=ipakcb)

    # ModflowRch

    rch = flopy.modflow.ModflowRch(model = swt, rech = my_params['rech'])

    # ## SSM
    ssm_data = my_params['ssm_data']


    # ## BTN

    #timprs = my_params['perlen'][:5]*5 +np.arange(365,np.sum(my_params['perlen']),5*355).tolist()
    timprs = np.cumsum(my_params['perlen'])
    timprs = [1]




    btn = flopy.mt3d.Mt3dBtn(
        swt,
        nprs=len(timprs),
        timprs=timprs,
        prsity=my_params['porosity'],
        sconc=my_params['sconc'], # starting concentration
        ifmtcn=0,
        chkmas=False,
        nprobs=10,
        nprmas=10,
        dt0=my_params['dt0'],  # The user-specified initial transport step size
        ttsmult=1.1,
        mxstrn = 50000,
        icbund = myicbund.reshape([74,1,110])
    )
    adv = flopy.mt3d.Mt3dAdv(swt, mixelm=my_params['mixelm'],percel=0.4,mxpart=200000,nadvfd=0)
    dsp = flopy.mt3d.Mt3dDsp(swt, al=0.2, trpt=my_params['trpt'], 
                            trpv=my_params['trpv'], 
                            dmcoef=my_params['dmcoef'])
    gcg = flopy.mt3d.Mt3dGcg(swt, iter1=50, mxiter=1, isolve=2, cclose=my_params['cclose'])
    ssm = flopy.mt3d.Mt3dSsm(swt, stress_period_data=ssm_data)

    # %% [markdown]
    # ## RCT

    # %%
    rct = flopy.mt3d.Mt3dRct(model=swt, isothm=my_params['isothm'],sp1=my_params['sp1'],sp2=my_params['sp2'],igetsc=0,rhob= 1.65E+09)





    # %%
    #Pht3d
    #phc = flopy.mt3d.Mt3dPhc(swt,mine=[1],ie=[1],surf=[1],mobkin=[1],minkin=[1],surfkin=[1],imobkin=[1])
    #phc.write_file()

    # %% [markdown]
    # # Write File

    # %%
    swt.write_input()

    # %%


    # %% [markdown]
    # ## ################   Run      ########################

    # %%

    success, buff = swt.run_model(silent=False, report=True)
    assert success, "SEAWAT did not terminate normally."

    # %%  dump
    swt.my_params = my_params
    swt.delv =  delv
    swt.delr = delr
    pickle.dump( swt, open( Path(workspace) / '_output' / 'model.pickle',"wb" ) )

    # %%


    # %% [markdown]
    # # Summarize Parameters

    # %%

    def plot_my_params():

        time_table = create_time_table()

        fig,ax = plt.subplots(figsize=(8,5))

        xs = time_table['Time']
        ax.scatter(xs,np.repeat(10,len(xs)),s=200,marker='|')
        ax.annotate('Stress Period:   {}'.format([n for n in time_table['perlen'].dropna().unique()]),(0,10.2))


        mask = time_table['wel_data'].isna()
        xs = time_table[~mask]['Time']
        ax.scatter(xs,np.repeat(9,len(xs)),s=200,marker='|',c=time_table[~mask]['wel_data'],cmap='coolwarm_r')
        ax.annotate('Well Input:   {}'.format([n for n in time_table['wel_data'].dropna().unique()]),(0,9.2))


        mask = time_table['ssm'].isna()
        xs = time_table[~mask]['Time']
        ax.scatter(xs,np.repeat(8,len(xs)),s=200,marker='|',c=time_table[~mask]['ssm'],cmap='coolwarm_r')
        ax.annotate('Input Concentration:   {}'.format([n for n in time_table['ssm'].dropna().unique()]),(0,8.2))


        ax.annotate('Aquifer:',(11000,9.2),weight="bold")
        ax.annotate(('hk={0:.3g} '.format(my_params['hk']) + 'vk={0:.3g}'.format(my_params['vk'])),(11000,9))
        ax.annotate('ss= {} sy={}'.format(my_params['ss'],my_params['sy']),(11000,8.8))
        ax.annotate('porosity={}'.format(my_params['porosity']),(11000,8.6))

        ax.annotate('Dispersion:',(16400,9.2),weight="bold")
        ax.annotate('la={}, trpt={}, trpv={}'.format(my_params['la'],my_params['trpt'],my_params['trpv']),(16400,9))


        ax.annotate('Advection:',(16400,8.7),weight="bold")
        d = {0:'FDM',3:'HMOC'}
        ax.annotate('Solution: {}'.format(d[my_params['mixelm']]),(16400,8.5))


        ax.annotate('Sorption:',(11000,8.3),weight="bold")
        d = {0:'No Sorption',1:'Linear',3:'Langmuir'}
        ax.annotate('{},'.format(d[my_params['isothm']]),(11000,8.1))
        if my_params['isothm']==1:
            ax.annotate('sp1={}'.format(my_params['sp1']),(11000,7.9))
            
        elif my_params['isothm']==3:
            ax.annotate('sp1={}'.format(my_params['sp1']),(11000,7.9))
            ax.annotate('sp2={}'.format(my_params['sp2']),(11000,7.7))
            
    

        ax.annotate('Recharge:',(16400,8.3),weight="bold")
        #d = {0:'No Sorption',1:'Linear'}
        ax.annotate('{0:.3g}'.format(my_params['rech']),(16000,8.1))


        ax.annotate('{}'.format(os.path.basename(os.path.normpath(workspace))),(200,7.3),weight="bold")

        ax.axes.get_yaxis().set_visible(False)
        plt.ylim(7,10.5)
        plt.xlim(-1000,65*365)
        plt.tight_layout()
        plt.savefig(Path(workspace) /'_output/_my_params.png')


    ## Run
    plot_my_params()


    # %% [markdown]
    # ## JSON

    # %%
    import json

    with open(Path(workspace) /'_output/my_params.json', 'w') as fp:
        json.dump(my_params,fp)

    # %%


    # %% [markdown]
    # ## Time_Table



    # %%
    create_time_table().to_csv(Path(workspace) /'_output/time_table.csv')   
    shutil.copytree('_imports',Path(workspace)/'_output/_imports',dirs_exist_ok =True)
    shutil.copyfile(__file__,Path(workspace) /'_output/{}'.format(os.path.basename(__file__)))


    return workspace
    #############################


# %%


# %%


if (__name__ == '__main__'):
    ################### Model Parameters  #########################
    my_params = {}
    my_params['name'] = 'test_wasaga'

    ## mt3dGcg
    cclose=my_params['cclose'] = 1e-5
    ## RCT

    my_params['hk'] = 6.9
    my_params['vk'] = 0.69

    my_params['porosity'] = 0.35
    my_params['sy'] = 0.33
    my_params['ss'] = 0.0002
    my_params['la'] = 0.2
    my_params['trpt'] = 0.1
    my_params['trpv'] = 0.1
    my_params['dmcoef'] = 3.7E-10 ####
    my_params['sconc'] = 20

    #Species ID
    sp_ID = {}
    #sp_ID['Cl-'] = 1
    sp_ID['SRP'] = 1
    my_params['sp_ID'] = sp_ID

    #Species Name
    sp_name = {}
    #sp_name[1] = 'Cl-'
    sp_name[1] = 'SRP'
    my_params['sp_name'] = sp_name


    ## P lens
    my_params['perlen'] = [1]+[30,92,31,212]*25 +[365]*5+[365*5]*5+[365*4]+[185,30,30,120]


    ## Wel data
    wel_data = {}
    #last_wel_id = len(my_params['perlen'])-1
    for y in range(25):
        p = (y)*4 +2
        wel_data[p] = [[50,0,15,0.5]]
        p += 1
        wel_data[p] = [[50,0,15,0.5]]
        p += 1
        wel_data[p] = [[50,0,15,0.5]]
        p += 1
        wel_data[p] = [[50,0,15,0.5]]
        p += 1

    wel_data[p] = [[50,0,15,0]]
    my_params['wel_data'] = wel_data


    # Mt3dSsm 
    ssm_data = {}
    itype = flopy.mt3d.Mt3dSsm.itype_dict()

    for y in range(25):
        p = (y)*4 +2
        ssm_data[p] = [(50, 0, 15, 3000.0, itype['WEL'])]
        p += 1
        ssm_data[p] = [(50, 0, 15, 3000.0, itype['WEL'])]
        p += 1
        ssm_data[p] = [(50, 0, 15, 3000.0, itype['WEL'])]
        p += 1
        ssm_data[p] = [(50, 0, 15, 3000.0, itype['WEL'])]
        p += 1
    ssm_data[p] = [(50, 0, 15, 0.0, itype['WEL'])]
    my_params['ssm_data'] = ssm_data



    #Mt3dBtn
    my_params['mixelm'] = 3 # HMOC
    my_params['isothm'] = 1 #Linear Sorption
    my_params['sp1'] = 0.012 
    my_params['sp2'] = 0.012 

    # Rch
    my_params['rech'] = 0.5/365 ## 500mm/year
    

    # dt0
    my_params['dt0'] = 0.1
    ##########################################



def execute_wasaga_draw(ws):
    # Define the path to the script and its folder
    script_path = Path(ws) / "wasaga_draw.py"
    #script_path = os.path.join("workspace", "wasaga_draw.py")
    script_folder = os.path.dirname(script_path)

    # Save the current working directory
    original_dir = os.getcwd()

    try:
        # Change the working directory to the script's folder
        os.chdir(script_folder)

        # Define the command to execute the script
        command = ["python", script_path]

        # Execute the command and capture the output
        output = subprocess.check_output(command, universal_newlines=True)
        print("Script executed successfully.")
        print("Output:")
        print(output)
    except subprocess.CalledProcessError as e:
        print("Error executing the script:")
        print(e)
    finally:
        # Change the working directory back to the original
        os.chdir(original_dir)


# %%
if (__name__ == '__main__'):

    
    run_wasaga(my_params)