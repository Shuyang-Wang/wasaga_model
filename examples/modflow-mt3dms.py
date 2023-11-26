import numpy as np
import flopy

# Assign name and create modflow model object
modelname = 'mf-mt'
mf = flopy.modflow.Modflow(modelname, exe_name='mf2005')

# Model domain and grid definition
Lx = 1000.
Ly = 1000.
ztop = 0.
zbot = -50.
nlay = 1
nrow = 50
ncol = 50
delr = Lx/ncol
delc = Ly/nrow
delv = (ztop - zbot) / nlay
botm = np.linspace(ztop, zbot, nlay + 1)

# Create the discretization object
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,top=ztop, botm=botm[1:])

# Variables for the BAS package
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
ibound[:, :, 0] = -1
ibound[:, :, -1] = -1
strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
strt[:, :, 0] = 10.
strt[:, :, -1] = 0.
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

# Add LPF package to the MODFLOW model
lpf = flopy.modflow.ModflowLpf(mf, hk=10., vka=10., ipakcb=53)

# Add OC package to the MODFLOW model
spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}
oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)

# well package
# Remember to use zero-based layer, row, column indices!
pumping_rate = 1000.
wcol = round(nrow/10) # x index for the well
wrow = round(ncol/2)  # y index for the well
wel_sp = [[0, wrow, wcol, pumping_rate]] # lay, row, col index, pumping rate
stress_period_data = {0: wel_sp} # define well stress period {period, well info dictionary}
wel = flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)

# PCG package for matrix computation
pcg = flopy.modflow.ModflowPcg(mf)

# linkage to mt3dms LMT package
lmt = flopy.modflow.ModflowLmt(mf, output_file_name='mt3d_link.ftl')

# Write the MODFLOW model input files
mf.write_input()

# Run the MODFLOW model
success, buff = mf.run_model()

##################################################################################
##################################################################################


# create mt3dms model object
mt = flopy.mt3d.Mt3dms(modflowmodel=mf, modelname=modelname, exe_name='./mt3dms5b.exe' ,ftlfilename='mt3d_link.ftl')

# basic transport package
btn = flopy.mt3d.Mt3dBtn(mt, prsity=0.3, icbund = 1, sconc=0.0, ncomp=1, perlen = 1000, nper=1, nstp = 50, tsmult = 1.0, nprs = -1, nprobs = 10, cinact = -1, chkmas=True)

# advaction package
adv = flopy.mt3d.Mt3dAdv(mt, mixelm=-1, percel=0.75)
# dispersion package
dsp = flopy.mt3d.Mt3dDsp(mt, al=10.0, trpt=0.1, trpv=0.1, dmcoef=1e-09)

# source/sink package
ssm_data = {}
itype = flopy.mt3d.Mt3dSsm.itype_dict()
ssm_data[0] = [(0, wrow, wcol, 10.0, itype['WEL'])]

ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=ssm_data)

# matrix solver package
gcg = flopy.mt3d.Mt3dGcg(mt, cclose=1e-6)

# write mt3dms input
mt.write_input()
# run mt3dms
mt.run_model()






# post-processing and plotting
# plot flow
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, aspect='equal')

wpt = ((wcol+0.5)*delr, Lx - ((wrow + 0.5)*delc)) # origin at low upper..
hds = bf.HeadFile(modelname+'.hds')
times = hds.get_times() # simulation time, steady state
head = hds.get_data(totim=times[-1])

cbb = bf.CellBudgetFile(modelname+'.cbc') # read budget file
frf = cbb.get_data(text='FLOW RIGHT FACE', totim=times[-1])[0]
fff = cbb.get_data(text='FLOW FRONT FACE', totim=times[-1])[0]

# create flopy plot object, plot grid and contour
modelmap = flopy.plot.ModelMap(model=mf, layer=0)
lc = modelmap.plot_grid() # grid
cs = modelmap.contour_array(head, levels=np.linspace(head.min(), head.max(), 21)) # head contour
plt.clabel(cs, fontsize=20, fmt='%1.1f', zorder=1) # contour label
quiver = modelmap.plot_discharge(frf, fff, head=head) # quiver
plt.plot(wpt[0],wpt[1],'ro') # well location
plt.show()

# plot conc
ucnobj = bf.UcnFile('MT3D001.UCN')
#print(ucnobj.list_records()) # get values

times = ucnobj.get_times() # simulation time
times1 = times[round(len(times)/4.)] # 1/4 simulation time
times2 = times[round(len(times)/2.)] # 1/2 simulation time
times3 = times[-1] # the last simulation time

conc1 = ucnobj.get_data(totim=times1)
conc2 = ucnobj.get_data(totim=times2)
conc3 = ucnobj.get_data(totim=times3)


# conc 100 day
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(model=mf, layer=0)
lc = modelmap.plot_grid() # grid
cs = modelmap.plot_array(conc1) # head contour
plt.colorbar(cs) # colrobar
plt.plot(wpt[0],wpt[1],'ro')
plt.title('C  %g day' % times1)
plt.show()

# conc in 500 days
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(model=mf, layer=0)
lc = modelmap.plot_grid() # grid
cs = modelmap.plot_array(conc2) # head contour
plt.colorbar(cs) # colrobar
plt.plot(wpt[0],wpt[1],'ro')
plt.title('C  %g day' % times2)
plt.show()


# conc in 1000 days
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(model=mf, layer=0)
lc = modelmap.plot_grid() # grid
cs = modelmap.plot_array(conc3) # head contour
plt.colorbar(cs) # colrobar
plt.plot(wpt[0],wpt[1],'ro')
plt.title('C  %g day' % times3)
plt.show()
