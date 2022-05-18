&parallelization
n_procs_s = 0
n_procs_v = 0
n_procs_w = 0
n_procs_x = 0
n_procs_y = 0
n_procs_z = 0
n_procs_sim = 0
n_parallel_sims = 1
/

&box
n_spec = 2
nx0 = 16 ! rad grid pnts; pow 2; upto 128 for NL
nz0 = 64 ! grid pnts in z-dir; even, typ 16-32
nv0 = 32 ! v_para grid pnts; even; typ 32-64
nw0 = 8  ! mu grid pnts; typ 8-16

nky0 = 1 ! fourier modes in y dir.
kymin = {{ kymin|default('0.05 !scanrange: 0.05,0.05,0.8', true) }}

lv = 3.00
lw = 9.00
/

&in_out
diagdir = './'
read_checkpoint = .F.
write_checkpoint = .F.
istep_nrg = {{ istep_nrg|default('10', true) }}
istep_field = {{ istep_field|default('100', true) }}
istep_omega = {{ istep_omega|default('100', true) }}
istep_energy = {{ istep_energy|default('500', true) }}
istep_mom = {{ istep_mom|default('0', true) }}
istep_vsp = {{ istep_vsp|default('0', true) }}
/

&general
! computation type
nonlinear = .F.
comp_type = 'IV'
perf_tsteps = 20
arakawa_zv_order = 4
bpar = .F.
delzonal = .F.
delzonal_fields = .F.
! initial value computation
ntimesteps = {{ ntimesteps|default('1E5', true) }} ! max timesteps
timelim = {{ timelim|default('43E3', true) }} ! max walltime
simtimelim = {{ simtimelim|default('5E2', true) }} ! max simulation time in Lref/cref
omega_prec = {{ omega_prec|default('1.E-3', true) }}
calc_dt = .T.
! initialization
init_cond = 'alm'
! species-independent physical parameters
beta = {{ beta|default('-1', true) }}
coll = {{ coll|default('-1', true) }}
collision_op = {{ collision_op|default("'landau'", true) }}
coll_cons_model = 'default'
coll_f_fm_on = .F.
! hyper-diffusion
hyp_z = 0.25
hyp_v = 0.20
hypz_opt = .F.
! eigenvalue computation
n_ev = 4
/

&nonlocal_x
/

&external_contr
/

&geometry
magn_geometry = {{ magn_geometry|default("'miller'", true) }}
! parameters for `miller` and `s_alpha`
trpeps = {{ trpeps|default('0.18', true) }}
q0 = {{ q0|default('1.4', true) }}
shat = {{ shat|default('0.8', true) }}
major_R = {{ major_R|default('1.', true) }}
amhd = {{ amhd|default('0.0', true) }}
! parameters for `miller`
minor_r = {{ minor_r|default('1.', true) }}
kappa = {{ kappa|default('1.0', true) }}
s_kappa = {{ s_kappa|default('0.0', true) }}
delta = {{ delta|default('0.0', true) }}
s_delta = {{ s_delta|default('0.0', true) }}
zeta = {{ zeta|default('0.0', true) }}
s_zeta = {{ s_zeta|default('0.0', true) }}
drR = {{ drR|default('0.0', true) }}
! equilibrium pressure gradient
dpdx_pm = {{ dpdx_pm|default('-2', true) }}
dpdx_term = {{ dpdx_term|default("'full_drift'", true) }}
/

&species
name = 'ions'
mass = 1.0
charge = 1

temp = {{ ion_temp|default('1.', true) }}
omt = {{ ion_omt|default('6.92', true) }}
dens = {{ ion_dens|default('1.0', true) }}
omn = {{ ion_omn|default('2.22', true) }}
/

&species
name = 'electrons'
mass = {{ ele_mass|default('1.E-2', true) }}
charge = -1

temp = {{ ele_temp|default('1.', true) }}
omt = {{ ele_omt|default('6.92', true) }}
dens = {{ ele_dens|default('1.0', true) }}
omn = {{ ele_omn|default('2.22', true) }}
/

&units
Tref = {{ Tref|default('0.0', true) }}
nref = {{ nref|default('0.0', true) }}
Bref = {{ Bref|default('0.0', true) }}
Lref = {{ Lref|default('0.0', true) }}
mref = {{ mref|default('2.0', true) }}
/

&scan
/

