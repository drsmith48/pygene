&parallelization
n_parallel_sims = {{ n_parallel_sims|default('1', true) }}
n_procs_s = {{ n_procs_s|default('2', true) }}
n_procs_v = {{ n_procs_v|default('0', true) }}
n_procs_w = {{ n_procs_w|default('0', true) }}
n_procs_x = {{ n_procs_x|default('0', true) }}
n_procs_y = {{ n_procs_y|default('1', true) }}
n_procs_z = {{ n_procs_z|default('0', true) }}
/

&box
! rad grid pnts; pow 2; 64-128 for NL
! fourier modes in y dir.; 16-64 for NL
! grid pnts in z-dir; even, typ 16-32
! v_para grid pnts; even; typ 32-64
! mu grid pnts; typ 8-16
n_spec = 2
nx0 = {{ nx0|default('16', true) }}
nky0 = {{ nky0|default('1', true) }}
nz0 = {{ nz0|default('64', true) }}
nv0 = {{ nv0|default('32', true) }}
nw0 = {{ nw0|default('8', true) }}

kymin = {{ kymin|default('0.03 !scanlist: 0.03,0.1,0.3,1.0,3.0', true) }}
lv = {{ lv|default('3.0', true) }}
lw = {{ lw|default('9.0', true) }}
lx = {{ lx|default('0', true) }}
/

&in_out
diagdir = '{{ diagdir|default("./", true) }}'
read_checkpoint = F
write_checkpoint = T
istep_nrg = {{ istep_nrg|default('50', true) }}
istep_energy = {{ istep_energy|default('50', true) }}
istep_omega = {{ istep_omega|default('100', true) }}
istep_field = {{ istep_field|default('2500', true) }}
istep_mom = {{ istep_mom|default('2500', true) }}
istep_vsp = {{ istep_vsp|default('10000', true) }}
/

&general
! computation type
nonlinear = {{ nonlinear|default('F', true) }}
comp_type = "{{ comp_type|default('IV', true) }}"
perf_vec = {{ perf_vec|default('0 0 0 0 0 0 0 0 0', true) }}
arakawa_zv_order = 4
bpar = {{ bpar|default("F", true) }}
delzonal = F
delzonal_fields = F
! initial value computation
timelim = {{ timelim|default('86E3', true) }} ! max walltime
simtimelim = {{ simtimelim|default('4E2', true) }} ! max simulation time in Lref/cref
omega_prec = {{ omega_prec|default('1.E-3', true) }}
ntimesteps = 1000000
calc_dt = T
! initialization
init_cond = 'alm'
! species-independent physical parameters
! calculated beta = {{ beta_calc|default('', true) }}
beta = {{ beta|default('-1', true) }}
! calculated coll = {{ coll_calc|default('', true) }}
coll = {{ coll|default('-1', true) }}
collision_op = "{{ collision_op|default('landau', true) }}"
coll_cons_model = 'default'
coll_f_fm_on = F
! hyper-diffusion
hyp_z = {{ hyp_z|default('0.25', true) }}
hyp_v = {{ hyp_v|default('0.20', true) }}
hypz_opt = F
! eigenvalue computation
n_ev = {{ n_ev|default('1', true) }}
/

&nonlocal_x
/

&external_contr
/

&geometry
magn_geometry = "{{ magn_geometry|default('miller', true) }}"
! parameters for `miller` and `s_alpha`
trpeps = {{ trpeps|default('0.18', true) }}
q0 = {{ q0|default('1.4', true) }}
shat = {{ shat|default('0.8', true) }}
major_R = {{ major_R|default('1.', true) }}
! amhd = -q^2 R * (2\mu/Bref^2) * (dp/dpsi)/(dr/dpsi)
! calculated amhd = {{ amhd_calc|default('', true) }}
amhd = {{ amhd|default('-1', true) }}
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
! dpdx_pm = (2\mu n_ref T_ref/ Bref^2) * sum(n_s T_s(omn_s + omt_s))
! calculated dpdx_pm = {{ dpdx_pm_calc|default('', true) }}
! dpdx_pm_2 = - (dp/dr) / (2\mu/Bref^2)
! calculated dpdx_pm_2 = {{ dpdx_pm_2_calc|default('', true) }}
dpdx_pm = {{ dpdx_pm|default('-2', true) }}
dpdx_term = "{{ dpdx_term|default('full_drift', true) }}"
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
mass = {{ ele_mass|default('2.5E-3', true) }}
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
