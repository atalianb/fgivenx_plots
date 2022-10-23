from os import O_ACCMODE
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize,integrate,interpolate
from matplotlib.pyplot import cm
from fgivenx import plot_contours, samples_from_getdist_chains
import matplotlib.style as style
style.use('seaborn-colorblind')

#####Gravitational Constant
G_kpc = 4.302e-6#kpc/SolarMass(km/s)^2
####
##Integration values
###
x0_0 = 2.#first integration
x0_0v1 = 0.0# Start of integration. Use 0.01 for continuity in l=3 solution, for 0,1,2 0.0 it's ok
xf_0v1 = 10.5# End of integration
step_0 = 0.2#Step to integrate from x0_0 until xf_0v1 is reached
h = 0.1
u1 = 2.58# 1st trial value of unknown init. cond. gamma_100
u2 = 3.65# 2nd trial value of unknown init. cond. gamma_210
u3 = 4.65# 3th trial value of unknown init. cond. gamma_320
u_A = np.array([u1,u2,u3])
##l=0
u0 = -1.5
u1 = -0.5#1.2# 1st trial value of unknown init. cond. gamma_100
u2 = -0.4#2.7# 2nd trial value of unknown init. cond. gamma_210
u3 = -1.9#4.9# 3th trial value of unknown init. cond. gamma_320
u_012_V = np.array([u0,u1, u2,u3])

def Integrate(func,x0,y0,x,h):
    # Finds value of y for a given x using step size h 
    # and initial value y0 at x0.
    def RK4(func,x0,y0,h):
        K0 = h*func(x0,y0)
        K1 = h*func(x0 + 0.5*h, y0 + 0.5*K0)
        K2 = h*func(x0 + 0.5*h, y0 + 0.5*K1)
        K3 = h*func(x0 + h, y0 + K2)
        return (K0 + 2.*K1 + 2.*K2 + K3)/6.
    X = []
    Y = []
    X.append(x0)
    Y.append(y0)
    while x0 < x:
         # Count number of iterations using step size or
        h = min(h,x - x0)# step height h
        y0 = y0 + RK4(func,x0,y0,h)#update next value of y
        x0 = x0 + h#update next value of x
        X.append(x0)
        Y.append(y0)
    return np.array(X),np.array(Y)
######
def shooting012_V(func,u,x0,x,xf,step,phi0,phi1,phi2,h=0.01):
    def IC_V(u):
        return np.array([phi0,0.,u[0],0.,u[1],0.,phi1,0.,u[2],0.,phi2,0.,u[3],0.])
    def res(u):# Boundary condition residual
        X,Y = Integrate(func,x0,IC_V(u),x,h)
        y = Y[len(Y) - 1]#last value of Y
        r = np.zeros(len(u))
        r[0] = y[0]#y0(inf)=0
        r[1] = y[2]/x + y[3]#y_3(inf)/r + y_4(inf)=0
        r[2] = y[6]#
        r[3] = y[10]#y6(inf)=0
        #r[1] = y[1] + np.sqrt(2.*y[4]**2.)*y[0]#y_2(inf)+sqrt(2y_5^2)y_1(inf)=0
        return r
    while x<=xf:
        root = optimize.root(res,u)
        #print(root.x)
        u = root.x
        root_temp = optimize.root(res,root.x)
        X,Y = Integrate(func,x0,IC_V(root_temp.x),x,h)
        x = x+step
    return X,Y,root_temp
#######
def rhs(x,y):
    dy = np.zeros(14)
    dy[0] = y[1]
    dy[2] = y[3]
    dy[4] = 0.
    dy[5] = y[0]**2.*x**2.
    dy[6] = y[7]
    dy[8] = 0.
    dy[9] = y[6]**2.*x**2.
    dy[10] = y[11]
    dy[12] = 0.
    dy[13] = y[10]**2.*x**2.
    if(x==0.):
        dy[1] = 2.*(y[2]-y[4])*y[0]/3.
        dy[3] = y[0]**2./3.
        dy[7] = 2.*(y[2] - y[8])*y[6]/5.
        dy[11] = 2.*(y[2]-y[12])*y[10]/7.
    else:
        dy[1] = 2.*(y[2]-y[4])*y[0] - 2.*y[1]/x
        dy[3] = y[0]**2. + 3.*x*2.*y[6]**2. + 5.*x**4.*y[10]**2. - 2.*y[3]/x
        dy[7] = 2.*(y[2] -y[8])*y[6] -4.*y[7]/x
        dy[11] = 2.*(y[2]-y[12])*y[10] - 6.*y[11]/x
    return dy
#####
def Mass_func(r,phi,l):
    Int = np.zeros(len(r))
    dr = np.diff(r)[0]
    if l==0.:
        phi_array = np.array(phi[:,0])
    if l==1.:
        phi_array = np.array(phi[:,6])
    if l==2.:
        phi_array = np.array(phi[:,10])
    for i in range(0,len(r)-1):
        Int[i+1] = dr*(phi_array[i+1]**2.*r[i+1]**(2.*l+2.)) + Int[i]
    return Int
####
def Vc2_cir(r,eps,M):
    units =8.95e10*eps**2.
    return (units*M)/r
####
##Units for r in kpc
###
def r_units(r,eps,m_a):
    return (6.39e-27*r)/(eps*m_a)
def Vc_(r,m_a,eps,phi0,phi1,phi2):
    X,Y,root_f=shooting012_V(rhs,u_012_V,x0_0v1,x0_0,xf_0v1,step_0,phi0,phi1,phi2)
    M_r0 = Mass_func(X,Y,l=0.)+Mass_func(X,Y,l=1.)+Mass_func(X,Y,l=2.)#Integrates rho(r) to obtain M(r)
    Vc2_r0 = Vc2_cir(X,eps,M_r0)#Vc^2[km/s]^2 theoretical
    X0_units = r_units(X,eps,m_a)#r[kpc] theoretical
    M_r0_units = M_r0*eps*1.34e-10/m_a
    if X0_units[-1]<r[-1]:
        #array from last element of the r[kpc] theoretical to the last element of the data array,
        # with 80 elements. It can be replaced by np.arange(X0_units[-1],vecRp_data[-1],0.1) 
        #but you have to be careful in the next function with interpolate
        r_array = np.linspace(X0_units[-1],r[-1],80)
        Vc2_rmayor = G_kpc*M_r0_units[-1]/r_array#Computes Vc^2 with with the last result from M(r)
        Vc2_total = np.append(Vc2_r0,Vc2_rmayor)#creates an array of Vc^2 with Vc2_r0 and Vc2_rmayor
        r_total = np.append(X0_units,r_array)
        return r_total,Vc2_total
    else:
        return X0_units,Vc2_r0
#####
def Vc_xi2(r,m_a,eps,phi0,phi1,phi2):
    Vc = Vc_(r,m_a,eps,phi0,phi1,phi2)
    #If you want to use np.arange in the previous function, It is recommended to use extrapolate
    f = interpolate.interp1d(Vc[0],Vc[1],fill_value='extrapolate')
    Vc_new = f(r)
    return Vc_new
####
def Vc_m_a_eps_l012(r,params):
    m_a,eps,phi0,phi1,phi2 = params
    Vc2 = Vc_xi2(r,m_a,eps,phi0,phi1,phi2)
    return np.sqrt(Vc2)
#####
def Vc_interpol(r,X,Vc):
    #If you want to use np.arange in the previous function, It is recommended to use extrapolate
    f = interpolate.interp1d(X,Vc,fill_value='extrapolate')
    Vc_new = f(r)
    return Vc_new
####3
data_path = "/Users/atalianb/Documents/data_LBSG/data_used_by_Tula/"
data = np.loadtxt(data_path+'U11648.dat')
Galaxy_name = 'UGC11648'
vecRp_data = np.array([row[1] for row in data])# galactocentric distance [kpc]
vecvRp_data = np.array([row[5] for row in data])# rotation velocity [km/s]
vecerrvRp_data = np.array([row[6] for row in data])# error in rotation velocity [km/s]
params = np.array([10**(-2.3464225E+01),10**(-3.2565943E+00),
                10**(-1.4565523E-02),10**(-1.0716972E+00),
                10**(-7.6590629E-01)])
path_nested = '/Users/atalianb/Documents/Doctorado/fgivenx_plots/chains/U11648'
chains = np.loadtxt(path_nested+'/Rotation_phy_RC_nested_dynesty_multi_1.txt')
phi0 = params[2]
phi1 = params[3]
phi2 = params[4]
X012,Y012,root_012=shooting012_V(rhs,u_012_V,x0_0v1,x0_0,xf_0v1,step_0,phi0,phi1,phi2)
m_a= params[0]
eps = params[1]
M_l0 =  Mass_func(X012,Y012,l=0.)
M_l1 = Mass_func(X012,Y012,l=1.)
M_l2 = Mass_func(X012,Y012,l=2.)
Vc2_l0 = Vc2_cir(X012,eps,M_l0)#Vc^2[km/s]^2 theoretical
Vc2_l1 = Vc2_cir(X012,eps,M_l1)
Vc2_l2 = Vc2_cir(X012,eps,M_l2)
X0_units = r_units(X012,eps,m_a)#r[kpc] theoretical
M_l0_units = M_l0*eps*1.34e-10/m_a
M_l1_units = M_l1*eps*1.34e-10/m_a
M_l2_units = M_l2*eps*1.34e-10/m_a
Vc_l0 = Vc_interpol(vecRp_data,X012,Vc2_l0)
Vc_l1 = Vc_interpol(vecRp_data,X012,Vc2_l1)
Vc_l2 = Vc_interpol(vecRp_data,X012,Vc2_l2)
#plt.errorbar(vecRp_data,vecvRp_data,yerr=vecerrvRp_data,fmt='.',label='data')
#plt.plot(vecRp_data,Vc_m_a_eps_l012(vecRp_data,params),label='total')
#plt.ylabel(r'$v_{c}(r)$[km/s]')
#plt.xlabel("r[kpc]")
#plt.title(Galaxy_name)
#plt.plot(X0_units,np.sqrt(Vc2_l0),label=r'$\psi_{100}$')
#plt.plot(X0_units,np.sqrt(Vc2_l1),label=r'$\psi_{210}$')
#plt.plot(X0_units,np.sqrt(Vc2_l2),label=r'$\psi_{320}$')
#plt.legend(loc='lower right')
#plt.xlim(0,vecRp_data[-1])
#####
##
#######
m_a_new = 10.**(chains.T[2][2000:10000])
eps_new = 10.**(chains.T[3][2000:10000])
phi0_new = 10.**(chains.T[4][2000:10000])
phi1_new = 10.**(chains.T[5][2000:10000])
phi2_new = 10.**(chains.T[6][2000:10000])
samples = np.array([(Anfw,rs,phi0,phi1,phi2) for Anfw,rs,phi0,phi1,phi2 in zip(m_a_new,eps_new,phi0_new,phi1_new,phi2_new)]).copy()
####
nx = 100
x = np.linspace(0.1, vecRp_data[-1], nx)
def PPS(r,theta):
    Anfw,rs,phi0,phi1,phi2=theta
    Vc = Vc_m_a_eps_l012(r,theta)
    return Vc
cbar = plot_contours(PPS,x,samples,contour_line_levels=[1,2],colors=plt.cm.Greys_r,alpha=0.8,parallel=7)#,ny=100)
cbar = plt.colorbar(cbar,ticks=[0,1,2])
cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$'])
plt.ylabel(r'$v_{c}(r)$[km/s]')
plt.xlabel("r[kpc]")
plt.plot(vecRp_data,Vc_m_a_eps_l012(vecRp_data,params),linewidth=2.5,c='blue', label='total')
plt.errorbar(vecRp_data,vecvRp_data,yerr=vecerrvRp_data,fmt='o',color='k',elinewidth=2.5,label='data')
plt.plot(X0_units,np.sqrt(Vc2_l0),label=r'$\psi_{100}$',linewidth=2.5)
plt.plot(X0_units,np.sqrt(Vc2_l1),label=r'$\psi_{210}$',linewidth=2.5)
plt.plot(X0_units,np.sqrt(Vc2_l2),label=r'$\psi_{320}$',linewidth=2.5)
plt.legend(loc='lower right')
plt.xlim(0,vecRp_data[-1])
plt.title(Galaxy_name)
plt.ylim(top=350)
plt.savefig('fgivenx_VcMultiL012_'+Galaxy_name+'_parallel_20porcent_burnin.pdf')
