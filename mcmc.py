import sys, warnings
import numpy as np
import matplotlib.pyplot as plt

#METODOS
#FUNCION DEL MODELO
def modelo(lr, lrho_0,lr_0, alpha, beta):
    #rho(r)=rho_0/((r/r_0)**alpha(1+(r/r_0))**beta)
    r=10**lr
    r_0=10**lr_0
    lrho=lrho_0-alpha*(lr-lr_0)-beta*(np.log10(r+r_0)-lr_0)
    return lrho
#likehood
def likehood(y_experimental,y_modelo):
    err=0.5*np.sum((y_experimental-y_modelo)**2)
    return -err
#MEJOR VALOR
def mejorValor(pasos):
    n,b=np.histogram(pasos,bins=20)
    i_maximo=np.argmax(n)
    mejorValor=(b[i_maximo]+b[i_maximo+1])/2.0
    return mejorValor
#MCMC
def mcmc(x_i,y_i):
    #Adivinando
    lrho_0_0=4.0
    lr_0_0=-1.0
    alpha_0=1
    beta_0=2.0
    #inicializacion
    x_0=x_i
    y_0=modelo(x_0,lrho_0_0,lr_0_0,alpha_0,beta_0)
    
    lrho_0_caminata=np.empty((0))
    lr_0_caminata=np.empty((0))
    alpha_caminata=np.empty((0))
    beta_caminata=np.empty((0))
    likehood_caminata=np.empty((0))
    
    lr_0_caminata=np.append(lr_0_caminata,lr_0_0)
    lrho_0_caminata=np.append(lrho_0_caminata,lrho_0_0)
    alpha_caminata=np.append(alpha_caminata,alpha_0)
    beta_caminata=np.append(beta_caminata,beta_0)
    likehood_caminata=np.append(likehood_caminata,likehood(y_i,y_0))
    #MCMC
    pasos=100000
    for i in range(pasos):
        lrho_0_temp=np.random.normal(lrho_0_caminata[i],0.1)
        lr_0_temp=np.random.normal(lr_0_caminata[i],0.1)
        alpha_temp=np.random.normal(alpha_caminata[i],0.1)
        beta_temp=np.random.normal(beta_caminata[i],0.1)
        
        y_pas=modelo(x_i,lrho_0_caminata[i],lr_0_caminata[i],alpha_caminata[i],beta_caminata[i])
        y_temp=modelo(x_i,lrho_0_temp,lr_0_temp,alpha_temp,beta_temp)
        
        likehood_pas=likehood(y_i,y_pas)
        likehood_temp=likehood(y_i,y_temp)
        
        d=likehood_temp/likehood_pas
        #Se comparan lo obtenido anteriormente con lo generado aleatoriamente
        if(d<=1.0):
            #se aceptan los temps y se agregan
            lrho_0_caminata=np.append(lrho_0_caminata,lrho_0_temp)
            lr_0_caminata=np.append(lr_0_caminata,lr_0_temp)
            alpha_caminata=np.append(alpha_caminata,alpha_temp)
            beta_caminata=np.append(beta_caminata,beta_temp)
            likehood_caminata=np.append(likehood_caminata,likehood_temp)
        else:
            #Metropolis-Hastings
            ale=np.random.random()
            if(np.log(ale)<=-d):
                lrho_0_caminata=np.append(lrho_0_caminata,lrho_0_temp)
                lr_0_caminata=np.append(lr_0_caminata,lr_0_temp)
                alpha_caminata=np.append(alpha_caminata,alpha_temp)
                beta_caminata=np.append(beta_caminata,beta_temp)
                likehood_caminata=np.append(likehood_caminata,likehood_temp)
            else:
                lrho_0_caminata=np.append(lrho_0_caminata,lrho_0_caminata[i])
                lr_0_caminata=np.append(lr_0_caminata,lr_0_caminata[i])
                alpha_caminata=np.append(alpha_caminata,alpha_caminata[i])
                beta_caminata=np.append(beta_caminata,beta_caminata[i])
                likehood_caminata=np.append(likehood_caminata,likehood_caminata[i])
    #MejorValor
    lrho_0=mejorValor(lrho_0_caminata[100:])
    lr_0=mejorValor(lr_0_caminata[100:])
    alpha=mejorValor(alpha_caminata[100:])
    beta=mejorValor(beta_caminata[100:])
    #Starndar Deviation
    log_rho_0_std = np.std(lrho_0_caminata[100:])
    log_r_c_std = np.std(lr_0_caminata[100:])
    alpha_std = np.std(alpha_caminata[100:])
    beta_std = np.std(beta_caminata[100:])

    #graficos
    datos=np.array([lrho_0_caminata[100:],lr_0_caminata[100:],alpha_caminata[100:],beta_caminata[100:]])
    datos=datos.T

    return lrho_0,lr_0,alpha,beta, log_rho_0_std, log_r_c_std, alpha_std, beta_std



warnings.filterwarnings("ignore")

def radio(data):
    E_pot=data[:,6]
    min_potencial=np.argmin(E_pot)
    x = data[:,0]-data[min_potencial,0]
    y = data[:,1]-data[min_potencial,1]
    z = data[:,2]-data[min_potencial,2]
    r = np.sqrt(x**2 + y**2 +z**2)
    r = np.sort(r)
    return r[1:]
if (len(sys.argv)!=2):
    sys.exit('No Cumple Formtato')
n_cuerpos=int(sys.argv[1])
data_final=np.loadtxt('final_{}.dat'.format(n_cuerpos))
#x
radio_final=radio(data_final)
lradio_final=np.log10(radio_final)
h,b=np.histogram(lradio_final,bins=10)
lr_centro=0.5*(b[1:]+b[:-1])
#y
lrho=np.log10(h)-2.0*lr_centro

plt.figure()
plt.plot(lr_centro,lrho,label='$\mathrm{data}$')

plt.xlabel(r'$\log{(r)}$')
plt.ylabel(r'$\log{(\rho (r))}$')
plt.legend(loc='lower left')

plt.savefig('perfil_densidad_datos.png')

lrho_0_0=4.0
lr_0_0=-1.0
alpha_0=1
beta_0=2.0

primer_ajuste=modelo(lr_centro,lrho_0_0,lr_0_0,alpha_0,beta_0)
#grafica del tanteo

plt.figure()

plt.plot(lr_centro, lrho, label='$\mathrm{data}$')
plt.plot(lr_centro, primer_ajuste, label='$\mathrm{INICIAL}$')

plt.xlabel(r'$\log{(r)}$')
plt.ylabel(r'$\log{(\rho (r))}$')
plt.legend(loc='lower left')

plt.savefig('perfil_densidad_INICIAL.png')
#grafica del ajuste final

lrho_0,lr_0,alpha,beta,log_rho_0_std, log_r_c_std, alpha_std, beta_std=mcmc(lr_centro,lrho)
ajuste=modelo(lr_centro,lrho_0,lr_0,alpha, beta)


plt.figure(figsize=(12,8))

plt.plot(lr_centro, lrho, label='$\mathrm{data}$')
plt.plot(lr_centro, ajuste, label='$\mathrm{ajuste}$')

plt.xlabel(r'$\log{(r)}$')
plt.ylabel(r'$\log{(\rho (r))}$')
plt.legend(loc='lower left')

plt.savefig('perfil_densidad_AJUSTE.png')

print('log(rho_0) = {} +/- {}'.format(lrho_0, log_rho_0_std))
print('log(r_0) = {} +/- {}'.format(lr_0, log_r_c_std))
print('alpha = {} +/- {}'.format(alpha, alpha_std))
print('beta = {} +/- {}'.format(beta, beta_std))