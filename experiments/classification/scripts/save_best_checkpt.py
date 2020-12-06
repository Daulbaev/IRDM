from glob import glob
import os

def get_normalizations(folder_stats):
   ### empty normalization key corresponds to "_bn1-BN_resblock-BNReLU_odeblock-LNReLU"
    normalizations = []
    anode_logs = folder_stats['anode_logs']


    for dirname in glob('{}*'.format(anode_logs, recursive = False)):
        normalizations.append('_' + dirname.strip('{}'.format(anode_logs)))

    normalizations.append("") 
    
    return normalizations

def get_configs(folder_stats):
    configs_h1 = folder_stats['configs_h1']
    configs_h2 = folder_stats['configs_h2']
    configs_z = folder_stats['configs_z']
    
    configs_resnet34 = []
    configs_resnet18 = []
    configs_resnet10 = []
    configs_resnet6 = []
    configs_resnet4 = []

    for el in (configs_h1 +\
               configs_h2 +\
               configs_z):
        if 'resnet10' in el:
            configs_resnet10.append(el)
        elif 'resnet6' in el:
            configs_resnet6.append(el)
        elif 'resnet18' in el:
            configs_resnet18.append(el)
        elif 'resnet34' in el:
            configs_resnet34.append(el)
        elif 'resnet4' in el:
            configs_resnet4.append(el)
            
    return configs_resnet4, configs_resnet6, configs_resnet10, configs_resnet18, configs_resnet34

def get_best_acc(log_filename):
    with open(log_filename, 'r') as f:
        lines = f.readlines()
    
    best = lines[-1]
    if best.startswith('Best'):
        best_acc, best_epoch = [el.split(':')[-1].strip() for el in best.strip().split(',')]
        best_acc = float(best_acc)
        best_epoch = int(best_epoch)
    
        return best_acc, best_epoch
    else:
        return None, None

if __name__=="__main__":
    anode_logs = "/gpfs/gpfs0/y.gusak/anode_workshop_logs/classification"
    inplanes = 64
    n = 8


    f = {'anode_logs' : anode_logs,
             'configs_z' : [('resnet10', 'Euler', 8, '1e-1', 512),\
               ('resnet10', 'Euler', 2, '1e-1', 512),\
               ('resnet4', 'Euler', 8, '1e-1', 512),\
               ('resnet4', 'Euler', 16, '1e-1', 512),\
                     ],
             'configs_h1' : [],
             'configs_h2' : [],
    }
    normalizations = get_normalizations(f)
    configs_resnet4, configs_resnet6, configs_resnet10, configs_resnet18, configs_resnet34 = get_configs(f)

    for normalization in normalizations:
      if len(normalization) < 1:
        continue
      for network in ['resnet4', 'resnet10']:
        for log_filename in glob('{}/{}/Euler_n{}_lr1e-1_bs512/*/logs'.format(anode_logs + normalization,\
                                                         "{}_inplanes{}".format(network, inplanes), n),\
                                 recursive = True):
            
            best_acc, best_epoch = get_best_acc(log_filename)
            if best_acc is not None:
                print(normalization, best_acc, best_epoch)
                
                prefix = log_filename.replace('_logs/', '/').strip('logs')
                #print(prefix)
               
                best_checkpt = '{}checkpt_{}.pth'.format(prefix, best_epoch)
                save_checkpt = '{}checkpt_best.pth'.format(log_filename.strip('logs'))
             
                os.system("cp {} {}".format(best_checkpt, save_checkpt))
