import pandas as pd

def fetch_system_info(addr):
    '''
    This function extracts the system information from the 'system_info.txt' file
    
    Parameter - 
    Addr: Address of the folder which contains 'info_files' folder 
    
    Returns (in this order only) -
    Number of channels: an integer
    Channel names: a list object
    Sampling frequency: an integer
    '''
    addr = addr + '\\info_files\\system_info.txt'
    system_info = pd.read_csv(addr, sep = '\t', header = None)
    #system_info[0,1] gives the number of channels
    #system_info[1,1] gives the string of channel names separated by commas
    #system_info[2,1] gives sampling frequency
    print system_info
    #print (system_info[0,1]), str(system_info[1,1]).split(sep = ','), (system_info[2,1])
    
    
fetch_system_info('E:\\SPADE')



stri = 'aman'
types = type(stri)
types

import pandas as pd
addr = 'E:\\SPADE' + '\\info_files\\system_info.txt'
system_info = pd.read_csv(addr, sep = '\t', header = None).values

system_info[1,1] = system_info[1,1].split(',')
system_info[1,1] = [int(i) for i in system_info[1,1]]
#system_info[0,1] gives the number of channels
#system_info[1,1] gives the string of channels separated by commas
#system_info[2,1] gives sampling frequency
#return int(system_info[0,1]), str(system_info[1,1]).split(sep = ','), int(system_info[2,1])
return int(system_info[0,1]), int(system_info[2,1])
   