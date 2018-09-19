##for 09th september
#Store time index as a counter and data in a list and convert to #numpy and save

from emotiv import Epoc
import time
import numpy 

def procedure():
    time.sleep(0.01)

#w, h = 14,4. 
#dataMatrix = [[0 for x in range(w)] for y in range(h)] 
c=numpy.zeros((4, 14))
epoc = Epoc()
t0 = time.clock()
procedure()
ctr=0
#c=[]
d=[]
while (time.clock() - t0)<10:
    data1 = epoc.get_raw() #14-by-n numpy array containing raw #data for AF3 through AF4
    #this is equivelant to:
    #c=np.concatenate((dataMatrix,c), 
    data = epoc.aquire([3,4,5,6,7,8,9,10,11,12,13,14,15,16]) #AF3 #through AF4
    #print type(data)
    #print data.shape
    #print (numpy.asarray(c)).shape
    #c=numpy.append(c,data)
    c=numpy.concatenate((numpy.asarray(c),numpy.asarray(data).T), axis=0)
    print (c.shape)
    times = epoc.times #array of interpolated timestamps, just as #before
    #print type(times)
    #print times.shape
    #except KeyboardInterrupt:
    #d=numpy.concatenate((numpy.asarray(d),numpy.asarray(times)), axis=0)
    d=numpy.append(d,times)


numpy.savetxt("abc.csv", c, delimiter=",")
numpy.savetxt("dabc1.csv", d, delimiter=",")

##numpy.savetxt("data_tushar_open_1.csv", c, delimiter=",")
##numpy.savetxt("times_tushar_open_1.csv", d, delimiter=",")