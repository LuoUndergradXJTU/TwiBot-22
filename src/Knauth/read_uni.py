import numpy as np
low_bounds=[]
with open('unicode.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        try:
            line=line[:line.find('：')]
            a,_=line.split('-')
            low_bounds.append(int(a,16))
        except:
            a,_=line.split('–')
        #print()
    
np.save('uni_class.npy',np.array(low_bounds))