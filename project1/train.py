la = [2,3,5,4]

def sort1():
    l = len(la)
    for i in range(l-1):
        for j in range(l-1):
            if la[i] > la[i+1]:
                la[i], la[i+1] = la[i+1], la[i]
sort1()
p = [1,2] + [3,4]
print(p)