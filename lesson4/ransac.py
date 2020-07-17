 
def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]
    
def is_inlier(norvec, x, threshold):
    return np.abs(norvec.dot(augment([x]).T)) < threshold
    
def myRansac(data):
    num=len(data)
    p=0.99
    e=0.5
    max_iterations=np.log(1-p)/np.log(1-np.power(1-e,3))
    best_ic=num*0.4
    goal_inliers=num*0.3
    inlinepoints=[]
    for i in range(max_iterations):    
        s = random.sample(data, 3)
        m = estimate(s)
        ic = 0
        for j in range(num):
            if is_inlier(m, data[j],0.05):
                ic += 1
                inlinepoints.append(data[j)]
        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers:
                break
    return best_model, best_ic, inlinepoints

def ground_segmentation(data):
    _, _, inlinepoints=myRansac(data)
    data=list(data)
    segpoint=set(data)-set(inlinepoints)
    return np.array(segpoint)
