
import math
import numpy as np
from PIL import Image

#import matplotlib.pyplot as plt 
#from mpl_toolkits.mplot3d import Axes3D 

#timeit.timeit('np.asarray(im)', 'import numpy as np; from PIL import Image; im = Image.open("bmw.jpg")', number=10) / 10
#timeit.timeit('np.array(data)', 'import numpy as np; from PIL import Image; im = Image.open("bmw.jpg"); data=im.getdata()', number=10) / 10
#ar2 = (ar * np.array([.5, .5, .5])).astype(np.uint8)
#new = Image.fromarray(ar2)
#np.linalg.norm(a-b)

#for row in np.niter(data)[0]:
#    print(str(row))

#iters = 100
k = 24

def save_colors(centroids, name):
    centroids = np.array(sorted(centroids, key=lambda row: np.sum(row), reverse=True))
    im = Image.new('RGB', (k, 1))
    data = [(c[0], c[1], c[2]) for c in centroids.astype(int)]
    im.putdata(data)                                          
    im.save(name)

def distance(a, b):
    dist = [(v1 - v2)**2 for v1, v2 in zip(a, b)]
    dist = math.sqrt(sum(dist))
    return dist

im = Image.open("elden.jpg")
im = im.resize((480, 480))
#im = im.resize((40, 100))
data = np.asarray(im)
flat_len = data.shape[0] * data.shape[1]
flat = np.reshape(data, (flat_len, 3))

centroids = np.empty(shape=[k, 3])
for i in range(k):
    pixel = flat[np.random.choice(flat.shape[0])]
    while pixel in centroids:
        pixel = flat[np.random.choice(flat.shape[0])]
    centroids[i] = pixel

def iterate():
    diff = 0
    diff_cube = np.array([np.linalg.norm(flat-centroid, axis=-1) for centroid in centroids])
    indices = np.argmin(diff_cube, axis=0)
    combined = np.insert(flat, 3, indices, axis=-1)
    sort = combined[combined[:, 3].argsort()]
    groups = np.split(sort[:, :], np.cumsum(np.unique(sort[:, 3], return_counts=True)[1])[:-1])
    #centroids = np.array([np.rint(np.mean(np.delete(group, 3, -1), axis=0)) for group in groups])

    for i in range(k):
        group = groups[i]
        chopped = np.delete(group, 3, -1)
        mean = np.rint(np.mean(chopped, axis=0))
        diff += distance(centroids[i], mean)
        centroids[i] = mean

    return diff

def run(iters):
    for i in range(iters):
        diff = iterate()
        print(f"Diff: {diff}")
        if diff == 0.0:
            break

run(50)

"""
def graph():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.scatter(data[:,0], data[:,1], data[:,2], s= 0.5)
    plt.show()
"""
