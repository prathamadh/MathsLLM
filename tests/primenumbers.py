import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy

# Number of iterations
num=10
iter = 100
s=5
# Initializing left and right sides
left = 0
right = 1
primes = list(sympy.primerange(1, sympy.prime(iter) + 1))
limit =10000
stoi=[]
stoj=[]
stog=[]
for i in primes:
  for j in primes:
     n=i*j
     for g in range(2,limit):
        if g**2 % n ==1 :
          stoi.append(i)
          stoj.append(j)
          stog.append(g)
          break

# Generating the first `iter` primes
# Define the function to visualize the voxel grid
def visualize_voxel_grid():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set view angle
    ax.view_init(elev=45, azim=45)
    
    # Plot voxels (replace stoi, stoj, stog with your voxel coordinate arrays)
    ax.scatter(stoi[num*100:num*100+100], stoj[num*100:num*100+100], stog[num*100:num*100+100], s=1)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title
    plt.title('Voxel Representation')
    plt.show()

# Call the function (ensure stoi, stoj, stog are defined with voxel coordinates)
visualize_voxel_grid()
