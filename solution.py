import numpy as np
import math
from numpy.lib import scimath
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
#from matplotlib import colors

# user-defined inputs for FEM mesh:
# nnx: number of nodes along x-direction
# nny: number of nodes along y-direction
nnx = 3
nny = 3

# initiation of variables
nn  = nnx*nny;       # total number of nodes
elemNodeCount = 4;   # number of nodes per element
elemipCount = 4;     # number of Gauss integration points per element
elemDofCount = elemNodeCount;  # number of degree-of-freedom (DOF) per element
dofCount = nn;             # total number of DoFs
elemNaturalCoords  = np.array([[-1,-1], [-1,1], [1,1], [1,-1]])  # local coordinates of element nodes


# ---- determine mesh coordinates ------
nelx = nnx-1
nely = nny-1
nel2d = nelx*nely;

xs = np.linspace(0, 1, nnx)
ys = np.linspace(0, 1, nny)  

# coordinate-x
x = xs;
for iz in range(nely):
  x = np.vstack((x,xs)) 

x = np.reshape(x, nn,order='F')

y = ys
for iy in range(nely):
  y = np.hstack((y,ys)) 

# ---- global coordinates of nodes ------
coord = np.transpose(np.vstack((x,y)))

# element connectivities
conn = np.zeros((nel2d,4),dtype=int)

in0 = 0;
for ix in range(1,nelx+1):
  in0 = in0 + 1;
  for iy in range(1,nely+1):
    iel = (ix-1)*nely + iy
    conn[iel-1,:]=np.array([in0, in0+1, in0+1+nny, in0+nny])
    in0 = in0 + 1;
 

# ---- determine constraint dofs ---- 
consDofB = np.arange(nny,nn-2*nny+1,nny)
consDofT = np.arange(2*nny-1,nn-nny, nny)
consDofL = np.arange(nny)
consDofR = np.arange(nn-nny,nn,1)
consDofs = np.concatenate((consDofB, consDofT, consDofL, consDofR))



# ---- create a K-matrix indices ---- 
row = np.zeros(16*nel2d)
col = np.zeros(16*nel2d)
data = np.zeros(16*nel2d)
cc = 0

#---- initialize solution vector and force vector ---- 
u = np.zeros(nn)
f = np.zeros(nn)



# ---- loop over elements ---- 
for iel in range(nel2d):
    
    elemConn = conn[iel, :]  

    elemglobDofIndex = np.array([elemConn[0]-1, 
    	elemConn[1] - 1 , elemConn[2]-1,
    	elemConn[3] - 1 ])


    #coordinates of the gauss integration points
    ipCoords = np.array([[ -1/math.sqrt(3), -1/math.sqrt(3) ],
        [ -1/math.sqrt(3), 1/math.sqrt(3) ],
        [ 1/math.sqrt(3), 1/math.sqrt(3) ],
        [ 1/math.sqrt(3), -1/math.sqrt(3) ]])
    weight = 1     
 
    # loop over integration points
    for ip in range(elemipCount):
        
        xip = ipCoords[ ip, 0 ];
        yip = ipCoords[ ip, 1 ];

        # compute shape function
        Ni = np.zeros(elemNodeCount)

        for ishape in range(elemNodeCount):
            
            Ni[ishape] = 1./4 * ( 1 + elemNaturalCoords[ishape, 0]*xip );
            Ni[ishape] = Ni[ishape] * ( 1 + elemNaturalCoords[ishape, 1]*yip );              
        
        matN = np.array([ Ni[0], Ni[1],  Ni[2],  Ni[3]])

        
        # get the coordinates of the current elements
        # elemeCoords( i,j): the ith node's jth coordinate
        # get the coordinates of all the nodes in the element
        elemCoords = coord[elemConn-1,:]


        # compute Gauss points coordinates
        ipGlobCoords = np.matmul(matN, elemCoords)

        # compute external force at integration points
        extf = 4*(-ipGlobCoords[1]**2+ipGlobCoords[1])*math.sin(math.pi*ipGlobCoords[0])

       
        # Gradient of N, GradN(i,j): the ith shape function with jth coordinate
        GradN = np.zeros( (4,2),dtype=float)	

        GradN[0,0] = 1./4 *  elemNaturalCoords[0,0] * ( 1 +  elemNaturalCoords[0, 1]* yip )
        GradN[0,1] = 1./4 *  elemNaturalCoords[0,1] * ( 1 +  elemNaturalCoords[0, 0]* xip )
        GradN[1,0] = 1./4 *  elemNaturalCoords[1,0] * ( 1 +  elemNaturalCoords[1, 1]* yip )
        GradN[1,1] = 1./4 *  elemNaturalCoords[1,1] * ( 1 +  elemNaturalCoords[1, 0]* xip )
        GradN[2,0] = 1./4 *  elemNaturalCoords[2,0] * ( 1 +  elemNaturalCoords[2, 1]* yip )
        GradN[2,1] = 1./4 *  elemNaturalCoords[2,1] * ( 1 +  elemNaturalCoords[2, 0]* xip )
        GradN[3,0] = 1./4 *  elemNaturalCoords[3,0] * ( 1 +  elemNaturalCoords[3, 1]* yip )
        GradN[3,1] = 1./4 *  elemNaturalCoords[3,1] * ( 1 +  elemNaturalCoords[3, 0]* xip )
     
        #compute jacobian

        matP = np.array([ [ GradN[0,0], GradN[1,0], GradN[2,0], GradN[3,0] ],
                [ GradN[0,1], GradN[1,1], GradN[2,1], GradN[3,1] ] ])


        matX = np.array([ [ elemCoords[0,0],  elemCoords[0,1] ],
                    [ elemCoords[1,0],  elemCoords[1,1] ],
                    [ elemCoords[2,0],  elemCoords[2,1] ],
                    [ elemCoords[3,0],  elemCoords[3,1] ] ])            
        
        Jac = np.matmul(matP , matX)

       
        # determinant of jacobian
        detJaco = np.linalg.det( Jac );
        
        
        # inverse of jacobian matrix
        Jac_inv = np.array([ [ Jac[1,1], -Jac[0,1] ],
                    [ -Jac[1,0], Jac[0,0] ] ])
                
        Jac_inv = 1. / detJaco * Jac_inv;


        # compute strain matrix B
        matB = np.transpose(GradN)

        # compute element stiffness matrix and force vector
        elemStiff = weight * detJaco * np.matmul(np.transpose(matB), matB)

        elemForce = weight * detJaco * extf * np.transpose(matN)

        # assemble global force vector
        f [elemglobDofIndex] += elemForce


    # prepare non-zero entries to create a spare matrix of stiffness 
    for ii in range(elemDofCount):
        
      gi = elemglobDofIndex [ii]
        
      for jj in range(elemDofCount): 
            
        gj = elemglobDofIndex [jj];
            
        row [cc] = gi                 
        col [cc] = gj
        data [cc] = elemStiff [ii,jj]

        cc = cc + 1


# ----  apply constraints ---- 
for ic in range(cc):
  if np.any(consDofs == row[ic]):
     if row[ic] == col[ic]:
        data [ic] = 1.
     else: 
        data [ic] = 0.



# ----  Assemble global stiffness matrix ---- 
K = coo_matrix((data, (row, col)), shape=(dofCount, dofCount)).tocsc()


# ----  Solve the system equation ---- 
u=spsolve(K,f)    

print('With a mesh of ' + str(nnx-1) + 'x' + str(nny-1) + ' elements, the solution is:')
print(u)


