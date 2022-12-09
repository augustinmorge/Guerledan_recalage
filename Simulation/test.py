import numpy as np
from numpy.linalg import inv, eig

def ls_ellipse(x,y):

   J = np.hstack((x*x, x*y, y*y, x, y))
   K = np.ones_like(x) #column of ones

   JT=J.transpose()
   JTJ = np.dot(JT,J)
   InvJTJ=np.linalg.inv(JTJ);
   ABC= np.dot(InvJTJ, np.dot(JT,K))

   # ABC has polynomial coefficients A..E
   # Move the 1 to the other side and return A..F
   # A x^2 + B xy + C y^2 + Dx + Ey - 1 = 0
   eansa=np.append(ABC,-1)

   return eansa

def polyToParams(v):

   # convert the polynomial form of the ellipse to parameters
   # center, axes, and tilt
   # v is the vector whose elements are the polynomial
   # coefficients A..F
   # returns (center, axes, tilt degrees, rotation matrix)

   #Algebraic form: X.T * Amat * X --> polynomial form

   Amat = np.array(
   [
   [v[0],     v[1]/2.0, v[3]/2.0],
   [v[1]/2.0, v[2],     v[4]/2.0],
   [v[3]/2.0, v[4]/2.0, v[5]    ]
   ])


   #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
   # equation 20 for the following method for finding the center
   A2=Amat[0:2,0:2]
   A2Inv=inv(A2)
   ofs=v[3:5]/2.0
   cc = -np.dot(A2Inv,ofs)

   # Center the ellipse at the origin
   Tofs=np.eye(3)
   Tofs[2,0:2]=cc
   R = np.dot(Tofs,np.dot(Amat,Tofs.T))

   R2=R[0:2,0:2]
   s1=-R[2, 2]
   RS=R2/s1
   (el,ec)=eig(RS)

   recip=1.0/np.abs(el)
   axes=np.sqrt(recip)

   return (cc,axes)
