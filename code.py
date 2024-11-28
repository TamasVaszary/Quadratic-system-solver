"""
Solving a system of quadratic system of equations using Carleman linearization

The system and the solutions:
https://www.wolframalpha.com/input?i=x%5E2%2By%5E2-x-xy%3D2%2C+2x%5E2-y-y%5E2%3D-4

Author: Tamas Vaszary

"""

import numpy as np
import matplotlib.pyplot as plt

def F(j):

    if j == 0:
        return np.array([[-2],[4]])
    elif j == 1:
        return np.array([[-1,0],[0,-1]]) 
    elif j == 2:
        return np.array([[1,-1,0,1],[2,0,0,-1]])


def little_A(i, j):
    #gives A_{i+j-1}^i
    Fj = F(j)
    
    n = F(1).shape[0]
    
    result = np.zeros((n**i,n**(i+j-1)))
    
    for nu in range(1, i + 1):
        stuff = np.kron(Fj,np.eye(n**(i-nu)))
            
        term = np.kron(np.eye(n**(nu-1)),stuff)
        
        result += term  
    
    return result


def generate_row(i, N):
    matrix_list=[]
    n = F(1).shape[0]
        
    if i==1:
        matrix_list.append(little_A(i,1))
        matrix_list.append(little_A(i,2))
        
        size_1 = int(n * (n**N-n**(i+1)) / (n-1))
        matrix_list.append(np.zeros((n**i,size_1)))
    
    elif i==N:
        size_2 = int((n**(i-1)-n) / (n-1) )
        matrix_list.append(np.zeros((n**i,size_2)))
            
        matrix_list.append(little_A(i,0))
        matrix_list.append(little_A(i,1))
    
    else:
        size_3 = int( (n**(i-1)-n) / (n-1) )
        matrix_list.append(np.zeros((n**i,size_3)))
            
        matrix_list.append(little_A(i,0))
        matrix_list.append(little_A(i,1)) 
        matrix_list.append(little_A(i,2))
        
        size_4 = int( n * (n**N-n**(i+1)) / (n-1) )
        matrix_list.append(np.zeros((n**i,size_4)))
            
    return np.hstack(matrix_list)


def A(N):
    matrix_list=[]
    
    for i in range(1,N+1):
        matrix_list.append(generate_row(i, N))
            
    return np.vstack(matrix_list)

def b(N):
    n = F(1).shape[0]
    siz = int((n**(N+1)-n) / (n-1))
    
    listi = [F(0),np.zeros((siz-n,1))]
    return np.vstack(listi)


#N = 10  # Truncation level of Carleman linearization

#B=b(N)
#mat = A(N)

#det=np.linalg.det(mat)

#eigenvalues = np.linalg.eigvals(mat)

#cond = np.linalg.cond(mat)

#x = -np.linalg.solve(mat, B)

#print(x[0],x[1])

def checker(a,b):
    x = np.array([[a],[b]])
    
    return F(2)@np.kron(x,x) + F(1)@x + F(0)
 
def rel_l2_diff(N,a_exact,b_exact):
    B=b(N)
    mat = A(N)
    num_sol = -np.linalg.solve(mat, B)

    num_sol_cropped = np.array([[num_sol[0][0]],[num_sol[1][0]]])
    exact = np.array([[a_exact],[b_exact]])

    return np.linalg.norm(exact-num_sol_cropped) / np.linalg.norm(exact)

a_exact = 0.21454170106513224517#from the wolfram link
b_exact = 1.5837601308672361516

y_ax = []
x_ax = []

for N in range(2,11):
    y_ax.append(rel_l2_diff(N,a_exact,b_exact))
    x_ax.append(N)
    

plt.plot(x_ax, y_ax, '.', color='blue')

plt.xlabel(r'Number of Carleman steps $N$')
plt.ylabel(r'Relative $l_2$ differenec of numerical and exact solutions ')
plt.title('Convergence of the numerical solution of a quadratic system')

plt.grid(True)  
plt.show()








