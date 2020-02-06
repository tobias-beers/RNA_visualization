import numpy as np

'''
When comparing two matrices to check whether they are rotations of each other, just multiply both with their transposes.
If A = BR, then AA^T = BRR^T B^T and since RR^T = I for any rotation matrix R, AA^T will yield the same result for any rotation of B.
'''



def rotation_check(A, B, threshold_R=0.01, threshold_angle=0.1, method='find_rotation_mat',verbose=False, order_angles_by_magnitude=False):
    '''
    Use 'method='find_rotation_mat'' to find the matrix R where AR=B and check whether det(R)=1 and R^T=R^-1
    or use 'method='angles'' to compare the angles and magnitudes of A and B.
    The first method is theoretically solid but is susceptible to numerical errors, the second is numerically more stable as is depends more on manual inspection, but the theoretical basis is less stable.
    '''
    
    M,N = np.shape(A)
    rotation = True
    if N != 2 and method=='angles':
        print("Sorry, we only handle matrices of shape Mx2 at this point. Try the 'find_rotation_mat'-method!")
        return False
    if np.shape(A) != np.shape(B):
        print('These matrices are not even of the same shape. They are not rotations of each other!')
        return False
    if method=='find_rotation_mat':
        R_calc=np.linalg.lstsq(A,B)[0]
        determinant = np.linalg.det(R_calc)
        same = np.matmul(R_calc.T, R_calc)-np.eye(np.shape(R_calc)[0])<threshold_R
        if abs(determinant-1)<threshold_R and np.all(same):
            if verbose==True:
                print('These are rotations!')
            return True
        else:
            if verbose==True:
                print('These might not be rotations. det(R)=', determinant, 'R.T*R=\n', np.matmul(R_calc.T,R_calc))
            return False
        
        
        
    elif method=='angles':
        # check lengths of axes and sort by them
        magnitudes_A = []
        magnitudes_B = []

        for i in range(M):
            a = sum([A[i,j]**2 for j in range(N)])
            b = sum([B[i,j]**2 for j in range(N)])
            magnitudes_A.append(a)
            magnitudes_B.append(b)

        order = np.argsort(magnitudes_A)
        magnitudes_A = np.sort(magnitudes_A)
        A = A[order]

        order_B = np.argsort(magnitudes_B)
        magnitudes_B = np.sort(magnitudes_B)
        B = B[order_B]

        for a,b in zip(magnitudes_A,magnitudes_B):
            diff = abs(a-b)
            if diff<threshold_angle*a and diff<threshold_angle*b:
                if verbose:
                    print('These matrices are approximately equal in magnitude on row %i as they are of length %.3f and %.3f (%.3fx).'%(i, a,b, a/b))
            else:
                if verbose:
                    print('These matrices differ in magnitude on row %i as they are of length %.3f and %.3f (%.3fx).'%(i, a,b, a/b))
                    rotation = False


        angles_A = []
        angles_B = []

        angles = []

        for i in range(M):
            for j in range(M):
                if i>=j:
                    continue
                else:
                    angles.append([i,j])
                    angles_A.append(np.arccos(np.dot(A[i],A[j])/(np.linalg.norm(A[i],2)*np.linalg.norm(A[j],2))))
                    angles_B.append(np.arccos(np.dot(B[i],B[j])/(np.linalg.norm(B[i],2)*np.linalg.norm(B[j],2))))

        if order_angles_by_magnitude==False:
            angles_A = np.sort(angles_A)
            angles_B = np.sort(angles_B)

        for i in range(len(angles_A)):
            diff = abs(angles_A[i]-angles_B[i])
            if diff<threshold_angle*angles_A[i] and diff<threshold_angle*angles_B[i]:
                if verbose:
                    print('These matrices are approximately equal on the angle between axis %i and %i as they are %.3f and %.3f.'%(angles[i][0], angles[i][1], angles_A[i], angles_B[i]))
            else:
                if verbose:
                    print('These matrices differ at least on the angle between axis %i and %i as they are %.3f and %.3f.'%(angles[i][0], angles[i][1], angles_A[i], angles_B[i]))
                rotation = False

        if rotation and verbose:              
            print('These matriceas are rotations of each other.')


        return rotation
                  
                  
                  