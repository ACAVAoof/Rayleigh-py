#Creates matrix with coupled beam and plate whose determinant will be varied and solved to be 0
#Different methods in optimizing speed

import numpy as np

def construct_matrix(freq, constraints, beam, plate):
    
    dim = len(constraints)
    A = np.zeros((dim, dim))

    for row_indx in range(dim):
        
        for col_indx in range(row_indx, dim):
            
            left = 0
            # Beam calculations
            for n in range(1, beam.indx+1):
                left += (beam.shape(n, constraints[row_indx][0]) * 
                         beam.shape(n, constraints[col_indx][0])) / \
                        ( beam.gen_mass * ( ( -freq**2 ) + ( beam.freqs[n-1] )**2 ) )

            right = 0
            # Plate calculations
            for rx in range(1, plate.x_indx+1):
                for ry in range(1, plate.y_indx+1):
                    right += (plate.shape(rx, ry, constraints[row_indx]) * 
                              plate.shape(rx, ry, constraints[col_indx])) / \
                             ( plate.gen_mass * ( (-freq**2) + ( plate.freqs[rx-1, ry-1] )**2 ) )

            A[row_indx, col_indx] = left + right
            A[col_indx, row_indx] = left + right  # Matrix is symmetric

    return A
            
def construct_matrix_med(freq, constraints, beam, plate):
    
    dim = len(constraints)
    A = np.zeros((dim, dim))

    beam_freq_diff_sq = beam.gen_mass * ((-freq**2) + beam.freqs**2 ) # This is a vector for each frequency
    plate_freq_diff_sq = plate.gen_mass * ((-freq**2) + plate.freqs**2)

    for row_indx in range(dim):
        for col_indx in range(row_indx, dim):

            left = 0
            # Beam calculations
            for n in range(1, beam.indx+1):
                left += (beam.shape(n, constraints[row_indx][0]) * 
                         beam.shape(n, constraints[col_indx][0])) / \
                         beam_freq_diff_sq[n-1]

            right = 0
            # Plate calculations
            for rx in range(1, plate.x_indx+1):
                for ry in range(1, plate.y_indx+1):
                    right += (plate.shape(rx, ry, constraints[row_indx]) * 
                              plate.shape(rx, ry, constraints[col_indx])) / \
                             plate_freq_diff_sq[rx-1,ry-1]

            A[row_indx, col_indx] = left + right
            A[col_indx, row_indx] = left + right  # Matrix is symmetric

    return A


def construct_matrix_maybefast(freq, constraints, beam, plate):
    dim = len(constraints)

    A = np.zeros((dim, dim))

    beam_freq_diff_sq = 1/ (beam.gen_mass * ((-freq**2) + beam.freqs**2 )) # This is a vector for each frequency
    plate_freq_diff_sq = 1/  (plate.gen_mass * ((-freq**2) + plate.freqs**2))
    
    for row_indx in range(dim):
        for col_indx in range(row_indx, dim):

            left = 0
            # Beam calculations
            for n in range(1, beam.indx+1):

                
                left += (beam.constraint_eval[n-1][row_indx] * 
                         beam.constraint_eval[n-1][col_indx]) * \
                         beam_freq_diff_sq[n-1]

            right = 0
            # Plate calculations
            for rx in range(1, plate.x_indx+1):
                for ry in range(1, plate.y_indx+1):
                   
                    
                    right += (plate.constraint_eval[rx-1, ry-1][row_indx] * 
                              plate.constraint_eval[rx-1, ry-1][col_indx]) * \
                             plate_freq_diff_sq[rx-1,ry-1]
                    

            A[row_indx, col_indx] = left + right
            A[col_indx, row_indx] = left + right  # Matrix is symmetric

    return A



def construct_matrix_maybefast_tiny(freq, dim, beam, plate, debug = False):
   

    A = np.zeros((dim, dim))

    beam_freq_diff_sq = 1/ (beam.gen_mass * ((-freq**2) + beam.freqs_sq )) # This is a vector for each frequency
    
    plate_freq_diff_sq = 1/  (plate.gen_mass * ((-freq**2) + plate.freqs_sq))
    
    
    for row_indx in range(dim):
        for col_indx in range(row_indx, dim):

            left = 0
            # Beam calculations
            for n in range(1, beam.indx+1):

                
                left += (beam.constraint_eval[n-1][row_indx] * 
                         beam.constraint_eval[n-1][col_indx]) * \
                         beam_freq_diff_sq[n-1]

            right = 0
            # Plate calculations
            for rx in range(1, plate.x_indx+1):
                for ry in range(1, plate.y_indx+1):
                   
                    
                    right += (plate.constraint_eval[rx-1, ry-1][row_indx] * 
                              plate.constraint_eval[rx-1, ry-1][col_indx]) * \
                             plate_freq_diff_sq[rx-1,ry-1]

            A[row_indx, col_indx] = left + right
            A[col_indx, row_indx] = left + right  # Matrix is symmetric

    return A





def construct_matrix_maybefast_tiny2(freq, constraints, beam, plate):
    dim = len(constraints)

    A = np.zeros((dim, dim))

    beam_freq_diff_sq = 1/  ((beam.gen_mass*(-freq**2) + beam.freqs_sq_mass)) # This is a vector for each frequency
    plate_freq_diff_sq = 1/ ((plate.gen_mass*(-freq**2) + plate.freqs_sq_mass))
    
    for row_indx in range(dim):
        for col_indx in range(row_indx, dim):

            left = 0
            # Beam calculations
            for n in range(1, beam.indx+1):

                
                left += (beam.constraint_eval[n-1][row_indx] * 
                         beam.constraint_eval[n-1][col_indx]) * \
                         beam_freq_diff_sq[n-1]

            right = 0
            # Plate calculations
            for rx in range(1, plate.x_indx+1):
                for ry in range(1, plate.y_indx+1):
                   
                    
                    right += (plate.constraint_eval[rx-1, ry-1][row_indx] * 
                              plate.constraint_eval[rx-1, ry-1][col_indx]) * \
                             plate_freq_diff_sq[rx-1,ry-1]
                    

            A[row_indx, col_indx] = left + right
            A[col_indx, row_indx] = left + right  # Matrix is symmetric

    return A





def construct_matrix_faster(freq, constraints, beam, plate):
    dim = len(constraints)

    A = np.zeros((dim, dim))

    beam_denom = 1 / (beam.gen_mass * ((-freq**2) + beam.freqs**2 )) # This is a vector for each frequency
    plate_denom = 1 / (plate.gen_mass * ((-freq**2) + plate.freqs**2))

    for row_indx in range(dim):
        for col_indx in range(row_indx, dim):

            
            
            left =  beam.constraint_eval[:, row_indx] * beam.constraint_eval[:, col_indx]
            left =  np.sum(np.multiply(left, beam_denom))

            #left = np.sum(np.multiply(beam.constraint_eval[:, row_indx], beam.constraint_eval[:, col_indx]) * beam_denom)
            
            right = plate.constraint_eval[:, :, row_indx] * plate.constraint_eval[:,:, col_indx]
            right = ( right * plate_denom ).sum()

            #right = 0
            # Plate calculations
            #for rx in range(1, plate.x_indx+1):
            #    for ry in range(1, plate.y_indx+1):
            #        right += (plate.constraint_eval[rx-1, ry-1][row_indx] * 
            #                  plate.constraint_eval[rx-1, ry-1][col_indx]) / \
            #                 plate_denom[rx-1,ry-1]
            
            A[row_indx, col_indx] = left + right
            A[col_indx, row_indx] = left + right  # Matrix is symmetric

    return A

def construct_matrix_faster2(freq, constraints, beam, plate):
    dim = len(constraints)

    A = np.zeros((dim, dim))

    beam_denom = 1 / (beam.gen_mass * ((-freq**2) + beam.freqs**2 )) # This is a vector for each frequency
    plate_denom = 1 / (plate.gen_mass * ((-freq**2) + plate.freqs**2))

    for row_indx in range(dim):
        for col_indx in range(row_indx, dim):

       
            
            
            
          
            left =  np.sum(np.multiply(beam.constraint_block[:, row_indx, col_indx], beam_denom))
            
            right = plate.constraint_eval[:, :, row_indx] * plate.constraint_eval[:,:, col_indx]
            right = ( right * plate_denom ).sum()

            #right = 0
            # Plate calculations
            #for rx in range(1, plate.x_indx+1):
            #    for ry in range(1, plate.y_indx+1):
            #        right += (plate.constraint_eval[rx-1, ry-1][row_indx] * 
            #                  plate.constraint_eval[rx-1, ry-1][col_indx]) / \
            #                 plate_denom[rx-1,ry-1]

            A[row_indx, col_indx] = left + right
            A[col_indx, row_indx] = left + right  # Matrix is symmetric

    return A


def construct_matrix_faster3(freq, dim, beam, plate):

    A = np.zeros((dim, dim))

    beam_denom = 1 / (beam.gen_mass * ((-freq**2) + beam.freqs**2 )) # This is a vector for each frequency
    plate_denom = 1 / (plate.gen_mass * ((-freq**2) + plate.freqs**2))

    for row_indx in range(dim):
        for col_indx in range(row_indx, dim):

       
            
            
            
          
            left = np.einsum('i,i->', beam.constraint_block[:, row_indx, col_indx], beam_denom)
            
            #right = plate.constraint_eval[:, :, row_indx] * plate.constraint_eval[:,:, col_indx]
            #right = ( right * plate_denom ).sum()

            #right = 0
            # Plate calculations
            #for rx in range(1, plate.x_indx+1):
            #    for ry in range(1, plate.y_indx+1):
            #        right += (plate.constraint_eval[rx-1, ry-1][row_indx] * 
            #                  plate.constraint_eval[rx-1, ry-1][col_indx]) / \
            #                 plate_denom[rx-1,ry-1]

            #A[row_indx, col_indx] = left + right
            A[row_indx, col_indx] = left 
            #A[col_indx, row_indx] = left + right
            A[col_indx, row_indx] = left
            # Matrix is symmetric

    return A
    
