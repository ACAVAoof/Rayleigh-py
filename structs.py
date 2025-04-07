import numpy as np


class Beam:

    def __init__(self, L, E, den, h, w, BC, indx):

        self.L   = L
        self.E   = E
        self.h   = h
        self.w   = w
        self.den = den
        self.BC  = BC
        self.indx = indx

        self.m = self.den * self.h * self.w ##mass per unit length

        self.I = self.w * self.h**3 / 12 ##moment of area

        

        if self.BC == 'PP':
            self.gen_mass = self.L * self.m / 2

        else:
            print('An error occurred, unknown BC Beam condition.')
            raise Exception('An error occurred, unknown BC Beam condition.')

        self.freqs = self.calc_first_freqs(self.indx)
        self.freqs_sq = self.freqs**2
        self.freqs_sq_mass = self.freqs**2 * self.gen_mass

        self.freq_prefactor = np.sqrt(self.E * self.I / self.m) * np.pi**2 / self.L**2

        

    def shape(self, n, x):
        if self.BC == 'PP':
            return np.sin( n * np.pi * x / self.L )
        
    def freq(self, n):
        if self.BC == 'PP':
            return np.sqrt( self.E * self.I / self.m ) * ( n * np.pi / self.L )**2

    def calc_first_freqs(self, indx):
        freqs = np.zeros((indx))
        for n in range(1, indx+1):
            freqs[n-1] = self.freq(n)

        return freqs

    def constraint_shapes(self, constraints):
        shapes = np.zeros( ( self.indx, len(constraints) ) ) 
                          

        for n in range(1, self.indx+1):
            for indx, constraint in enumerate(constraints):
                shapes[n-1][indx] = self.shape(n, constraint[0])
        return shapes

    def make_constraint_block(self, constraints):
        constraint_block = np.zeros( ( self.indx, len(constraints), len(constraints) ) ) 

        for row_indx in range(len(constraints)):
            for col_indx in range(len(constraints)):
                #print(self.constraint_eval)
                constraint_block[:, row_indx, col_indx] = self.constraint_eval[:, row_indx] * self.constraint_eval[:, col_indx]
        #print(constraint_block)
        return constraint_block


class Plate:

    def __init__(self, a, b, E, den, h, nu, BC, x_indx, y_indx):

        self.a = a ##length of the plate in the x direction
        self.b = b ##length of the plate in the y direction
        self.E = E
        self.den = den
        self.h = h
        self.BC = BC
        self.x_indx = x_indx
        self.y_indx = y_indx

        self.m = self.den * self.h ##mass per unit area

        self.D = E * h**3 / ( 12 * (1 - nu**2) )

        self.freq_prefactor = self.D * np.pi**4 / self.m

        if self.BC == 'PPPP':
            self.gen_mass = self.m * a * b / 4
        else:
            print('An error occurred, unknown BC Plate condition.')
            raise Exception('An error occurred, unknown BC Plate condition.')

        self.freqs = self.calc_first_freqs(x_indx, y_indx)
        self.freqs_sq = self.freqs**2
        self.freqs_sq_mass = self.freqs_sq * self.gen_mass

    def shape(self, rx, ry, position):
        if self.BC == 'PPPP':
            return np.sin( rx * np.pi * position[0] / self.a) * np.sin( ry * np.pi * position[1] / self.b )


    def freq(self, rx, ry):
        return np.sqrt(self.D /self.m) * ( (rx * np.pi / self.a)**2 + (ry * np.pi / self.b)**2 )

    def calc_first_freqs(self, x_indx, y_indx):
        
        freqs = np.zeros( (x_indx, y_indx) )
        
        for rx in range(1, x_indx + 1):
            for ry in range(1, y_indx + 1):
                freqs[rx-1, ry-1] = self.freq(rx, ry)

        return freqs

    def constraint_shapes(self, constraints):
        shapes = np.zeros((self.x_indx, self.y_indx, len(constraints)))
 
        for rx in range(1, self.x_indx+1):
            for ry in range(1, self.y_indx+1):
                for indx, constraint in enumerate(constraints):

                    shapes[rx-1, ry-1, indx] = self.shape(rx, ry, constraint)
        return shapes
                
        
            
