import numpy as np
import random
import time
np.set_printoptions(precision=9)

class tactics():
    def f0(self, indata:np):# roll

        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3))
            rseed = np.random.random(1)
            if(rseed<=0.2):
                np.roll(ins,random.randint(0,10),axis=0)
            elif(rseed<=0.4):
                np.roll(ins,random.randint(0,10),axis=1)
            elif(rseed<=0.8):
                np.roll(ins,random.randint(0,10),axis=2)
            else:
                np.roll(ins,1,axis=3)
            return np.reshape(ins,(128,3072))

    def f1(self, indata:np):# loss
        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3))
            rseed =  np.random.random(1)
            rloc = random.randint(0,10)
            fac = np.ones_like(ins,dtype=ins.dtype)

            delta = np.random.random(1)
            if(rseed<=0.33):
                fac[rloc][:][0][0]= delta
            elif(rseed<=0.67):
                fac[rloc][0][:][0]= delta
            else:
                fac[rloc][0][0][:]= delta

            ins = ins*fac
            return np.reshape(ins,(128,3072))

    def f2(self, indata:np):# enhance
        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3))
            rseed =  np.random.random(1)
            rloc = random.randint(0,10)
            fac = np.ones_like(ins,dtype=ins.dtype)
            delta = np.array(random.randint(2,3)/1.9)
            if(rseed<=0.33):
                fac[rloc][:][0][0]=delta
            elif(rseed<=0.67):
                fac[rloc][0][:][0]=delta
            else:
                fac[rloc][0][0][:]=delta

            ins = ins*fac
            return np.reshape(ins,(128,3072))
    def f3(self, indata:np):# add normal noise
        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3))
            fac = np.random.normal(0, 0.01, size=ins.shape).astype(ins.dtype)
            ins = ins+fac
            return np.reshape(ins,(128,3072))

class generaltactics():
    def f0(self, indata:np,order:str):# roll

        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3),order=order)
            rseed = np.random.random(1)
            if(rseed<=0.2):
                np.roll(ins,random.randint(0,4),axis=0)
            elif(rseed<=0.4):
                np.roll(ins,random.randint(0,4),axis=1)
            elif(rseed<=0.8):
                np.roll(ins,random.randint(0,4),axis=2)
            else:
                np.roll(ins,1,axis=3)
            return np.reshape(ins,(128,3072),order=order)
        else:
            Originshape = indata.shape
            ins = indata.flatten(order=order)
            rseed = np.random.random(1)
            if(rseed<=0.2):
                np.roll(ins,1)
            elif(rseed<=0.4):
                np.roll(ins,2)
            elif(rseed<=0.8):
                np.roll(ins,3)
            else:
                np.roll(ins,4)
            return np.reshape(ins,Originshape,order=order)

    def f1(self, indata:np,order:str):# loss
        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3),order=order)
            rseed =  np.random.random(1)
            rloc = random.randint(0,10)
            fac = np.ones_like(ins,dtype=ins.dtype)

            delta = np.random.random(1)
            if(rseed<=0.33):
                fac[rloc][:][0][0]= delta
            elif(rseed<=0.67):
                fac[rloc][0][:][0]= delta
            else:
                fac[rloc][0][0][:]= delta

            ins = ins*fac
            return np.reshape(ins,(128,3072),order=order)
        else:
            Originshape = indata.shape
            ins = indata.flatten(order=order)

            datalen = ins.shape[0]
            if (datalen==0):
                print('input too short [error]')
                exit()
            len1 = int(datalen/10+1)
            len2 = int(len1/10 +1)
            len3 = int(len2/10 +1)
            fac = np.ones_like(ins,dtype=ins.dtype)
            delta = np.random.random(1)
            rseed = np.random.random(1)
            if(rseed<=0.2):
                fac[0:len1]=delta
            elif(rseed<=0.8):
                fac[0:len2]=delta
            else:
                fac[0:len3]=delta
            ins = ins*fac
            return np.reshape(ins,Originshape,order=order)

    def f2(self, indata:np,order:str):# enhance
        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3),order=order)
            rseed =  np.random.random(1)
            rloc = random.randint(0,10)
            fac = np.ones_like(ins,dtype=ins.dtype)
            delta = np.array(random.randint(2,3)/1.9)
            if(rseed<=0.33):
                fac[rloc][:][0][0]=delta
            elif(rseed<=0.67):
                fac[rloc][0][:][0]=delta
            else:
                fac[rloc][0][0][:]=delta

            ins = ins*fac
            return np.reshape(ins,(128,3072),order=order)
        else:

            Originshape = indata.shape
            ins = indata.flatten(order=order)
            datalen = ins.shape[0]
            len1 = int(datalen/10+1)
            len2 = int(len1/10 +1)
            len3 = int(len2/10 +1)

            fac = np.ones_like(ins,dtype=ins.dtype)
            delta = np.array(random.randint(2,3)/1.9)
            rseed = np.random.random(1)
            if(rseed<=0.2):
                fac[0:len1]=delta
            elif(rseed<=0.8):
                fac[0:len2]=delta
            else:
                fac[0:len3]=delta
            ins = ins*fac
            return np.reshape(ins,Originshape,order=order)
    def f3(self, indata:np,order:str):# add uniform noise
        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3),order=order)
            fac = np.random.normal(0, 0.01, size=ins.shape).astype(ins.dtype)
            ins = ins+fac
            return np.reshape(ins,(128,3072),order=order)
        else:
            Originshape = indata.shape
            inmean = np.mean(indata)
            ins = indata.flatten(order=order)
            datalen = ins.shape[0]
            len1 = int(datalen/10+1)
            len2 = int(len1/10 +1)
            len3 = int(len2/10 +1)

            noise = np.zeros_like(ins,dtype=ins.dtype)
            delta = np.random.uniform(low=inmean/100,high=inmean/10,size=ins.shape)
            rseed = np.random.random(1)
            if(rseed<=0.2):
                noise[0:len1]=delta[0:len1]
            elif(rseed<=0.8):
                noise[0:len2]=delta[0:len2]
            else:
                noise[0:len3]=delta[0:len3]
            ins = ins+noise
            return np.reshape(ins,Originshape,order=order)
    def f4(self, indata:np,order:str):# sub uniform noise
        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3),order=order)
            fac = np.random.normal(0, 0.01, size=ins.shape).astype(ins.dtype)
            ins = ins+fac
            return np.reshape(ins,(128,3072),order=order)
        else:
            Originshape = indata.shape
            inmean = np.mean(indata)
            ins = indata.flatten(order=order)
            datalen = ins.shape[0]
            len1 = int(datalen/10+1)
            len2 = int(len1/10 +1)
            len3 = int(len2/10 +1)

            noise = np.zeros_like(ins,dtype=ins.dtype)
            delta = np.random.uniform(low=inmean/100,high=inmean/10,size=ins.shape)
            rseed = np.random.random(1)
            if(rseed<=0.2):
                noise[0:len1]=delta[0:len1]
            elif(rseed<=0.8):
                noise[0:len2]=delta[0:len2]
            else:
                noise[0:len3]=delta[0:len3]
            ins = ins-noise
            return np.reshape(ins,Originshape,order=order)
    def f5(self, indata:np,order:str):# single pix change
        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3),order=order)
            fac = np.random.normal(0, 0.01, size=ins.shape).astype(ins.dtype)
            ins = ins+fac
            return np.reshape(ins,(128,3072,),order=order)
        else:
            Originshape = indata.shape
            inmean = np.mean(indata)
            ins = indata.flatten(order=order)
            datalen = ins.shape[0]

            rseed = np.random.random(1)*inmean

            ins[0] = ins[0]+rseed
            np.roll(ins,1,axis=0)
            return np.reshape(ins,Originshape,order=order)


import numpy as np

def simpleDE(fobj ,x0, bounds, dtype, mut=0.8, crossp=0.7, popsize=50, its=20,
             callback = True, normalflag = None,):
    dimensions = len(bounds)
    if normalflag :
        pop = np.random.normal(0,0.1,size=(popsize, dimensions)).astype(dtype)

    pop = np.random.rand(popsize, dimensions).astype(dtype)
    h = np.abs(np.random.standard_cauchy(dimensions).astype(dtype))
    pop[0:5] = h/np.max(h)+0.5
    h = np.abs(np.random.normal(0,0.2,dimensions))
    pop[5:10] = h/np.max(h)+0.5
    min_b, max_b = np.asarray(bounds).T
    min_b = min_b.astype(dtype)
    max_b = max_b.astype(dtype)
    diff = np.fabs(min_b - max_b).astype(dtype)
    pop_denorm = min_b.astype(dtype) + pop * diff
    pop_denorm[11] =x0
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
            # if callback is not None:
            #     print('at x, fn is :',trial_denorm,f,'\n')
        yield best, fitness[best_idx]
        # return x ,y


def dual_annealing():
    pass
    # equations from https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html#rbaa258a99356-3
    """
        def visiting(self, x, step, temperature):
        # Based on the step in the strategy chain, new coordinated are
        # generated by changing all components is the same time or only
        # one of them, the new values are computed with visit_fn method
        #
        dim = x.size
        if step < dim:
            # Changing all coordinates with a new visiting value
            visits = self.visit_fn(temperature, dim)
            upper_sample, lower_sample = self.rand_gen.uniform(size=2)
            visits[visits > self.TAIL_LIMIT] = self.TAIL_LIMIT * upper_sample
            visits[visits < -self.TAIL_LIMIT] = -self.TAIL_LIMIT * lower_sample
            x_visit = visits + x
            a = x_visit - self.lower
            b = np.fmod(a, self.bound_range) + self.bound_range
            x_visit = np.fmod(b, self.bound_range) + self.lower
            x_visit[np.fabs(
                x_visit - self.lower) < self.MIN_VISIT_BOUND] += 1.e-10
        else:
            # Changing only one coordinate at a time based on strategy
            # chain step
            x_visit = np.copy(x)
            visit = self.visit_fn(temperature, 1)
            if visit > self.TAIL_LIMIT:
                visit = self.TAIL_LIMIT * self.rand_gen.uniform()
            elif visit < -self.TAIL_LIMIT:
                visit = -self.TAIL_LIMIT * self.rand_gen.uniform()
            index = step - dim
            x_visit[index] = visit + x[index]
            a = x_visit[index] - self.lower[index]
            b = np.fmod(a, self.bound_range[index]) + self.bound_range[index]
            x_visit[index] = np.fmod(b, self.bound_range[
                index]) + self.lower[index]
            if np.fabs(x_visit[index] - self.lower[
                    index]) < self.MIN_VISIT_BOUND:
                x_visit[index] += self.MIN_VISIT_BOUND
        return x_visit

    def visit_fn(self, temperature, dim):
        x, y = self.rand_gen.normal(size=(dim, 2)).T
        # Formula Visita from p. 405 of reference [2]
        factor1 = np.exp(np.log(temperature) / (self._visiting_param - 1.0))
        factor4 = self._factor4_p * factor1

        # sigmax
        x *= np.exp(-(self._visiting_param - 1.0) * np.log(
            self._factor6 / factor4) / (3.0 - self._visiting_param))

        den = np.exp((self._visiting_param - 1.0) * np.log(np.fabs(y)) /
                     (3.0 - self._visiting_param))

        return x / den
    """

from numpy.random import randn,rand,randint
# simulated annealing algorithm
def simulated_annealing(objective, bounds, its=150, x0=None, step_size=0.1, temp=20):
    # generate an initial point
    bounds = np.array(bounds)
    if x0 is not None:
        best = x0
        x0 = None
    else:
        best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    xshape = np.array(best).shape
    # evaluate the initial point
    best_eval = objective(best)
    # current working solution
    curr, curr_eval = best, best_eval
    # run the algorithm
    for i in range(its):
        # take a step
        candidate = curr + randn(len(bounds)) * step_size
        candidate = np.clip(candidate,np.reshape(bounds[:, 0],xshape),np.reshape(bounds[:,1],xshape))
        # evaluate candidate point
        candidate_eval = objective(candidate)
        # check for new best solution
        if candidate_eval < best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # report progress
            # print('>>>%d f(%s) = %.5f' % (i, best, best_eval))
        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = np.exp(-diff / t)
        # check if we should keep the new point
        if diff < 0 or rand() < metropolis:
            curr, curr_eval = candidate, candidate_eval
    yield best, best_eval


# genetic algorithm search for continuous function optimization
from numpy.random import randint, rand


# decode bitstring to numbers
def decode1(bounds, n_bits, bitstring, type):
    decoded = list()
    largest = 2**n_bits
    lengths = len(bounds)
    for i in range(lengths):
        # extract the substring
        start, end = i * n_bits, (i * n_bits)+n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value.astype(type))
    return decoded

# decode bitstring to numbers
def decode2(bounds, n_bits, bitstring, type):
    decoded = list()
    largest = 2**n_bits
    lengths = len(bounds)
    for i in range(lengths):
        # extract the substring
        start, end = i * n_bits, (i * n_bits)+n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range

        # store
        decoded.append(integer)
    return decoded

def encode(bounds, n_bits, value):
    largest = 2**n_bits
    bitstring = []
    for i in range(len(bounds)):
        integer = int((value[i] - bounds[i][0])/(bounds[i][1] - bounds[i][0]) * largest)
        chars = bin(integer).replace('0b','')
        lens = len(chars)
        chars = (n_bits-lens)*'0'+chars
        bitstring += [int(c) for c in chars]
    return bitstring

# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop)-1, size=k):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

def crossover2(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]
def pmutation(bitstring, r_mut):
    flipflag = rand(len(bitstring))
    mut = np.array([r_mut]*len(bitstring))
    bitmat = np.array(bitstring)
    changeloc = flipflag < mut
    bitstring = ((1-bitmat)*np.equal(changeloc, 1)+bitmat*np.equal(changeloc, 0)).tolist()
def fmutation(bitstring, r_mut, T):
    T = int(T)
    for i in range(2):#int(T/50+1)
        index = randint(11,len(bitstring)-1)   #10-T/10:0->10
        if rand() < r_mut*len(bitstring)*2:
            bitstring[index] = 1 - bitstring[index]

# genetic algorithm
def genetic_algorithm(objective,x0, bounds, r_mut, type, n_bits=16, its=60, n_pop=60, r_cross=0.1, intflag=None):
    if intflag is None:
        decode = decode1
    else:
        decode = decode2
    changeflag = False
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
    lendict = len(x0) if len(x0)<5 else 5
    for i in range(lendict):
        for k in range(1):
            pop[6*k+i] = encode(bounds, n_bits, x0[i])
    # keep track of best solution
    best, best_eval = x0[0], objective(x0[0].astype(type)) # decode(bounds, n_bits, pop[0])
    children = list()
    # enumerate generations
    for gen in range(its):
        t0 = time.time()
        # decode population
        decoded = [decode(bounds, n_bits, p, type) for p in pop]
        # evaluate all candidates in the population
        scores = [objective(d) for d in decoded]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                changeflag = True
                best, best_eval = pop[i], scores[i]
                print(">%d, new bkest f(x)) = %.9f" % (gen,  scores[i]))
        # select parents
        selected = [selection(pop, scores, i) for i in range(n_pop)]
        # create the next generation
        children.clear()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                fmutation(c, r_mut, float(its-gen)/its*100)
                # pmutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
        # print('one me time', time.time()-t0)
    best = best if not changeflag else decode(bounds, n_bits, best, type)
    yield best , best_eval
