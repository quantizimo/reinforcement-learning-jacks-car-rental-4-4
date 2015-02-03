# Solution to Example 4.4 Jack's Car Rental in Reinforcement Learning by Sutton an Barto
# 
# Written by David Anisman, Feb 2015
#
# Comments and questions : quantizimo@gmail.com
#
#
# This module generates an array of transition probabilities from the morning state
# of each of the two locations to the evening state after cars have been rented out
# and returned, each according to the probabilistic dynamics. It also generates a 
# array of expected number of cars rented out for each location
# The arrays are computed using a Monte Carlo simulation. This allows to generate
# different transition probabilities and expected rewards based on different number
# of samples (ie, days) which can give us an indication of the sensitivity of the 
# optimal policy to the sample size. I expect this to be useful in real applications.

# To reproduce these results, this module needs to be run before solving the problem. 
# It saves the resulting arrays in a file. This file and so the arrays are read by the solver.

# Code is written for readability and for the most part follows Google's Style Guide for R


muList = list("req1"= 3, "req2"= 4, "ret1"= 3, "ret2"= 2)
maxCars = 20
maxTransfer = 5
numSamples = 20000
maxCarsPlusOne = maxCars + 1

dimVector = rep(maxCarsPlusOne, 4)

# generate samples
req1 = rpois(numSamples, muList$req1)
ret1 = rpois(numSamples, muList$ret1)

req2 = rpois(numSamples, muList$req2)
ret2 = rpois(numSamples, muList$ret2)

transitionProbs = array(0, dim=dimVector)
R = array(0, dim=dimVector)

matchCount = array(0, dimVector)
cumReq = array(0, dimVector)

for (i in 1:maxCarsPlusOne) {
  for (j in 1:maxCarsPlusOne) {
        
        eveNext1 = 0
        eveNext2 = 0
        
        morning1 = i - 1
        morning2 = j - 1
        
        for (sample in 1:numSamples){
          
          eveNext1 = min(morning1 + ret1[sample] - min(req1[sample], morning1), maxCars)
          
          eveNext2 = min(morning2 + ret2[sample] - min(req2[sample], morning2), maxCars)
          
          matchCount[i,j,eveNext1+1, eveNext2+1] = matchCount[i,j,eveNext1+1, eveNext2+1] + 1
            
          cumReq[i,j,eveNext1+1, eveNext2+1] = cumReq[i,j,eveNext1+1, eveNext2+1] + min(req1[sample], morning1) + min(req2[sample], morning2) 
        }
        print(paste(i,j))
  }
}


for (i in 1:maxCarsPlusOne) {
  
  for (j in 1:maxCarsPlusOne) {
    
    for (k in 1:maxCarsPlusOne) {

      for (l in 1:maxCarsPlusOne) {
        
        transitionProbs[i,j,k,l] = matchCount[i,j,k,l]/numSamples
        
        if (matchCount[i,j,k,l] > 0) {
          
          R[i,j,k,l] = cumReq[i,j,k,l]/matchCount[i,j,k,l]
          
        }else {
          
          R[i,j,k,l] = 0
          
        }
      }
    }
  }
}


save(transitionProbs, R, file="transition-probs-rewards.dat")















