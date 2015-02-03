# Solution to Example 4.4 Jack's Car Rental in Reinforcement Learning by Sutton an Barto
# 
# Written by David Anisman, Feb 2015
#
# Comments and questions : quantizimo@gmail.com
#
# Solver for the example using in-place Policy Iteration.
#
# Uses the transition probability and expected average cars previously generated
#
# Code is written for readability and for the most part follows Google's Style Guide for R
# 
# the results are in the V (Value) and P (policy) arrays
#
# The module uses a slightly modified version of the plotting routine found here: http://www.phaget4.org/R/image_matrix.html



maxCars = 20
maxTransfer = 5
reward = 10
transferCost = 2 
discount = 0.9
theta = 0.1  # used to stop the policy evaluation iteration

V = array(0, dim=c(maxCars+1, maxCars+1))  # the Value array
P = array(0, dim=c(maxCars+1, maxCars+1))  # the policy array
employeeP = array(0, dim=c(maxCars+1, maxCars+1))  # the employee policy array

transitionProbs = 0
R = 0
load(file="transition-probs-rewards.dat")   # load precalcuated transition probabilities and average rented cars


# perform one step of the iterative policy evaluation
CalcBellman = function (loc1From, loc2From, currPolicy, currEmployeePolicy, maxCars, VV, R, transitionProbs, discount, reward, transferCost) {
  
  curr1 = loc1From - 1   # indices for the problem are zero-based
  curr2 = loc2From - 1
  
  # if the number of transferred cars is incompatible with the either of the current states
  # then only move the maximum number of cars that is actually compatible with the current states
  # the loop counts down to zero from the current policy stopping once a policy is found that satisfies
  # the constraints
  for (i in currPolicy:0){
    
    if (((curr1 - i) >= 0) && ((curr2 + i) <= maxCars) && ((curr1 - i) <= maxCars) && ((curr2 + i) >= 0)){
      
      constrainedPolicy = i
      break
    }
  }
  
  newValue = -abs(constrainedPolicy) * transferCost  # initialize with the transfer cost just like the original
  
  # but if there is at least one car going to the second location, and the employee policy is asserted, 
  # then one can be taken by the employee at no cost
  
  if ((constrainedPolicy > 0) && (currEmployeePolicy == 1)){
    
    newValue = newValue + 1 * transferCost  
  }
  
  # the number of cars at the start of each day after transferring any cars
  morning1 = loc1From - constrainedPolicy
  morning2 = loc2From + constrainedPolicy
  
  # if more than 10 cars are kept over night at a location (after transferring any cars) then an additional $4 parking
  # cost is incurred
  
  if (morning1 > 10){
    
    newValue = newValue - 4 
  }
  
  if (morning2 > 10){
    
    newValue = newValue - 4
  }
  
  
  for (loc1To in 1:(maxCars+1)) {
    for (loc2To in 1:(maxCars+1)) {
      
      transitionProb = transitionProbs[morning1, morning2, loc1To, loc2To]
      
      expectedReward = reward * R[morning1, morning2, loc1To, loc2To] 
      
      newValue = newValue + transitionProb * (expectedReward + discount * VV[loc1To, loc2To]) 
    }
  }
 
return(newValue)

}


# A full backup of policy iteration until convergence
PolicyEvaluation = function (P, employeeP, maxCars, V, R, transitionProbs, discount, reward, transferCost, theta) {
  
  delta = 0
  flag = TRUE
  maxIters = 100
  numIters = 0
  
  while (flag && (numIters < maxIters)) {
    
    numIters = numIters + 1
    
    saveV = V   # save V
    
    # iterate over all states
    for (loc1From in 1:(maxCars+1)) {
      for (loc2From in 1:(maxCars+1)) {
        
        currPolicy = P[loc1From, loc2From]
        
        currEmployeePolicy = employeeP[loc1From, loc2From]
        
        V[loc1From, loc2From] = CalcBellman(loc1From, loc2From, currPolicy, currEmployeePolicy, maxCars, V, R, transitionProbs, discount, reward, transferCost)
      }
    }
    
    maxConvergenceError = max(abs(saveV - V))
    
    print(paste0("max convergence error: ", maxConvergenceError))
    if (maxConvergenceError < theta) {
      flag = FALSE
    }    
  }
  
  return(V)
}


# for each action and state, calculate the value function and then choose the best, updating the policy
PolicyImprovement = function (P, employeeP, actions, employeeActions, V, R, transitionProbs, discount, reward, transferCost, maxCars) {
  
  policyStable = TRUE
  
  qVal = array(0, dim = c(length(actions), length(employeeActions)))  # an array of action values combining action and employee action
    
  for (loc1From in 1:(maxCars+1)) {
    for (loc2From in 1:(maxCars+1)) {
      
      savePolicy = P[loc1From, loc2From]
      
      saveEmployeePolicy = employeeP[loc1From, loc2From]
      
      for (action in actions) {
        
        for (empAction in employeeActions){
          
          # there is no value in having an employee move a car when the main policy contradicts that
          if (!(empAction == 1  && action <= 0)){
            
            currPolicy = action
            actionValue = CalcBellman(loc1From, loc2From, currPolicy, empAction, maxCars, V, R, transitionProbs, discount, reward, transferCost)
            
            qVal[which(action == actions), which(empAction == employeeActions)] = actionValue 
          }
        }
      }
      
      matchIndex = which(max(qVal) == qVal, arr.ind = TRUE)[1,]
      
      P[loc1From, loc2From] = actions[matchIndex[1]]
      employeeP[loc1From, loc2From] = employeeActions[matchIndex[2]]
      
      if ((P[loc1From, loc2From] != savePolicy) || (employeeP[loc1From, loc2From] != saveEmployeePolicy))  {
        
        policyStable = FALSE
      }
    }
  }
  
  result = list()
  result$P = P
  result$employeeP = employeeP
  result$policyStable = policyStable
  
  return (result)

}


# main loop

policyStable = FALSE
numPolicyIters = 0

while (policyStable == FALSE) {
  
  # Policy evaluation
  V = PolicyEvaluation(P, employeeP, maxCars, V, R, transitionProbs, discount, reward, transferCost, theta)

  myImagePlot(V)
  
  # Policy improvement
  
  # the order of actions is important. In case of equal values, the one with the lowest cars
  # moved will be selected
  actions = c(0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5)
  
  employeeActions = c(0,1)

  result = PolicyImprovement(P, employeeP, actions, employeeActions, V, R, transitionProbs, discount, reward, transferCost, maxCars)

  P = result$P
  
  employeeP = result$employeeP
  
  policyStable = result$policyStable

  myImagePlot(P)

  numPolicyIters = numPolicyIters + 1
}



myImagePlot((employeeP))




