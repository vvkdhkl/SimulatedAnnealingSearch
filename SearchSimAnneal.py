# coding: utf-8


import sys
from random import randint

#****************************************** functions to model the constraints and the cost functions *******************************

#This function calculates the U(m,r) value ie the used capacity of machine m and resource r in current assignment
def Uval(quickAssign,m,r):
  global init_assign
  sum = 0
  for p in range(init_assign.num_processes):
	  if (quickAssign[p] == m):
		  sum = sum + init_assign.process_requirements[p][r]
  #print(sum)
  return(sum)

#This function checks if the machine capacity constraint (MCCon) is satisfied. This uses U values from the function Uval.
#Returns true if satisfied.
def mccon(quickAssign):
  global init_assign
  #print(len(quickAssign))
  num_res = init_assign.num_resources
  #print(num_res)
  num_mach = init_assign.num_machines
  #print(len(init_assign.assignment))
  U = [[0 for k in range(num_res)] for l in range(num_mach)]
  #print(len(U))
  result = True
  for m in range(num_mach):
	  for r in range(num_res):
		  U[m][r] = Uval(quickAssign,m,r)
		  #print(len(init_assign.machine_capacities))
		  result = result & (U[m][r] <= init_assign.machine_capacities[m][r] )

  #print(len(U))
  return(result, U)

#any two processes of a service need to be assigned to different machines. Returns True is satisfied
def sccon(quickAssign):
  global init_assign
  num_proc = init_assign.num_processes

  num_serv = init_assign.num_services
  result = True
  print('start: sccon')
  for p1 in range(num_proc):
	  for p2 in range(num_proc):
		  if ((init_assign.process_services[p1] == init_assign.process_services[p2]) & (p1!=p2)):
			  result = result & (quickAssign[p1]!=quickAssign[p2])
  print('end: sccon')
  return(result)


# service spread constraint. Returns true is satisfied
def sscon(quickAssign):
  global init_assign
  num_serv = init_assign.num_services
  num_proc = init_assign.num_processes

  result = True
  lenarr = []
  for s in range(num_serv):
      locations = set()
      for p in range(num_proc):
          if (init_assign.process_services[p] == s):
            locations.add(init_assign.machine_locations[quickAssign[p]])
      lenarr.append(len(locations))
      result = result & (len(locations) >= init_assign.service_min_spreads[s])
  return(result)

#for an assignment, returns a set of processes served by a machine with index m
def groupProcByMach(quickAssign, m):
  global init_assign
  procByMach = set()
  for k in quickAssign:
      if (k==m):
        procByMach.add(quickAssign.index(k))
  #print(procByMach)
  return(procByMach)


'''checks changes in the constraint value. I have implemented finding a new solution by changing the assignment
of one process at a time to a new machine. In that case, this method takes in parameters like process switched and
machine switched to and from, and the original U values and the new assignment, and evaluates constraints as function of
the changed variables provided the earlier assignment (from which this is derived) satisfies the constraints.

This is done to reduce computations for every check of constraints. increased the check speed drastically for me.'''

def constChange(p, m1, m2, U, quickAssign):
  #print(m1, m2)

  global init_assign
  result1 = True
  #print(U)
  #check for changed MCCon, and hence satisfaction. We reduce the earlier sum of capacity used by the switched process,
  #and add the capacity to the new machine. The new machine should then satisfy the capacity constraint.

  for r in range(init_assign.num_resources):
      #print('a', U[m1][r], U[m2][r], init_assign.process_requirements[p][r])
      U[m1][r] = U[m1][r]-init_assign.process_requirements[p][r]
      #print(len(U))
      U[m2][r] = U[m2][r]+init_assign.process_requirements[p][r]
      result1 =  result1 & (U[m2][r] <= init_assign.machine_capacities[m2][r])
      #print('b', U[m1][r], U[m2][r])

  #check that the switched process does not belong to the same service as any of the processes running on
  #the machine where it is switched to

  '''bymach = groupProcByMach(quickAssign, m2)
  result2 = True
  #print(bymach)
  for proc in bymach:
      if (p != proc):

        result2 = result2 & (init_assign.process_services[p] != init_assign.process_services[proc])'''
  result2 = sccon(quickAssign)
  print(result2)
  #check service spread. the new switched process should not disturb the spread of service in the original machine's location
  s = init_assign.process_services[p]
  locations = set()
  for p1 in range(init_assign.num_processes):

	  if ((init_assign.process_services[p1] == s) & (p1 != p)):
	    mach = quickAssign[p1]

	    locations.add(init_assign.machine_locations[mach])

  result3 = (len(locations) >= init_assign.service_min_spreads[s])
  #print(result1 & result2 & result3)

  #return the 'and' of all three constraints above
  return(result1 & result2 & result3)

#calculate the cost of a constrained feasible solution.
def cost(quickAssign):
  global init_assign
  num_proc = init_assign.num_processes
  pmcost = sum([int(quickAssign[p] != init_assign.assignment[p])*init_assign.process_moving_costs[p] for p in range(num_proc)])
  mlcost = sum([(max(0, Uval(quickAssign,m,r)-init_assign.soft_machine_capacities[m][r])) for r in range(init_assign.num_resources) for m in range(init_assign.num_machines)])
  return(pmcost+mlcost)


#obtaining new solutions from existing ones. Here I obtain it by switching randomly one process at a time from one machine to another.
#this helps traverse the solution space without disturbing the constraints by big margin, it is expected.

#**************************************** search algortithm: traversing solution space **************************************
def getNewAssign(best):
  global init_assign
  cond = False

  while not cond:

      new_ = best[:]

      index1 = randint(0,init_assign.num_machines-1)
      index2 = randint(0,init_assign.num_processes-1)

      old_mindex = new_[index2]
      new_[index2] = index1
      result, U = mccon(best)
      #print(U)
      cond = constChange(index2, old_mindex, index1, U, new_[:])

  return(new_)

#part of the simulated annealing procedure. acceptance rates for a new feasible solution as the current
#source of neighbourhood.
def acceptance(energy, newEnergy, temp):
  if (newEnergy<energy):
	  return(1)
  return(energy/(newEnergy*temp))

#the simulated annealing procedure. takes in the output file obtained from the main function, and saves the results in it
#as they are obtained.
def simAnneal(outfile):
  global init_assign
  #initialization parameters
  temp = 1000.0
  coolingRate = 0.001
  best = init_assign.assignment

  best_cost = cost(init_assign.assignment)

  curr = best[:]
  print('****************************')
  print('Simulated annealing algorithm in work!\nTemperature:')
  f = open(outfile, 'w')

  while(temp>1):
    #print(curr)
    quickAssign = getNewAssign(curr)
    #print(quickAssign)
    sys.stdout.write('\r%.1f' %temp)
    sys.stdout.flush()

    bestEnergy = cost(best)
    newEnergy = cost(quickAssign)
    if (acceptance(float(bestEnergy), float(newEnergy), temp)>0.001):
	    curr = quickAssign[:]
    curr_cost = cost(quickAssign)
    #print('h',curr_cost)
    if curr_cost<best_cost:
        best = quickAssign[:]
        #write the best solution to the file
        for k in best:
          f.write(str(k)+' ')
        f.write('\n')
        best_cost = curr_cost
        temp = temp*(1+10*coolingRate)
    else:
	    temp = temp*(1-coolingRate)

	#assign.dump_instance(filename=outfile)
    #dump_assignment(assignment.assignment, filename=outfile, mode='a')
  f.close()

#************************************* definition of the ProcessAssignment class ****************************************************

class InstanceError(BaseException):
	pass

class AssignmentError(BaseException):
	pass

class InvalidArgumentException(ValueError):
	pass

class ProcessAssignment:
	"""Stores an instance of the process assignment program."""
	num_resources = 0
	num_machines = 0
	num_processes = 0
	num_services = 0
	num_locations = 0

	# The capacities and the requirements will be presented as lists of
	# tuples. Suppose there are three resources, and two machines. Then
	# the contents of machine_capacities could be [(2, 11, 4), (9, 33, 1)].
	machine_capacities = []
	soft_machine_capacities = []
	process_requirements = []

	# The following are just lists of integers
	machine_locations = []
	service_min_spreads = []
	process_services = []
	process_moving_costs = []

	# This is a list of machines, one for each process
	assignment = None

	def __init__(self, filename=None):
		if filename:
			self._read_instance_file(filename)

	def dump_instance(self, filename=None, mode='w'):
		"""Writes the current instance in human-readable format to
                a given file or stdout."""

		if filename:
			if mode not in ['a', 'w']:
				raise InvalidArgumentException("Allowed modes are 'a' and 'w'")
			f = open(filename, mode)
		else:
			f = sys.stdout

		print("Problem instance:\n", file=f)
		print("  Resources: %4d" % self.num_resources, file=f)
		print("  Machines: %5d" % self.num_machines, file=f)
		print("  Processes: %4d" % self.num_processes, file=f)
		print("  Services: %5d" % self.num_services, file=f)
		print("  Locations: %4d" % self.num_locations, file=f)

		print("\n  Machine Capacities (soft/hard):\n", file=f)

		for machine in range(self.num_machines):
			print("    m: %d" % machine, file=f)
			capacities = [('%d/%d' % t).rjust(10) for t in zip(self.machine_capacities[machine], self.soft_machine_capacities[machine])]
			print("      %s" % "".join(capacities), file=f)

		print("\n  Machine Locations:\n", file=f)

		for machine in range(self.num_machines):
			print("    m: %d" % machine, file=f)
			print("        %4d" % self.machine_locations[machine], file=f)

		print("\n  Minimum Service Spreads:\n", file=f)

		for service in range(self.num_services):
			print("    s: %d" % service, file=f)
			print("        %4d" % self.service_min_spreads[service], file=f)

		print("\n  Process Requirements:\n", file=f)

		for process in range(self.num_processes):
			print("    p: %d" % process, file=f)
			requirements = [('%d' % req).rjust(6) for req in self.process_requirements[process]]
			print("      %s" % "".join(requirements), file=f)

		print("\n  Process Services:\n", file=f)

		for process in range(self.num_processes):
			print("    p: %d" % process, file=f)
			print("        %4d" % self.process_services[process], file=f)

		print("\n  Process Moving Costs:\n", file=f)

		for process in range(self.num_processes):
			print("    p: %d" % process, file=f)
			print("        %4d" % self.process_moving_costs[process], file=f)

		print("", file=f)

		if f is not sys.stdout:
			f.close()

	def _read_instance_file(self, filename):
		"""Parses an instance file and overwrites any saved values
                with new data. Note that only very crude error checking is
                performed here (concerning the values and the formatting of
                the file). Most things will raise an exception, e.g. if the
		values in the file are not integer, or are outside the
                allowed range."""
		with open(filename) as instancefile:
			# The first line contains the number of resources
			self.num_resources = int(instancefile.readline().strip())
			# The second line contains the number of machines
			self.num_machines = int(instancefile.readline().strip())

			if (self.num_resources < 1 or self.num_resources > 10):
				raise InstanceError("The number of resources is not within limits")

			if (self.num_machines < 1 or self.num_machines > 500):
				raise InstanceError("The number of machines is not within limits")

			# Initialize some lists (delete any previous data
                        # that may have been there)
			self.machine_capacities = [0] * self.num_machines
			self.soft_machine_capacities = [0] * self.num_machines
			self.machine_locations = [0] * self.num_machines

			# debug
			print("LOG: read num_resources and num_machines")

			# Next num_machines lines contain the following things:
			# <location> <capacity for resource
                        # i=1...num_resources> <soft capacity for resource
                        # i=1...num_resources>
			for machine in range(self.num_machines):
				# Split the line at each space character to
                                # form a list of values
				tokens = instancefile.readline().strip().split()

				if len(tokens) != (1 + 2 * self.num_resources):
					raise InstanceError("Wrong number of values (expected %d, found %d)" % (1 + 2 * num_resources, len(tokens)))

				# First, read the machine location
				location = int(tokens[0])

				if (location < 0 or location > self.num_machines):
					raise InstanceError("Invalid machine location: %d" % location)

				# Here, we're just updating the number of
                                # locations as we observe more of them.
                                # The assumption is of course that all
                                # locations from 1 to n are in use, otherwise
                                #the number wouldn't technically be accurate.
				if self.num_locations < location + 1:
					self.num_locations = location + 1

				self.machine_locations[machine] = location

				# Then the (hard) capacities
				self.machine_capacities[machine] = tuple([int(t) for t in tokens[1:self.num_resources+1]])
				# ...and the soft capacities
				self.soft_machine_capacities[machine] = tuple([int(t) for t in tokens[self.num_resources+1:]])

				# debug
			print("LOG: read machine locations and capacities")

			# The next line has the number of services
			self.num_services = int(instancefile.readline().strip())

			if (self.num_services < 1 or self.num_services > 2000):
				raise InstanceError("The number of services is not within limits")

			self.service_min_spreads = [0] * self.num_services

			# The next num_services lines contain the minimum
                        # spreads for each service
			for service in range(self.num_services):
				min_spread = int(instancefile.readline().strip())

				if (min_spread < 0 or min_spread > self.num_locations):
					raise InstanceError("Invalid service spread value: %d" % min_spread)

				self.service_min_spreads[service] = min_spread

			# debug
			print("LOG: read service spreads")

			# The next line has the number of processes
			self.num_processes = int(instancefile.readline().strip())

			if (self.num_processes < 1 or self.num_processes > 2000):
				raise InstanceError("The number of processes is not within limits")

			self.process_services = [0] * self.num_processes
			self.process_requirements = [0] * self.num_processes
			self.process_moving_costs = [0] * self.num_processes

			# The next num_processes lines contain the
                        # requirements for each process
			for process in range(self.num_processes):
				# Split the line at each space character to
                                # form a list of values
				tokens = instancefile.readline().strip().split()

				if len(tokens) != (2 + self.num_resources):
					raise InstanceError("Wrong number of values (expected %d, found %d)" % (2 + num_resources, len(tokens)))

				# First, read the service this process
                                # belongs to
				service = int(tokens[0])

				if (service < 0 or service > self.num_services):
					raise InstanceError("Invalid service value for process: %d" % service)

				self.process_services[process] = service

				# Then, read the requirements this process
                                # has for each resource
				self.process_requirements[process] = tuple([int(t) for t in tokens[1:self.num_resources+1]])

				# The last value is the moving cost of the
                                # process
				moving_cost = int(tokens[self.num_resources + 1])

				if (moving_cost < 0 or moving_cost > 1000):
					raise InstanceError("The moving cost is not within limits")

				self.process_moving_costs[process] = moving_cost

			# debug
			print("LOG: read processes")
			print("LOG: finished reading instance")

	def update_assignment(self, filename):
		"""Reads an assignment from a file, overwrites a previous assignment if one existed."""
		with open(filename) as assignmentfile:
			tokens = assignmentfile.readline().strip().split()

			if len(tokens) != self.num_processes:
				raise AssignmentError("Wrong number of assigned processes (expected %d, found %d)" % (self.num_processes, len(tokens)))

			self.assignment = [0] * self.num_processes

			# Parse the machine the process is assigned to
			for process in range(self.num_processes):
				self.assignment[process] = int(tokens[process])
		# debug
		print("LOG: finished reading assignment")

	def quick_update_assignment(self, arr):
		for process in range(self.num_processes):
				self.assignment[process] = arr[process]


def dump_assignment(assignment, filename=None, mode='w'):
	"""Writes an assignment in human-readable format to a given file or
    stdout."""

	if filename:
		if mode not in ['a', 'w']:
			raise InvalidArgumentException("Allowed modes are 'a' and 'w'")
		f = open(filename, mode)
	else:
		f = sys.stdout

	'''print("Assignment (process -> machine):\n", file=f)
	for process in range(len(assignment)):
	    print("%d" % (assignment[process]), file=f)'''
	print(assignment, file=f)
	print("", file=f)
	if f is not sys.stdout:
		f.close()

#***************************************************** main ****************************************************
if __name__ == "__main__":
	if len(sys.argv) not in [3, 4]:
		print("Usage: python processassignment.py <instance_file> <initial_solution_file> [<output_file>]")
	else:
		# Read the instance and the assignment and initialise a new
                # ProcessAssignment object
		try:
			assignment = ProcessAssignment(filename=sys.argv[1])
		except BaseException as e:
			print("Could not initialize a ProcessAssignment.", file=sys.stderr)
			print(repr(e), file=sys.stderr)
			sys.exit(1)

		try:
			assignment.update_assignment(filename=sys.argv[2])
		except BaseException as e:
			print("Could not load the initial assignment.", file=sys.stderr)
			print(repr(e), file=sys.stderr)
			sys.exit(1)


		# Print a representation of the instance and the assignment
                # to the given <output_file>
	if len(sys.argv) == 3:
      # Print to stdout
		outfile = None
	else:
		outfile = sys.argv[3]
	init_assign = assignment

	simAnneal(outfile)
