import math
import random
import numpy as np
import scipy as sp
import scipy.stats
from helper import *

class Problem_data:
    def __init__(self, students = 0, problems = 0):
        self.students = students
        self.problems = problems
        self.problem_names = [ "" for i in range(problems) ]
        self.student_names = [ "" for i in range(students) ]
        self.times = nans(self.students, self.problems)
        self.order = nans(self.students, self.problems)
        self.order_ready = False
        self.valid_times = 0

    def compute_stats(self):
        times_nans = np.isnan(self.times)
        self.problem_solved = self.students - np.sum(times_nans, axis = 0)
        self.student_solved = self.problems - np.sum(times_nans, axis = 1)
        self.valid_times = self.students*self.problems - np.sum(times_nans)
        self.problem_mean_time = np.nansum(self.times, axis = 0) / self.problem_solved
        if self.order_ready:
            self.problem_mean_order = sp.stats.nanmean(self.order, axis = 0)
        else:
            self.problem_mean_order = nans(self.students, self.problems)
        self.time_dev = self.times - np.tile(self.problem_mean_time, (self.students,1) )
        self.student_mean_deviation = sp.stats.nanmean(self.time_dev, axis = 1)
        

    def print_stats(self, verbose = 0):
        print "Students:", self.students
        print "Problems:", self.problems
        print "Valid times data:", self.valid_times, "("+str(int(100*self.valid_times / (self.students * self.problems)))+"%)"
        if verbose:
            print self.student_solved
            print self.student_mean_deviation
            print self.problem_solved
            print self.problem_mean_time

    # predpoklada napocitane stats
    def restrict(self, min_student_solved = 15, min_problem_solved = 30):
        is_problem_ok = np.zeros(self.problems)
        is_student_ok = np.zeros(self.students)
        new_problem_names = []
        for p in range(self.problems):
            if self.problem_solved[p] >= min_problem_solved:
                is_problem_ok[p] = 1
                new_problem_names.append(self.problem_names[p])
        new_student_names = []
        for s in range(self.students):
            if self.student_solved[s] >= min_student_solved:
                is_student_ok[s] = 1
                new_student_names.append(self.student_names[s])
        new_times = nans(len(new_student_names), len(new_problem_names))
        new_order = nans(len(new_student_names), len(new_problem_names))
        news = 0
        for s in range(self.students):
            if is_student_ok[s]:
                newp = 0
                for p in range(self.problems):                    
                    if is_problem_ok[p]:
                        new_times[news][newp] = self.times[s][p]
                        new_order[news][newp] = self.order[s][p]
                        newp += 1
                news += 1
        self.problem_names = new_problem_names
        self.problems = len(self.problem_names)        
        self.student_names = new_student_names
        self.students = len(self.student_names)
        self.times = new_times
        self.order = new_order
        self.compute_stats()

    # to by chtelo zapsat kompaktneji dohromady s restrict
    # predpoklada napocitane stats
    def eliminate_empty_students(self):
        is_student_ok = np.zeros(self.students)
        new_student_names = []
        for s in range(self.students):
            if self.student_solved[s] > 0:
                is_student_ok[s] = 1
                new_student_names.append(self.student_names[s])
        new_times = nans(len(new_student_names), self.problems)
        new_order = nans(len(new_student_names), self.problems)
        news = 0
        for s in range(self.students):
            if is_student_ok[s]:
                for p in range(self.problems):                    
                    new_times[news][p] = self.times[s][p]
                    new_order[news][p] = self.order[s][p]
                news += 1
        self.student_names = new_student_names
        self.students = len(self.student_names)
        self.times = new_times
        self.order = new_order
        self.compute_stats()
            
    def save_to_file(self, filename, black_list = []):
        f = open(filename, "w")
        f.write("Login")
        for pn in self.problem_names:
            f.write(","+pn)
        f.write("\n")
        for s in range(self.students):
            if not self.student_names[s] in black_list:
                f.write(self.student_names[s])
                for p in range(self.problems):
                    f.write(",")
                    if not math.isnan(self.times[s][p]):
                        f.write(str(int(round(math.pow(2,self.times[s][p])))))
                f.write("\n")
        f.close()

########### Rozdelovani dat, ruzne zpusoby #################
        
def split_data(data, prob_set_1 = 0.8):
    data1 = Problem_data(data.students, data.problems)
    data2 = Problem_data(data.students, data.problems)
    for s in range(data.students):
        for p in range(data.problems):
            if data.times[s][p] != np.nan:
                # order nechavam v obou, to je zatim hack kvuli modelu s learning
                data1.order[s][p] = data.order[s][p]
                data2.order[s][p] = data.order[s][p]
                if random.random() < prob_set_1:
                    data1.times[s][p] = data.times[s][p]
                else:
                    data2.times[s][p] = data.times[s][p]
    data1.compute_stats()
    data2.compute_stats()
    return (data1, data2)

# splitovani na train, test, aby test byl vzdy pozdeji nez train
def split_data_using_order(data, test_students = 0.2, test_portion = 0.3):
    train = Problem_data(data.students, data.problems)
    test = Problem_data(data.students, data.problems)
    for s in range(data.students):
        if random.random() < test_students:
            test_s = 1
        else:
            test_s = 0
        for p in range(data.problems):
            if data.times[s][p] != np.nan:
                # !!! order nechavam v obou, to je zatim nechutny! hack kvuli modelu s learning
                test.order[s][p] = data.order[s][p]
                train.order[s][p] = data.order[s][p]
                if test_s and data.order[s][p] > (1-test_portion) * data.student_solved[s]:
                    test.times[s][p] = data.times[s][p]
                else:
                    train.times[s][p] = data.times[s][p]
    train.compute_stats()
    test.compute_stats()
    return (train, test)

# rozhozeni podle order na dve poloviny
def split_data_odd_even(data):
    data1 = Problem_data(data.students, data.problems)
    data2 = Problem_data(data.students, data.problems)
    for s in range(data.students):
        for p in range(data.problems):
            if data.times[s][p] != np.nan:
                # order nechavam v obou, to je zatim hack kvuli modelu s learning
                data1.order[s][p] = data.order[s][p]
                data2.order[s][p] = data.order[s][p]
                if data.order[s][p] % 2 == s % 2:  # podle parity studenta, abych to nemel u vsech stejne
                    data1.times[s][p] = data.times[s][p]
                else:
                    data2.times[s][p] = data.times[s][p]
    data1.compute_stats()
    data2.compute_stats()
    return (data1, data2)

# trochu hack, protoze vola restrict, ktery je jen pro human data
# a taky to restrict vola fixne s default hodnotama
def split_data_by_students(data, test_students = 0.2):
    train = Problem_data(data.students, data.problems)
    test = Problem_data(data.students, data.problems)
    for s in range(data.students):
        if random.random() < test_students:
            test_s = 1
        else:
            test_s = 0
        for p in range(data.problems):
            if data.times[s][p] != np.nan:
                if test_s:
                    test.times[s][p] = data.times[s][p]
                    test.order[s][p] = data.order[s][p]
                else:
                    train.times[s][p] = data.times[s][p]
                    train.order[s][p] = data.order[s][p]
    train.compute_stats()
    test.compute_stats()
    train.eliminate_empty_students()
    test.eliminate_empty_students()
    return (train, test)    

#################### Spojovani dat z vice zdroju, ruzne zpusoby ##############

def join_data(data_list):
    data = Problem_data()    
    global_student_id = {}
    data.students = 0
    data.problems = 0
    data.student_names = []
    data.problem_names = []
    for i in range(len(data_list)):
        for sid in range(data_list[i].students):
            sname = data_list[i].student_names[sid]
            if not global_student_id.has_key(sname):
                global_student_id[sname] = data.students
                data.students += 1
                data.student_names.append(sname)
        data.problem_names +=  data_list[i].problem_names
        data.problems += data_list[i].problems
        
    data.problem_tags = np.zeros(data.problems)
    data.times = nans(data.students, data.problems)
    data.order = nans(data.students, data.problems)
    problem_offset = 0
    for i in range(len(data_list)):
        for p in range(data_list[i].problems):
            for sid in range(data_list[i].students):
                data.times[global_student_id[data_list[i].student_names[sid]],
                           problem_offset + p] = data_list[i].times[sid][p]
            data.problem_tags[problem_offset + p] = i
        problem_offset += data_list[i].problems
    data.compute_stats()
    
    return data

def join_data_common_students(data_list):
    data = Problem_data()    
    data.students = 0
    data.problems = 0
    data.student_names = []
    data.problem_names = []
    student_in_sets = {}
    for i in range(len(data_list)):
        data.problem_names +=  data_list[i].problem_names
        data.problems += data_list[i].problems
        for sname in data_list[i].student_names:
            student_in_sets[sname] = student_in_sets.get(sname,0) + 1 
    global_student_id = {}
    for sname in student_in_sets.keys():
        if student_in_sets[sname] == len(data_list):
            global_student_id[sname] = data.students
            data.students += 1
            data.student_names.append(sname)
                    
    data.problem_tags = np.zeros(data.problems)
    data.times = nans(data.students, data.problems)
    data.order = nans(data.students, data.problems)
    problem_offset = 0
    for i in range(len(data_list)):
        for p in range(data_list[i].problems):
            for sid in range(data_list[i].students):
                sname = data_list[i].student_names[sid]
                if sname in global_student_id:
                    data.times[global_student_id[sname],
                               problem_offset + p] = data_list[i].times[sid][p]
            data.problem_tags[problem_offset + p] = i
        problem_offset += data_list[i].problems
    data.compute_stats()
    
    return data
        

        
