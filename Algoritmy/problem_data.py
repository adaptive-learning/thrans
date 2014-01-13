import math
import random
import numpy as np
import scipy as sp
import scipy.stats

def nans(x, y):
    out = np.empty((x,y))
    out.fill(np.nan)
    return out

class Problem_data:
    def __init__(self, students = 0, problems = 0):
        self.students = students
        self.problems = problems
        self.times = nans(self.students, self.problems)
        self.order = nans(self.students, self.problems)
        self.valid_times = 0
        self.data_source = "Not loaded"
        
    def compute_stats(self):
        times_nans = np.isnan(self.times)
        self.problem_solved = self.students - np.sum(times_nans, axis = 0)
        self.student_solved = self.problems - np.sum(times_nans, axis = 1)
        self.valid_times = self.students*self.problems - np.sum(times_nans)
        self.problem_mean_time = np.nansum(self.times, axis = 0) / self.problem_solved
        self.problem_mean_order = sp.stats.nanmean(self.order, axis = 0)
        self.time_dev = self.times - np.tile(self.problem_mean_time, (self.students,1) ) # rozdil oproti prumernemu casu
        self.student_mean_deviation = sp.stats.nanmean(self.time_dev, axis = 1)
        
    def print_stats(self, verbose = 0):
        print "Source:", self.data_source
        print "Students:", self.students
        print "Problems:", self.problems
        print "Valid times data:", self.valid_times
        if verbose:
            print self.student_names
            print self.problem_names
            print self.student_solved
            print self.student_mean_deviation
            print self.problem_solved
            print self.problem_mean_time

class Simulated_data(Problem_data):        

    # min_solved = 0 znamena vsechna data
    def __init__(self, students = 0, problems = 0, mode = "12", min_solved = 15):
        Problem_data.__init__(self, students, problems)
        self.mode = mode
        self.data_source = "simulated " + mode
        self.valid_times = students * problems
        self.create_common_data()
        self.create_mode_dependent_data()
        self.create_order()
        self.create_times()
        if min_solved != 0:
            self.create_unsolved(min_solved)
        self.compute_stats()

    #TODO tady mozna nejak lepe uchopit ty "meta-parametry" - udelat to jako samostatnou tridu?
    def create_common_data(self):
        self.sim_skills = np.zeros((3,self.students))
        for i in range(3):
            self.sim_skills[i] = np.random.randn(self.students) * 0.7 # 0.7 podle human data
        self.sim_ppars = np.zeros((3, self.problems))
#        self.sim_ppars[0] = np.random.rand(self.problems)*10 + 2  # B: 2 az 10
        self.sim_ppars[0] = np.random.randn(self.problems)*2 + 7  # B - normalne rozlozeno kolem 7        
#        self.sim_ppars[1] = np.random.rand(self.problems) - 1.5   # A: -1.5 az 0.5
        self.sim_ppars[1] = np.random.randn(self.problems)*0.4 - 1   # A: normalne kolem 1; bylo 0.4
#        self.sim_ppars[2] = np.random.rand(self.problems) - 1.5   # A2: -1.5 az 0.5
        self.sim_ppars[2] = np.random.randn(self.problems)*0.4 - 1   # A2: normalne kolem 1
        # puvodni varinta         
#        self.sim_sd = np.random.rand(self.problems)*0.6 + 0.6   # C: 0.6 az 1.2, uniforme
        # nove delam variance, zvlast pro problem a student... cisla ujasnit
        self.sim_pvar = np.random.rand(self.problems)*1 + 0.1   # C: 0.1 az 1.1, uniforme
        self.sim_svar = np.random.rand(self.students)*1 #+ 0.1   # C: 0.1 az 1.1, uniforme

    def create_mode_dependent_data(self):
        if self.mode == "21L":
            self.sim_skills[1] = np.random.rand(self.students) * 0.6
        if self.mode == "22":
            self.sim_ppars[1] = np.random.randn(self.problems) * 2
        if self.mode == "2typy":
            self.sim_ppars[1] = np.zeros(self.problems)
            self.sim_ppars[2] = np.zeros(self.problems)
            for p in range(self.problems):
                self.sim_ppars[random.randint(1,2)][p] = random.gauss(0,1) - 1.5

    def create_order(self):
        # umoznit, aby poradi korelovalo s obtiznosti uloh?
        for s in range(self.students):
            tmp = range(1, self.problems + 1)
            random.shuffle(tmp)
            for p in range(self.problems):
                self.order[s][p] = tmp[p] 
                
    def create_times(self):
        for s in range(self.students):
            for p in range(self.problems):
                # puvodni varianta: sd = self.sim_sd[p]
                if self.mode == "12":                    
                    sd = math.sqrt(self.sim_pvar[p] + self.sim_ppars[1][p]**2 * self.sim_svar[s])
                else:
                    sd = math.sqrt(self.sim_pvar[p] + self.sim_svar[s])
                self.times[s][p] = random.gauss(self.simulated_mode_mean(s,p),sd)

    def create_unsolved(self, min_solved = 15):
        if min_solved >= self.problems: return
        student_solved = np.random.random_integers(min_solved, self.problems, self.students)
        # todo umoznit aby korelovalo se skillem?
#        student_solved = np.zeros(self.students)
#        for s in range(self.students):
#            student_solved[s] = np.random.random_integers(min_solved, self.problems)
        for s in range(self.students):
            for p in range(self.problems):
                if self.order[s][p] > student_solved[s]:
                    self.times[s][p] = np.nan

    def simulated_mode_mean(self, s, p):
        if self.mode == "11":
            return self.sim_ppars[0][p] - self.sim_skills[0][s]
        elif self.mode == "12":
            return self.sim_ppars[0][p] + self.sim_ppars[1][p] * self.sim_skills[0][s]
        elif self.mode == "22":
            return self.sim_ppars[0][p] - self.sim_skills[0][s] + self.sim_ppars[1][p] * self.sim_skills[1][s]        
        elif self.mode == "2typy":
            return self.sim_ppars[0][p] + self.sim_ppars[1][p] * self.sim_skills[0][s] + self.sim_ppars[2][p] * self.sim_skills[1][s]
        elif self.mode == "21L":
#            return self.sim_ppars[0][p] - self.sim_skills[0][s] - self.sim_skills[1][s] * (self.order[s][p] ** 0.5)
            return self.sim_ppars[0][p] - self.sim_skills[0][s] - self.sim_skills[1][s] * math.log(self.order[s][p] + 1)            
        elif self.mode == "11L":
#            return self.sim_ppars[0][p] - self.sim_skills[0][s] - 0.4 * (self.order[s][p] ** 0.5)
            return self.sim_ppars[0][p] - self.sim_skills[0][s] - 0.4 * math.log(self.order[s][p] + 1)
        print "simulated data: bad mode"
        return 0

    def ideal_prediction_matrix(self):
        pred_times = np.zeros((self.students, self.problems))        
        for s in range(self.students):
            for p in range(self.problems):
                pred_times[s][p] = self.simulated_mode_mean(s,p)
        return pred_times

    def ideal_prediction_rmse(self, test_data = None):
        if test_data == None: test_data = self
        suma = np.nansum((test_data.times - self.ideal_prediction_matrix()) ** 2)
        return math.sqrt( suma / test_data.valid_times)
    
    
class Human_data(Problem_data):
    
    def read_times(self, filename, max_time = 15000, restrict = 1):
        self.data_source = filename
        f = open(filename)
        line = f.readline()
        self.problem_names = line.rstrip().split(",")[1:]
        self.problems = len(self.problem_names)
        all_lines = f.readlines()
        self.students = len(all_lines)
        self.student_names = []
        self.times = nans(self.students, self.problems)
        self.order = nans(self.students, self.problems)
        
        for s in range(self.students):
            tmp = all_lines[s].rstrip().split(",")
            self.student_names.append(tmp[0])
            tmp = tmp[1:]
            for p in range(self.problems):
                if tmp[p] != "" and int(tmp[p]) > 0 and int(tmp[p]) < max_time:
                    self.times[s][p] = math.log(int(tmp[p]),2)
        f.close()
        self.compute_stats()
        if restrict:
            self.restrict()

    # nacitani ze stareho datetime exportu
    def read_order_datetime(self, filename):
        f = open(filename)
        line = f.readline()
        problem_names = line.rstrip().split(",")[1:]
        if problem_names != self.problem_names:
            print "ERROR: PROBLEM NAMES DIFFER"
            print problem_names
            print self.problem_names
            return
        for line in f.readlines():
            tmp = line.rstrip().split(",")
            student = tmp[0]
            if student in self.student_names:
                print student
                sid = self.student_names.index(student)
                tmp = tmp[1:]
                q = []
                for i in range(len(tmp)):
                    if tmp[i] != '': q.append((int(tmp[i]), i))
                q.sort()
                for i in range(len(q)):
                    self.order[sid][q[i][1]] = i
                    print sid, q[i][1], i
        f.close()

    def read_order(self, filename):
        f = open(filename)
        line = f.readline()
        problem_names = line.rstrip().split(",")[1:]
        if problem_names != self.problem_names:
            print "ERROR: PROBLEM NAMES DIFFER"
            print problem_names
            print self.problem_names
            return
        for line in f.readlines():
            tmp = line.rstrip().split(",")
            student = tmp[0]
            if student in self.student_names:
                sid = self.student_names.index(student)
                tmp = tmp[1:]
                for i in range(len(tmp)):
                    if tmp[i] != '':
                        self.order[sid][i] = int(tmp[i])
        np.savetxt("temp.txt", self.order)                    
        f.close()
        
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
            if self.student_solved[s] >= min_student_solved:# and self.student_names[s] != " ": # docasny hack kvuli chybe v datech
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
                
                
def split_data(data, prob_set_1 = 0.8):
    data1 = Problem_data(data.students, data.problems)
    data2 = Problem_data(data.students, data.problems)
    data1.student_names = data.student_names
    data2.student_names = data.student_names
    data1.problem_names = data.problem_names
    data2.problem_names = data.problem_names
    data1.data_source = "subset of "+data.data_source
    data2.data_source = "subset of "+data.data_source
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
    
def join_data(data_list):
    data = Problem_data()
    global_student_id = {}
    data.students = 0
    data.problems = 0
    data.data_source = "join of"
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
        data.data_source += " " + data_list[i].data_source 
        
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
