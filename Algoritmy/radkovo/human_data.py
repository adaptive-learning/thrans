from problem_data import *

# s datetime pracuju trochu ad hoc...

DIR = "filtrovanadata"
#ALL_PROBLEMS = [ "Sokoban", "Nurikabe","Binary","Kulicka","Robotanik","Rozdelovacka","Rushhour","Ploty"]
ALL_PROBLEMS = [ "Sokoban", "Nurikabe","Robotanik","Rushhour","Ploty","Kalkulacka", "Binary", "Kulicka", "Rozdelovacka"]
ALL_PROBLEMS.sort()

def set_dir(new_dir):
    global DIR
    DIR = new_dir

def read_human_data(filename, filename_order=None, filename_datetime=None, verbose = 1, min_student_solved = 15, min_problem_solved = 30):
    data = Human_data()
    data.read_times(filename, restrict = 0)
    if filename_order != None:
        data.read_order(filename_order)
    data.restrict(min_student_solved, min_problem_solved)
    if filename_datetime != None:
        data.read_datetime(filename_datetime)   #trochu hack, protoze restrict() nezohlednuje datetime...
    if verbose: data.print_stats()
    return data

def read_problem(problem_name, min_student_solved = 15, min_problem_solved = 30, ordering = 1, datetime = 0):
    filename1 = DIR + "/"+problem_name +"_user_time.csv"
    if ordering:
        filename_order = DIR + "/"+problem_name +"_user_time_ordering.csv"
    else:
        filename_order = None
    if datetime:
        filename_datetime = DIR + "/"+problem_name +"_user_time_datetime.csv"
    else:
        filename_datetime = None
    return read_human_data(filename1, filename_order, filename_datetime, 0, min_student_solved, min_problem_solved)

def find_full_name(problem_name, pid):
    filename = DIR + "/" + problem_name + "_mapping.txt"
    try:
        f = open(filename)
        for l in f.readlines():
            p = l.rstrip().split(';')
            if p[0] == pid:
                return p[1]
        return "unknown"
    except:
        return "unknown (no mapping file)"
        

class Human_data(Problem_data):
    
    def read_times(self, filename, max_time = 15000, restrict = 1):
        f = open(filename)
        line = f.readline()
        self.problem_names = line.rstrip().split(",")[1:]
        self.problems = len(self.problem_names)
        all_lines = f.readlines()
        self.students = len(all_lines)
        self.student_names = []
        self.times = nans(self.students, self.problems)
        self.order = nans(self.students, self.problems)
        self.order_ready = False
        
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

    def read_order(self, filename):
        f = open(filename)
        line = f.readline()
        problem_names = line.rstrip().split(",")[1:]
        pid_remaping = [ None for i in range(len(problem_names)) ]
        for i in range(len(problem_names)):
            if problem_names[i] in self.problem_names:
                pid_remaping[i] = self.problem_names.index(problem_names[i])
        for line in f.readlines():
            tmp = line.rstrip().split(",")
            student = tmp[0]
            if student in self.student_names:
                sid = self.student_names.index(student)
                tmp = tmp[1:]
                for i in range(len(tmp)):
                    if tmp[i] != '' and pid_remaping[i] != None:
                        self.order[sid][pid_remaping[i]] = int(tmp[i])
        self.order_ready = True
        f.close()
        
    def read_datetime(self, filename):
        self.datetimes_order = []
        f = open(filename)
        line = f.readline()
        problem_names = line.rstrip().split(",")[1:]
        pid_remaping = [ None for i in range(len(problem_names)) ]
        for i in range(len(problem_names)):
            if problem_names[i] in self.problem_names:
                pid_remaping[i] = self.problem_names.index(problem_names[i])
        for line in f.readlines():
            tmp = line.rstrip().split(",")
            student = tmp[0]
            if student in self.student_names:
                sid = self.student_names.index(student)
                tmp = tmp[1:]
                for i in range(len(tmp)):
                    if tmp[i] != '' and pid_remaping[i] != None:
                        self.datetimes_order.append((int(tmp[i]), sid, pid_remaping[i]))
        f.close()
        self.datetimes_order.sort()

        
