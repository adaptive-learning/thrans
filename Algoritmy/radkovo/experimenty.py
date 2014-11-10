#!/usr/bin/python -u

from human_data import *
from mymodels import *
from grad_desc import *
from iter_est import *
import pylab as plt
import scipy as sp

def param_histograms(model):
    for param in model.par:
        plot_hist(model.par[param], param)    

def param_correlations(model):
    for par1 in model.par:
        for par2 in model.par:
            if par1 > par2 and len(model.par[par1]) == len(model.par[par2]):
                plot_scatter(model.par[par1], model.par[par2], par1, par2)
    
################## zakladni fitovani modelu ###################
                
def basic_model_fitting(data):
    data.print_stats()
    train, test = split_data_using_order(data, test_students = 0.3, test_portion = 0.2)
    train.print_stats()
    test.print_stats()
    model = Model22L(train)
    obs = GDobserver(model, test)
    obs.verbose = 1
    stochastic_gradient_descent(model, observer = obs)
    obs.plot_rmses()
#    param_histograms(model)
    param_correlations(model)
    plt.show()

#basic_model_fitting(read_problem("Ploty"))

def two_models_param_comparison(data, M1, M2):
    data.print_stats()
    model1, model2 = M1(data), M2(data)
    stochastic_gradient_descent(model1)
    stochastic_gradient_descent(model2)
    for param in model1.par:
        if param in model2.par:
            plot_scatter_line(model1.par[param], model2.par[param], param)
    plt.show()

#two_models_param_comparison(read_problem("Robotanik"), Model22L, Model12)    

################### analyza hodnot parametru #######################

def skill_delta_correlations(problem_types):
    for problem_type in problem_types:
        print problem_type, 
        data = read_problem(problem_type)
        for it, gamma, delta_init in [ (50, 0.005, 0.3), (100, 0.002, 0.4), (80, 0.003, 0.1) ]:
            model = Model22L(data, delta_init = delta_init)
            stochastic_gradient_descent(model, iterations = it, gamma = gamma)
            print spearmanr(model.par['skill'], model.par['delta'])[0],
        print

#skill_delta_correlations(ALL_PROBLEMS)

################## analyza residuals ###################

def analyse_residuals(data):
    model = Model12(data)
    iterative_estimation(model, 3)
    bins = 30
    residuals = np.array([ model.error(s,p) for s in range(data.students) for p in range(data.problems) if not math.isnan(model.error(s,p))])
    mean, std = np.mean(residuals), np.std(residuals)
    print "std", std
    n, bins, patches = plt.hist(residuals, bins, normed=1)
    y = plt.normpdf(bins, mean, std)
    plt.rcParams.update({'font.size': 16})
    plt.plot(bins, y, linewidth=3)
    plt.title("Histogram of residuals")
    plt.savefig("results/residuals-hist.svg")
    print "skew", sp.stats.skew(residuals)
    print "skewtest", sp.stats.skewtest(residuals)
    plt.figure()
    sp.stats.probplot(residuals, dist="norm", plot=plt)
    plt.savefig("results/residuals-pp.png", dpi = 300)
#    plt.show()
    
#analyse_residuals(read_problem("Robotanik"))
        
################### two halves experiment ##########################3#
    
def two_halves_single_run(data, M = Model22L):
    data.print_stats()
    splitdata = split_data_odd_even(data)
    model = [ M(splitdata[0]), M(splitdata[1]) ]
    stochastic_gradient_descent(model[0])
    stochastic_gradient_descent(model[1])
    for param in model[0].par:
        plot_scatter_line(model[0].par[param],model[1].par[param],param)
    plt.show()

#two_halves_single_run(read_problem("Sokoban"))    
    
def two_halves_multiple_runs(problem_types, M = Model22L, rep = 5):
    logger = MultipleRunLogger(1)
    for problem_type in problem_types:
        data = read_problem(problem_type)
        for i in range(rep):
            splitdata = split_data_odd_even(data)
            model = [ M(splitdata[0]), M(splitdata[1]) ]
            stochastic_gradient_descent(model[0], iterations = 100)
            stochastic_gradient_descent(model[1], iterations = 100)
            for param in model[0].par:
                logger.log(param, problem_type[:5], 
                           spearmanr(model[0].par[param],model[1].par[param])[0])
        logger.print_table()
    logger.print_table(" & ", "\\\\")
        
#two_halves_multiple_runs(ALL_PROBLEMS, rep = 1)
#two_halves_multiple_runs(["Ploty"], rep = 1)            
                
################# porovnani ruznych modelu ####################

def model_comparison(problem_types, models, rep = 10):
    logger = MultipleRunLogger(1)
    for problem_type in problem_types:
        data = read_problem(problem_type)
        for i in range(rep):
            train, test = split_data_using_order(data, test_students = 0.3, test_portion = 0.2)
            for M in models:
                model = M(train)
                obs = GDobserver(model, test)
                stochastic_gradient_descent(model, obs)
                logger.log(model.name, problem_type[:5], obs.final_test_rmse())
        logger.print_table()
    logger.print_table(" & ", "\\\\")            

#model_comparison(["Ploty"], [ModelBaseline, Model12, Model22L, Model22Lreg ], 3)    
#model_comparison(["Sokoban","Nurikabe"], [ ModelMean, ModelBaseline, Model12 ], 2)
#model_comparison(ALL_PROBLEMS, [ ModelMean, ModelBaseline, Model12, Model12var, Model22L ])
#model_comparison(["Ploty"], [ ModelMean, ModelBaseline, Model12, Model12var, Model22L ])    
#model_comparison(ALL_PROBLEMS, [ Model22L, Model22Lsqrt ], 4)    

################### experimenty s postupnym pruchodem ################

# zkousel jsem i abs. hodnoty misto rmse, dava v podstate stejne vysledky
def evaluate_seq_test(test, model, discount=1, window = None):
    sqsum = 0.0
    count = 0 
    for s in range(test.students):
        problem_ids = get_problem_id_in_order(test.order[s], test.student_solved[s])
        times_seq = np.array([ test.times[s][p] for p in problem_ids ])

        if window == None:
            res = model.prediction_sequence_residuals(problem_ids, times_seq, discount)
        else:
            res = model.prediction_sequence_residuals_window(problem_ids, times_seq, window)
            
        if not np.any(np.isnan(res)): # nastava vyjimecne u Robotanika
            sqsum += np.sum(res**2)
            count += len(res)
    return math.sqrt(sqsum / count) 
            
def model_comparison_seq(problem_types, rep = 10):
    # todo pridat "bayes" verzi
    logger = MultipleRunLogger(1)
    for problem_type in problem_types:
        data = read_problem(problem_type)
        for i in range(rep):
            train, test = split_data_by_students(data, 0.1)
            modelb = ModelBaseline(train)
            model12 = Model12(train)
#            iterative_estimation(model12)
            stochastic_gradient_descent(model12)
            
            logger.log("baseline    ", problem_type[:5], evaluate_seq_test(test, modelb))
            logger.log("baseline 0.9", problem_type[:5], evaluate_seq_test(test, modelb, 0.9))
            logger.log("model12     ", problem_type[:5], evaluate_seq_test(test, model12))
            logger.log("model12 0.9 ", problem_type[:5], evaluate_seq_test(test, model12, 0.9))        
        logger.print_table()
    logger.print_table(" & ", "\\\\")            

#model_comparison_seq(ALL_PROBLEMS)    

############### discounting - zastarale, nove v exp_online ##############

dvals = [ 0.6, 0.7, 0.75, 0.8, 0.83, 0.85, 0.88, 0.9, 0.92, 0.95, 0.99, 1 ]

def sensitivity_discounting_get_data(problem_type, rep = 10):
    data = read_problem(problem_type)
    train, test = [], []
    for i in range(rep):
        t1, t2 = split_data_by_students(data, 0.1)
        train.append(t1)
        test.append(t2)
    rmses = [ 0 for _ in range(len(dvals)) ]
    for j in range(len(dvals)):    
        d = dvals[j]
        for i in range(rep):
            model = ModelBaseline(train[i])
            rmse = evaluate_seq_test(test[i], model, d)            
            rmses[j] += rmse
        rmses[j] = rmses[j] / rep
    return rmses

windows = [ 3, 5, 7, 9, 11, 13, 15, 18, 21, 25 ]
def sensitivity_discounting_get_data_windows(problem_type, rep = 3):
    data = read_problem(problem_type)
    train, test = [], []
    for i in range(rep):
        t1, t2 = split_data_by_students(data, 0.1)
        train.append(t1)
        test.append(t2)
    rmses = [ 0 for _ in range(len(windows)) ]
    for j in range(len(windows)):    
        w = windows[j]
        for i in range(rep):
            model = ModelBaseline(train[i])
            rmse = evaluate_seq_test(test[i], model, window = w)            
            rmses[j] += rmse
        rmses[j] = rmses[j] / rep
    return rmses

def sensitivity_discounting_plot_single(problem_type):
    rmses = sensitivity_discounting_get_data(problem_type,5)
    plt.plot(dvals, rmses)
    plt.figure()
    rmses = sensitivity_discounting_get_data_windows(problem_type,5)
    plt.plot(windows, rmses)
    plt.title(problem_type)
    plt.xlabel("Discount coefficient")
    plt.ylabel("RMSE")
    plt.show()
    
#sensitivity_discounting_plot_single("Sokoban")    

def sensitivity_discounting_normalized(problem_types):
    legend = []
    for pt in problem_types:
        legend.append(pt)
        rmses = sensitivity_discounting_get_data(pt, 3)
        rmse1 = rmses[-1]
        nrmses = [ x / rmse1 for x in rmses ]
        plt.plot(dvals, nrmses)
    plt.xlabel("Discount coefficient")
    plt.ylabel("normalized RMSE")
    plt.legend(legend, loc = 2)
    plt.show()

#sensitivity_discounting_normalized(["Ploty","Nurikabe","Binary", "Robotanik","Sokoban"])
#sensitivity_discounting_normalized(ALL_PROBLEMS)        


################ porovnani ruznych zpusobu mereni odlisnych problemu ####

def get_mean_correlation(data):
    cor = np.zeros((data.problems, data.problems))
    for p1 in range(data.problems):
        for p2 in range(data.problems):
            c = correlation_common(data.times[:,p1], data.times[:,p2])
            cor[p1][p2] = - c # hack, aby to bylo usporadano stejnym smerem
    return np.mean(cor, axis = 0)

def get_discrimination(data):
    model = Model12(data)
    iterative_estimation(model, 3)
    return model.par['a']

def get_randomness(data):
    model = Model11(data)
    stochastic_gradient_descent(model)
    # tohle by slo  napsat kompaktne maticove...
    randomness = np.zeros(data.problems)    
    for p in range(data.problems):
        ss = 0
        k = 0
        for s in range(data.students):
            if not math.isnan(data.times[s][p]):
                ss += (data.times[s][p] - model.predict(s,p))**2
                k += 1                
        randomness[p] = math.sqrt(ss/k)
    return randomness
        

def get_skill_time_cor(data):
    model = Model11(data)
    stochastic_gradient_descent(model)
    stc = np.zeros(data.problems)
    for p in range(data.problems):
        stc[p] = correlation_common(model.par['skill'], data.times[:,p])
    return stc


def largest_names(title, metric, names, k):
    print title
    tmp = sorted(metric)
    tmp = tmp[::-1] # otocime to... typicke
    for i in range(k):
        idx = list(metric).index(tmp[-1-i])
        print names[idx], find_full_name("Robotanik", names[idx]) # hack

def compare_problem_measures(data):
    # k = 5
    ran = get_skill_time_cor(data)
    # largest_names("randomness", ran, data.problem_names, k)    
    mcor = get_mean_correlation(data)
    # largest_names("mean cor", mcor, data.problem_names, k)
    a = get_discrimination(data)
    # largest_names("discrim", a, data.problem_names, k)
    stc = get_skill_time_cor(data)
    # largest_names("skill time cor", stc, data.problem_names, k)    
    plot_scatter(ran, mcor, "randomness", "mean cor")
    plot_scatter(ran, a, "randomness", "discrimination")
    plot_scatter(a, mcor, "discrimination", "mean cor")
    plot_scatter(a, stc, "discrimination", "skill time cor")
    plot_scatter(stc, mcor, "skill time cor", "mean cor")
    plt.show()

set_dir("../../Koncepty/Data")
#compare_problem_measures(read_problem("Sokoban", ordering = 0))
data = read_problem("Sokoban", ordering = 0)
result = {}
for name, value in zip(data.problem_names, get_discrimination(data)):
    result[name] = value

import json
with open("../../Koncepty/Outliers data/{0} disc by Radek.json".format("Sokoban"), "w") as f:
        f.write(json.dumps(result))

################## ruzne metriky podobnosti problemu ############

def cosine_common(x, y):
    sumxy, sumx2, sumy2, common = 0, 0, 0, 0
    for i in range(len(x)):
        if not math.isnan(x[i]) and not math.isnan(y[i]):
            sumxy += x[i] * y[i]
            sumx2 += x[i]**2
            sumy2 += y[i]**2
            common += 1
    return sumxy/(math.sqrt(sumx2) * math.sqrt(sumy2)), common
    
def compare_metrics(data):
    spearmans, cosines, common, difficsum = [], [], [], []
    cor = np.zeros((data.problems, data.problems))
    cos = np.zeros((data.problems, data.problems))
    for p1 in range(data.problems):
        for p2 in range(data.problems):
            spearman = correlation_common(data.times[:,p1], data.times[:,p2])
            cor[p1][p2] = spearman
            cosine, com = cosine_common(data.times[:,p1], data.times[:,p2])
            cos[p1][p2] = cosine
#            print p1, p2, spearman, cosine
            spearmans.append(spearman)
            cosines.append(cosine)
            common.append(com)
            difficsum.append(data.problem_mean_time[p1] + data.problem_mean_time[p2])
    plot_scatter(spearmans,cosines, "spearman", "cosine")
#    plot_scatter(common, spearmans, "common", "spearman")
#    plot_scatter(common, cosines, "common", "cosine")
    plot_scatter(difficsum, spearmans, "difficsum", "spearman")
    plot_scatter(difficsum, cosines, "difficsum", "cosine")
    plt.matshow(cor)
    plt.matshow(cos)
    plt.show()
            
#compare_metrics(read_problem("Sokoban"))    
        
################## porovnani poradi #####################
# ukazuje v podstate zadne rozdily nebo mirne negativni vysledky pro slozitejsi model
# mozno asi zahodit
        
def compare_ordering(data):
    # todo split train, test
    modelmean = ModelMean(data)
    model12 = Model12(data)
    iterative_estimation(model12, 6)
    difm = modelmean.par['m']
    difb = model12.par['b']
    plt.plot(difm, difb, "o")
    plt.figure()
    x, y = [], []
    sm, sb = 0, 0
    summ, sumb = 0, 0
    for s in range(data.students):
        corm = correlation_common(data.times[s], difm)
        corb = correlation_common(data.times[s], difb)
        summ += corm
        sumb += corb
        if corm > corb: sm += 1
        if corm < corb: sb += 1
        print s, corm, corb
        x.append(corm)
        y.append(corb)
    print "Cor m mean", summ / data.students
    print "Cor b mean", sumb / data.students
    print "Cor m better", sm
    print "Cor b better", sb
    plt.plot(x,y,"x")
    plt.show()
    
#compare_ordering(read_problem("Sokoban"))


####################### debug #########################

def test(data):
    data.print_stats()

#test(read_problem("Nurikabe"))

