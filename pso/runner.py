import testfuncts.testfuncts as tf

class Runner:
    
    def __init__(self):
        pass

    def run():
        pass

def test_PSO():
    pso_hyperparameters = dc.PSOHyperparameters(
        num_part = p.NUM_PART,
        num_dim=p.NUM_DIM, 
        alpha = p.ALPHA,
        max_iterations=p.MAX_ITERATIONS, 
        w=p.W, 
        c1=p.C1, 
        c2=p.C2, 
        tolerance=p.TOLERANCE, 
        mv_iteration=p.NO_MOVEMENT_TERMINATION
    )

    ccd_hyperparameters = dc.CCDHyperparameters(
        ccd_alpha=p.CCD_ALPHA, 
        ccd_tol=p.CCD_TOL, 
        ccd_max_its=p.CCD_MAX_ITS,
        ccd_third_term_its=p.CCD_THIRD_TERM_ITS
    )

    domain_data = dc.DomainData(
        upper_bound = p.UPPER_BOUND,
        lower_bound = p.LOWER_BOUND
    )

    runner_config = MPSORunnerConfigs(
        use_ccd=True
    )

    optimum = optimum=p.OPTIMUM
    bias=p.BIAS,
    function = tf.TF.generate_function(p.FUNCT, optimum=optimum, bias=bias)

    pso = PSO(
        pso_hyperparameters = pso_hyperparameters,
        ccd_hyperparameters = ccd_hyperparameters,
        domain_data = domain_data,
        function = function
    )

    logging_settings = dc.PSOLoggerConfig(
        log_level = dc.LogLevels.NO_LOG
    )

    runner = MPSO(
        pso=pso, 
        runs=5, 
        logging_settings=logging_settings,
        runner_settings=runner_config
    )

    runner.mpso_ccd()
    #runner = PSORunner(pso)
    #runner.mpso_ccd()        """Set the g_best for this object"""
    #print(runner.pso.pso)
    print(runner.pso.g_best)