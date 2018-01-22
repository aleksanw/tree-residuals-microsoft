from helpers.plotter import compose_path


def write_variable_to_latex(variable_value, *subnames):
    filename = compose_path('variables', '.tex', *subnames) 
    target = open(filename, 'w')
    var = str(variable_value)
    if len(var)>1:
        var = var.replace('  ', ', ')

    target.write(var)
    target.close()

def write_variables_to_latex(agent_run):
    for variable_name in agent_run.wanted_written:
        write_variable_to_latex(agent_run.config[variable_name], agent_run.env_name,
                agent_run.agent_name, variable_name,
                agent_run.start_time)
