def write_variable_to_latex(var, var_name):
    filename = f'variables/{var_name}.tex'
    target = open(filename, 'w')
    var = str(var)
    if len(var)>1:
        var = var.replace('  ', ', ')

    target.write(var)
    target.close()

def write_variables_to_latex(variables, names, env_name, agent_name):
    for variable_name in names:
        write_variable_to_latex(variables[variable_name], env_name +
                agent_name + variable_name)
