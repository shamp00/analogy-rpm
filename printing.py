#%%
import cairo
import numpy as np
import os
import copy
from math import pi

from IPython.display import SVG, display
import pyRavenMatrices.matrix as mat
import pyRavenMatrices.element as elt
import pyRavenMatrices.lib.sandia.definitions as defs
import pyRavenMatrices.lib.sandia.generators as gen
import pyRavenMatrices.transformation as tfm

# pylint: disable-msg=E1101 
# E1101: Module 'cairo' has no 'foo' member - of course it has! :) 

def cell_path(cell):
    return os.path.join('.', cell.id + '.svg')    

def test_element(element, cell_size = 64):
    cell_margin = cell_size // 8

    cell_structure = mat.CellStructure("generated" + str(0), cell_size, cell_size, cell_margin, cell_margin)

    surface = cairo.SVGSurface(cell_path(cell_structure), cell_structure.width, cell_structure.height)
    ctx = cairo.Context(surface)
    # set colour of ink to middle grey
    #ctx.set_source_rgb(0.5, 0.5, 0.5)
    
    element.draw_in_context(ctx, cell_structure)

    ctx.stroke()
    surface.finish()
    display(SVG(cell_path(cell_structure)))


def test_matrix(elements, cell_size = 64, is_correct = None):
    cell_margin = cell_size // 8
    if elements == None:
        return
    if len(elements) == 2:
        element1 = elements[0]
        element2 = elements[1]
        cell_structure = mat.CellStructure("generated" + str(0), cell_size, cell_size, cell_margin, cell_margin)

        surface = cairo.SVGSurface(cell_path(cell_structure), cell_structure.width * 2, cell_structure.height)
        
        ctx = cairo.Context(surface)    
        ctx.rectangle(0, 0, cell_structure.width * 2, cell_structure.height)
        if is_correct == False:
            ctx.set_source_rgb(1.0, 0.9, 0.9)            
        elif is_correct == True:
            ctx.set_source_rgb(0.9, 1.0, 0.9)
        else:
            ctx.set_source_rgb(0.9, 0.9, 0.9)

        ctx.fill()
        ctx.set_source_rgb(0, 0, 0)        

        element1.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width, 0)    
        ctx.stroke()

        element2.draw_in_context(ctx, cell_structure)    
        ctx.stroke()

        surface.finish()

        display(SVG(cell_path(cell_structure)))

    if len(elements) == 4:
        element1 = elements[0]
        element2 = elements[1]
        element3 = elements[2]
        element4 = elements[3]

        cell_structure = mat.CellStructure("generated" + str(0), cell_size, cell_size, cell_margin, cell_margin)

        surface = cairo.SVGSurface(cell_path(cell_structure), cell_structure.width * 2, cell_structure.height * 2)
        
        ctx = cairo.Context(surface)    
        ctx.rectangle(0, 0, cell_structure.width * 2, cell_structure.height * 2)
        if is_correct == False:
            ctx.set_source_rgb(1.0, 0.9, 0.9)            
        elif is_correct == True:
            ctx.set_source_rgb(0.9, 1.0, 0.9)
        else:
            ctx.set_source_rgb(0.9, 0.9, 0.9)
        ctx.fill()
        ctx.set_source_rgb(0, 0, 0)        

        element1.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width, 0)    
        ctx.stroke()

        element2.draw_in_context(ctx, cell_structure)    
        ctx.translate(-cell_structure.width, cell_structure.height)    
        ctx.stroke()

        element3.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width, 0)    
        ctx.stroke()

        element4.draw_in_context(ctx, cell_structure)
        ctx.stroke()

        surface.finish()

        display(SVG(cell_path(cell_structure)))

    if len(elements) == 6:
        element1 = elements[0]
        element2 = elements[1]
        element3 = elements[2]
        element4 = elements[3]
        element5 = elements[4]
        element6 = elements[5]

        cell_structure = mat.CellStructure("generated" + str(0), cell_size, cell_size, cell_margin, cell_margin)

        surface = cairo.SVGSurface(cell_path(cell_structure), cell_structure.width * 3, cell_structure.height * 2)
        
        ctx = cairo.Context(surface)    
        ctx.rectangle(0, 0, cell_structure.width * 3, cell_structure.height * 2)
        if is_correct == False:
            ctx.set_source_rgb(1.0, 0.9, 0.9)            
        elif is_correct == True:
            ctx.set_source_rgb(0.9, 1.0, 0.9)
        else:
            ctx.set_source_rgb(0.9, 0.9, 0.9)
        ctx.fill()
        ctx.set_source_rgb(0, 0, 0)        

        element1.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width, 0)    
        ctx.stroke()

        element2.draw_in_context(ctx, cell_structure)    
        ctx.translate(cell_structure.width, 0)    
        ctx.stroke()

        element3.draw_in_context(ctx, cell_structure)    
        ctx.translate(-2 * cell_structure.width, cell_structure.height)    
        ctx.stroke()

        element4.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width, 0)    
        ctx.stroke()

        element5.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width, 0)    
        ctx.stroke()

        element6.draw_in_context(ctx, cell_structure)
        ctx.stroke()

        surface.finish()

        display(SVG(cell_path(cell_structure)))

    if len(elements) == 9:
        element1 = elements[0]
        element2 = elements[1]
        element3 = elements[2]
        element4 = elements[3]
        element5 = elements[4]
        element6 = elements[5]
        element7 = elements[3]
        element8 = elements[4]
        element9 = elements[5]

        cell_structure = mat.CellStructure("generated" + str(0), cell_size, cell_size, cell_margin, cell_margin)

        surface = cairo.SVGSurface(cell_path(cell_structure), cell_structure.width * 2, cell_structure.height * 2)
        
        ctx = cairo.Context(surface)    
        ctx.rectangle(0, 0, cell_structure.width * 3, cell_structure.height * 2)
        if is_correct == False:
            ctx.set_source_rgb(1.0, 0.9, 0.9)            
        elif is_correct == True:
            ctx.set_source_rgb(0.9, 1.0, 0.9)
        else:
            ctx.set_source_rgb(0.9, 0.9, 0.9)
        ctx.fill()
        ctx.set_source_rgb(0, 0, 0)        

        element1.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width, 0)    
        ctx.stroke()

        element2.draw_in_context(ctx, cell_structure)    
        ctx.translate(cell_structure.width, 0)    
        ctx.stroke()

        element3.draw_in_context(ctx, cell_structure)    
        ctx.translate(-2 * cell_structure.width, cell_structure.height)    
        ctx.stroke()

        element4.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width, 0)    
        ctx.stroke()

        element5.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width, 0)    
        ctx.stroke()

        element6.draw_in_context(ctx, cell_structure)    
        ctx.translate(-2 * cell_structure.width, cell_structure.height)    
        ctx.stroke()

        element7.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width, 0)    
        ctx.stroke()

        element8.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width, 0)    
        ctx.stroke()

        element9.draw_in_context(ctx, cell_structure)
        ctx.stroke()

        surface.finish()

        display(SVG(cell_path(cell_structure)))

def analyze_element(element, routine_gen, decorator_gen, include_shape_variants = True):
    # Get all the available targets for modification
    targets = tfm.get_targets(element)

    # Get the basic starting element
    basic_element = targets[0](element)

    # Modify the shape routine to get an analogy element
    analogy_element = copy.deepcopy(element)
    basic_analogy_element = targets[0](analogy_element)

    basic_analogy_element.routine = routine_gen.sample()[0]
    while basic_analogy_element.routine == basic_element.routine:
        basic_analogy_element.routine = routine_gen.sample()[0]        
    basic_analogy_element.params = routine_gen.sample_params(basic_analogy_element.routine)[0]

    # Extract the parameters of the shapes and decorator
    shape_index = list(routine_gen.routines.keys()).index(basic_element.routine)
    shape = np.zeros(6)
    shape[shape_index] = 1.
    shape_params = np.array([basic_element.params['r'] / 8])
    
    analogy_shape_index = list(routine_gen.routines.keys()).index(basic_analogy_element.routine)
    analogy_shape = np.zeros(6)
    analogy_shape[analogy_shape_index] = 1.
    analogy_shape_params = np.array([basic_analogy_element.params['r'] / 8])

    initial_decoration = np.zeros(4)

    decorator = np.zeros(4)
    decorator_params = np.zeros(4)

    for target in targets[1:]:
        modification = target(element)

        decorator_index = list(decorator_gen.decorators.keys()).index(modification.decorator)
        decorator[decorator_index] = 1.
        decorator_params[decorator_index] = list(modification.params.values())[0]
        # exception for rotation which is in radians
        if decorator_index == 1:
            decorator_params[decorator_index] = decorator_params[decorator_index] / (2 * pi) 
        # exception for luminosity which is 1-0 instead of 0-1
        if decorator_index == 2:
            decorator_params[decorator_index] = abs(decorator_params[decorator_index] - 1) 
        # exception for numerosity which is 1-8 instead of 0-1
        if decorator_index == 3:
            decorator_params[decorator_index] = (decorator_params[decorator_index] - 1) / 8 

    test = np.concatenate((shape, shape_params, initial_decoration))
    analogy = np.concatenate((analogy_shape, analogy_shape_params, initial_decoration))
    
    transformation = np.concatenate((decorator, decorator_params))

    matrix = []
    matrix.append(basic_element)
    matrix.append(element)
    matrix.append(basic_analogy_element)
    matrix.append(analogy_element)
    # return matrix, sample, transformation, analogy
    return matrix, test, transformation, analogy

def generate_all_sandia_matrices(num_modifications = [0, 1, 2, 3], include_shape_variants = True):
    # zero modifications, generate a basic shape
    routine_gen = gen.RoutineGenerator()
    decorator_gen = gen.DecoratorGenerator()

    for shape in routine_gen.routines:
        for shape_param in routine_gen.params[shape]:
            basic_element = elt.BasicElement()
            basic_element.routine = shape
            shape_variants = routine_gen.params[shape][shape_param]
            if not include_shape_variants:
                shape_variants = [4]
            for key in shape_variants:
                basic_element.params = { 'r' : key }                
                if 0 in num_modifications:
                    yield analyze_element(basic_element, routine_gen, decorator_gen)
                if 1 in num_modifications:
                    for decorator in decorator_gen.decorators:
                        for decorator_param in decorator_gen.params[decorator]:
                            modifier = elt.ElementModifier()
                            modifier.decorator = decorator
                            for key in decorator_gen.params[decorator][decorator_param]:
                                modifier.params =  { decorator_param : key }
                                modified_element = elt.ModifiedElement(basic_element, modifier)
                                yield analyze_element(modified_element, routine_gen, decorator_gen)
                if 2 in num_modifications:
                    decorators = list(decorator_gen.decorators.keys())
                    for i, decorator1 in enumerate(decorators):
                        for decorator2 in decorators[i+1:]:
                            for decorator1_param in decorator_gen.params[decorator1]:
                                for decorator2_param in decorator_gen.params[decorator2]:
                                    modifier1 = elt.ElementModifier()
                                    modifier1.decorator = decorator1
                                    for key in decorator_gen.params[decorator1][decorator1_param]:
                                        modifier1.params =  { decorator1_param : key }
                                        modifier2 = elt.ElementModifier()
                                        modifier2.decorator = decorator2
                                        for key in decorator_gen.params[decorator2][decorator2_param]:
                                            modifier2.params =  { decorator2_param : key }
                                            modified_element = elt.ModifiedElement(basic_element, modifier1, modifier2)
                                            yield analyze_element(modified_element, routine_gen, decorator_gen)

def generate_sandia_matrix(num_modifications = -1, include_shape_variants=True):
    if num_modifications == 0 or (num_modifications == -1 and np.random.randint(4) == 0):
        # zero modifications, generate a basic shape
        branch = {
            'basic': 1.,
            'composite': 0.,
            'modified': 0.
        }
        modifier_num = {
            1: 1 / 3,
            2: 1 / 3,
            3: 1 / 3
        }
    else:
        branch = {
            'basic': 0.,
            'composite': 0.,
            'modified': 1.
        }
        if num_modifications == 1:
            modifier_num = {
                1: 1,
                2: 0,
                3: 0
            }
        elif num_modifications == 2:
            modifier_num = {
                1: 0,
                2: 1,
                3: 0
            }
        elif num_modifications == 3:
            modifier_num = {
                1: 0,
                2: 0,
                3: 3
            }
        else:
            modifier_num = {
                1: 1 / 3,
                2: 1 / 3,
                3: 1 / 3
            }
    # at least one modification
    structure_gen = gen.StructureGenerator(
        branch = branch,
        modifier_num = modifier_num
    )
    routine_gen = gen.RoutineGenerator()
    decorator_gen = gen.DecoratorGenerator()

    # Generate an element. For now this will be a modified element with num_modification modifications.
    element = gen.generate_sandia_figure(structure_gen, routine_gen, decorator_gen)
    if not include_shape_variants:
        element.params = { 'r': 4 }

    return analyze_element(element, routine_gen, decorator_gen)

def generate_sandia_matrix_2_by_3(include_shape_variants=True):
    matrix, test, transformation, analogy = generate_sandia_matrix(0, include_shape_variants)
    
    # For 2x2 and 3x3 allow 1 modification
    num_modifications = 1

    routine_gen = gen.RoutineGenerator()    
    decorator_gen = gen.DecoratorGenerator()
    
    decorators = decorator_gen.sample(num_modifications * 2, replace=False)

    decorator = decorators.pop()
    decorator_params = decorator_gen.sample_params(decorator).pop()
    elementModifier1 = elt.ElementModifier()
    elementModifier1.decorator = decorator
    elementModifier1.params = decorator_params

    decorator = decorators.pop()
    decorator_params = decorator_gen.sample_params(decorator).pop()
    elementModifier2 = elt.ElementModifier()
    elementModifier2.decorator = decorator
    elementModifier2.params = decorator_params

    basic_element = matrix[1]
    modified_element1 = elt.ModifiedElement(basic_element, elementModifier1)
    modified_element2 = elt.ModifiedElement(basic_element, elementModifier1, elementModifier2)

    basic_analogy_element = matrix[3]
    modified_analogy_element1 = elt.ModifiedElement(basic_analogy_element, elementModifier1)
    modified_analogy_element2 = elt.ModifiedElement(basic_analogy_element, elementModifier1, elementModifier2)

    _, test1, transformation1, analogy1 = analyze_element(modified_element1, routine_gen, decorator_gen)
    _, test2, transformation2, analogy2 = analyze_element(modified_element2, routine_gen, decorator_gen)

    transformation2 = transformation2 - transformation1

    matrix = []
    matrix.append(basic_element)
    matrix.append(modified_element1)
    matrix.append(modified_element2)
 
    matrix.append(basic_analogy_element)
    matrix.append(modified_analogy_element1)
    matrix.append(modified_analogy_element2)

    # return matrix, sample, transformation, analogy

    return matrix, test1, test2, transformation1, transformation2, analogy1, analogy2


def generate_rpm_sample(num_modifications = -1):
    """Generate a vector representing a 2x2 RPM matrix"""
    # scales = np.random.randint(0, 8)
    # rotation = np.random.randint(0, 8)
    # shading = np.random.randint(0, 8)
    # numerosity = np.random.randint(0, 8)

    # Create a vector like this [1 0 0 0] for shape1, say, ellipse
    shape_ints = np.random.choice(range(6), 2, replace=False)
    shape = np.zeros(6)
    shape[shape_ints[0]] = 1.

    shape_param = np.zeros(1)
    shape_param[0] = np.random.choice([0.25, 0.5, 1, 2, 4, 8]) / 8

    analogy_shape = np.zeros(6)
    analogy_shape[shape_ints[1]] = 1.

    shape_features = np.zeros(4) # for scale, rotation, shading, numerosity
    #shape_features = np.random.randint(4, size=4) / 4

    # To follow the relational priming example, we would need a 'causal agent'.
    #
    # Causal agent is,
    #   shape = transformer 
    #   scale = enlarger/shrinker, 
    #   shading = shader, 
    #   rotation = rotator, 
    #   numerosity = multiplier. 
    #
    # (Seems a little artificial but for now we'll go with it). Also, the
    # causal agent does not have the notion of degree, i.e., a slightly
    # cut apple versus a very cut apple, whereas a shape can be slightly 
    # shaded or slightly rotated.
    #
    # A shape transformation from say, triangle to circle is presumably a 
    # different causal agent than from triangle to square, so we'd end up with a 
    # separate causal agent for each transformation.
    # 
    # But we need to avoid this for the feature changes. We need to be careful that a change of shading from 1 to 2 is 
    # in some way the same causal agent as a change of shading from 3 to 4. Otherwise we end
    # up with each possible transformation having a separate casual agent.
    # In other words, what is the 'shape' equivalent of 
    #   apple, bread, lemon all being acted on by a knife.
    #   circle, triangle, square all being acted on by a modifier with a parameter?

    # scale, shading, rotation or numerosity
    modification_type = np.zeros(4)
    # make 0-3 modifications
    if num_modifications == -1:
        num_modifications = np.random.randint(4)
    modifications = np.random.choice(range(4), num_modifications, replace=False)
    for modification in modifications:
        modification_type[modification] = 1.

    modification_parameters = np.zeros(4)
    for modification in modifications:
        parameter = np.random.randint(8)
        modification_parameters[modification] = parameter / 8

    sample = np.concatenate((shape, shape_param, shape_features))
    transformation = np.concatenate((modification_type, modification_parameters))
    analogy = np.concatenate((analogy_shape, shape_param, shape_features))
    # return matrix, sample, transformation, analogy
    return None, sample, transformation, analogy

def display_one_random_2_by_2():
    matrix, test, transformation, analogy = generate_sandia_matrix()
    print(f'Test    = {test}')
    print(f'Analogy = {analogy}')
    print(f'Transformation = {np.round(transformation, 3)}')
    test_matrix(matrix, is_correct=True)

def display_one_random_2_by_3():
    matrix, test1, test2, transformation1, transformation2, analogy1, analogy2 = generate_sandia_matrix_2_by_3()
    print(f'Test1    = {test1}')
    print(f'Test2    = {test2}')
    print(f'Analogy1 = {analogy1}')
    print(f'Analogy2 = {analogy2}')
    print(f'Transformation1 = {np.round(transformation1, 3)}')
    print(f'Transformation2 = {np.round(transformation2, 3)}')
    test_matrix(matrix, is_correct=True)

def display_one_random_3_by_3():
    matrix, test, transformation, analogy = generate_sandia_matrix_3_by_3()
    print(f'Test    = {test}')
    print(f'Analogy = {analogy}')
    print(f'Transformation = {np.round(transformation, 3)}')
    test_matrix(matrix, is_correct=True)

def display_all_sandia_matrices(num=3, num_modifications = [0,1,2,3]):
    i=0
    for matrix, test, transformation, analogy in generate_all_sandia_matrices(num_modifications):
        i += 1
        print(f'Test    = {test}')
        print(f'Analogy = {analogy}')
        print(f'Transformation = {np.round(transformation, 3)}')
        test_matrix(matrix[0:2], is_correct=True)
        if i==num:
            break


#%%
#display_one_random_2_by_2()
#display_all_sandia_matrices(100, [0])
#print(sum(1 for i in generate_all_sandia_matrices([0], include_shape_variants = False)))
display_one_random_2_by_3()

#%%
