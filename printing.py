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
        element7 = elements[6]
        element8 = elements[7]
        element9 = elements[8]

        cell_structure = mat.CellStructure("generated" + str(0), cell_size, cell_size, cell_margin, cell_margin)

        surface = cairo.SVGSurface(cell_path(cell_structure), cell_structure.width * 3, cell_structure.height * 3)
        
        ctx = cairo.Context(surface)    
        ctx.rectangle(0, 0, cell_structure.width * 3, cell_structure.height * 3)
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

    basic_analogy_element.routine = routine_gen.sample().pop()
    while basic_analogy_element.routine == basic_element.routine:
        basic_analogy_element.routine = routine_gen.sample().pop()        
    basic_analogy_element.params = routine_gen.sample_params(basic_analogy_element.routine).pop()

    if not include_shape_variants:
        basic_analogy_element.params = { 'r': 4 }

    # Extract the parameters of the shapes and decorator
    shape_index = list(routine_gen.routines.keys()).index(basic_element.routine)
    shape = generate_shape(shape_index)
    shape_params = np.array([basic_element.params['r'] / 8])
    
    analogy_shape_index = list(routine_gen.routines.keys()).index(basic_analogy_element.routine)
    analogy_shape = generate_shape(analogy_shape_index)
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
                    yield analyze_element(basic_element, routine_gen, decorator_gen, include_shape_variants)
                if 1 in num_modifications:
                    for decorator in decorator_gen.decorators:
                        for decorator_param in decorator_gen.params[decorator]:
                            modifier = elt.ElementModifier()
                            modifier.decorator = decorator
                            for key in decorator_gen.params[decorator][decorator_param]:
                                modifier.params =  { decorator_param : key }
                                modified_element = elt.ModifiedElement(basic_element, modifier)
                                yield analyze_element(modified_element, routine_gen, decorator_gen, include_shape_variants)
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
                                            yield analyze_element(modified_element, routine_gen, decorator_gen, include_shape_variants)

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

    return analyze_element(element, routine_gen, decorator_gen, include_shape_variants)

def generate_sandia_matrix_2_by_3(include_shape_variants=True):
    matrix, test, _, analogy = generate_sandia_matrix(0, include_shape_variants)
    
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

    _, _, transformation1, _ = analyze_element(modified_element1, routine_gen, decorator_gen, include_shape_variants)
    _, _, transformation2, _ = analyze_element(modified_element2, routine_gen, decorator_gen, include_shape_variants)

    transformation2 = transformation2 - transformation1

    matrix = []
    matrix.append(basic_element)
    matrix.append(modified_element1)
    matrix.append(modified_element2)
 
    matrix.append(basic_analogy_element)
    matrix.append(modified_analogy_element1)
    matrix.append(modified_analogy_element2)

    # return matrix, sample, transformation, analogy

    return matrix, test, transformation1, transformation2, analogy

def generate_sandia_matrix_3_by_3(include_shape_variants=True):
    matrix, test, _, analogy = generate_sandia_matrix(0, include_shape_variants)
    
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

    # Modify the shape routine to get an analogy element
    basic_analogy2_element = copy.deepcopy(basic_element)

    basic_analogy2_element.routine = routine_gen.sample().pop()
    while basic_analogy2_element.routine == basic_element.routine or basic_analogy2_element.routine == basic_analogy_element.routine:
        basic_analogy2_element.routine = routine_gen.sample().pop()        
    basic_analogy2_element.params = routine_gen.sample_params(basic_analogy2_element.routine).pop()

    if not include_shape_variants:
        basic_analogy2_element.params = { 'r': 4 }

    modified_analogy2_element1 = elt.ModifiedElement(basic_analogy2_element, elementModifier1)
    modified_analogy2_element2 = elt.ModifiedElement(basic_analogy2_element, elementModifier1, elementModifier2)

    analogy2, _, transformation1, _ = analyze_element(modified_analogy2_element1, routine_gen, decorator_gen, include_shape_variants)
    _, _, transformation2, _ = analyze_element(modified_analogy2_element2, routine_gen, decorator_gen, include_shape_variants)

    transformation2 = transformation2 - transformation1

    matrix = []
    matrix.append(basic_element)
    matrix.append(modified_element1)
    matrix.append(modified_element2)
 
    matrix.append(basic_analogy_element)
    matrix.append(modified_analogy_element1)
    matrix.append(modified_analogy_element2)

    matrix.append(basic_analogy2_element)
    matrix.append(modified_analogy2_element1)
    matrix.append(modified_analogy2_element2)

    # return matrix, sample, transformation, analogy

    return matrix, test, transformation1, transformation2, analogy, analogy2

from dataclasses import dataclass

# shapes
ellipse: int = 0
triangle: int = 1
rectangle: int = 2
diamond: int = 3
trapezoid: int = 4
tee: int = 5

# transformations
scale = 0
rotation = 1
shading = 2
numerosity = 3

@dataclass
class Lexicon:
    """Allows for easy restriction of the lexicon used during the generation of training/test
    data to facilitate experiments on interpolation/extrapolation."""
    # available shapes
    shapes = [ellipse, triangle, rectangle, diamond, trapezoid, tee]

    # available shape parameters
    ellipse_params = range(3)
    triangle_params = range(5)
    rectangle_params = range(3)
    diamond_params = range(5)
    trapezoid_params = range(3)
    tee_params = range(5)

    # available transformations
    transformations = [scale, rotation, shading, numerosity]

    # available transformation parameter values
    scale_values = range(0, 4)
    rotation_values = range(0, 8)
    shading_values = range(0, 8)
    numerosity_values = range(1, 9)


def generate_base_elements_as_vector(lexicon: Lexicon):
    """Generate a vector representing the starting shapes corresponding to 
    an element and an analogy"""

    # Create a vector like this [1 0 0 0 0 0] for shape1, say, ellipse
    shape_int, analogy_shape_int = np.random.choice(lexicon.shapes, 2, replace=False)

    shape = generate_shape(shape_int)
    shape_param = generate_shape_params(lexicon, shape_int)

    analogy_shape = generate_shape(analogy_shape_int)
    analogy_shape_param = generate_shape_params(lexicon, shape_int)

    scale_param = np.random.choice(lexicon.scale_values, 1) / len(lexicon.scale_values)
    rotation_param = np.random.choice(lexicon.rotation_values) / len(lexicon.rotation_values)
    shading_param = np.random.choice(lexicon.shading_values) / len(lexicon.shading_values)
    numerosity_param = np.random.choice(lexicon.numerosity_values) / len(lexicon.numerosity_values)

    shape_features = np.zeros(4)
    shape_features[scale] = scale_param 
    shape_features[rotation] = rotation_param
    shape_features[shading] = shading_param
    shape_features[numerosity] = numerosity_param
    
    sample = np.concatenate((shape, shape_param, shape_features))
    analogy = np.concatenate((analogy_shape, analogy_shape_param, shape_features))
    
    # return matrix, sample, transformation, analogy
    return sample, analogy


def generate_shape(shape_int):
    shape = np.zeros(6)
    shape[shape_int] = 1.
    return shape


def generate_shape_params(lexicon: Lexicon, shape_int: int):
    shape_param = np.zeros(1)
    if shape_int in [ellipse]:       
        shape_param[0] = np.random.choice(lexicon.ellipse_params) / (len(lexicon.ellipse_params) - 1)
    if shape_int in [triangle]:       
        shape_param[0] = np.random.choice(lexicon.triangle_params) / (len(lexicon.triangle_params) - 1)
    if shape_int in [rectangle]:       
        shape_param[0] = np.random.choice(lexicon.rectangle_params) / (len(lexicon.rectangle_params) - 1)
    if shape_int in [diamond]:       
        shape_param[0] = np.random.choice(lexicon.diamond_params) / (len(lexicon.diamond_params) - 1)
    if shape_int in [trapezoid]:       
        shape_param[0] = np.random.choice(lexicon.trapezoid_params) / (len(lexicon.trapezoid_params) - 1)
    if shape_int in [tee]:       
        shape_param[0] = np.random.choice(lexicon.tee_params) / (len(lexicon.tee_params) - 1)
    return shape_param


def generate_transformation_params(lexicon: Lexicon, base_element, num_modifications = -1):
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

    # the shape features are the last four elements
    shape_features = base_element[-4:]

    # existing shape_features to transform
    scale_param = shape_features[scale]
    rotation_param = shape_features[rotation]
    shading_param = shape_features[shading]
    numerosity_param = shape_features[numerosity]

    analogy_scale_param = np.random.choice([i for i in lexicon.scale_values if i != int(scale_param * len(lexicon.scale_values))]) / len(lexicon.scale_values)
    analogy_rotation_param = np.random.choice([i for i in lexicon.rotation_values if i != int(rotation_param * len(lexicon.rotation_values))]) / len(lexicon.rotation_values)
    analogy_shading_param = np.random.choice([i for i in lexicon.shading_values if i != int(shading_param * len(lexicon.shading_values))]) / len(lexicon.shading_values)
    analogy_numerosity_param = np.random.choice([i for i in lexicon.numerosity_values if i != int(numerosity_param * len(lexicon.numerosity_values))]) / len(lexicon.numerosity_values)

    # scale, shading, rotation or numerosity
    features = np.zeros(4)
    # make 0-3 modifications
    if num_modifications == -1:
        num_modifications = np.random.randint(4)
    modifications = np.random.choice(range(4), num_modifications, replace=False)

    transformation_params = np.full(4, 0.5)
    for feature in modifications:
        features[feature] = 1.
        if feature in [scale]:
            feature_param = analogy_scale_param - scale_param
        if feature in [rotation]:
            feature_param = analogy_rotation_param - rotation_param
        if feature in [shading]:
            feature_param = analogy_shading_param - shading_param
        if feature in [numerosity]:
            feature_param = analogy_numerosity_param - numerosity_param
        # normalise to [0, 1]
        feature_param = feature_param / 2 + 0.5 
        transformation_params[feature] = feature_param

    return transformation_params


def display_one_random_2_by_2():
    matrix, test, transformation, analogy = generate_sandia_matrix()
    print(f'Test    = {test}')
    print(f'Analogy = {analogy}')
    print(f'Transformation = {np.round(transformation, 3)}')
    test_matrix(matrix, is_correct=True)


def display_one_random_2_by_3():
    matrix, test, transformation1, transformation2, analogy = generate_sandia_matrix_2_by_3()
    print(f'Test    = {test}')
    print(f'Analogy = {analogy}')
    print(f'Transformation1 = {np.round(transformation1, 3)}')
    print(f'Transformation2 = {np.round(transformation2, 3)}')
    test_matrix(matrix, is_correct=True)


def display_one_random_3_by_3():
    matrix, test, transformation1, transformation2, analogy1, analogy2 = generate_sandia_matrix_3_by_3()
    print(f'Test    = {test}')
    print(f'Analogy1 = {analogy1}')
    print(f'Analogy2 = {analogy2}')
    print(f'Transformation1 = {np.round(transformation1, 3)}')
    print(f'Transformation2 = {np.round(transformation2, 3)}')
    test_matrix(matrix, is_correct=None)


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
    transformation = np.concatenate((modification_parameters))
    analogy = np.concatenate((analogy_shape, shape_param, shape_features))
    # return matrix, sample, transformation, analogy
    return None, sample, transformation, analogy


#%%
#display_one_random_2_by_2()
#display_all_sandia_matrices(100, [0])
#print(sum(1 for i in generate_all_sandia_matrices([0], include_shape_variants = False)))
#display_one_random_2_by_3()
#display_one_random_3_by_3()

# lexicon = Lexicon()
# p, a = generate_base_elements_as_vector(lexicon)
# t = generate_transformation(lexicon, p, num_modifications = 1)
# print(np.round(p, 3))
# print(np.round(t, 3))
# print(np.round(a, 3))
#%%
