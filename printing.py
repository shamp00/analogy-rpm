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


from dataclasses import dataclass

# shapes
ellipse: int = 0
triangle: int = 1
rectangle: int = 2
trapezoid: int = 3
diamond: int = 4
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
    shapes = [ellipse, triangle, rectangle, trapezoid, diamond, tee]

    # routine for each shape
    routines = {
        ellipse: defs.ellipse,
        triangle: defs.triangle,
        rectangle: defs.rectangle,
        trapezoid: defs.trapezoid,
        diamond: defs.diamond,
        tee: defs.tee
    }

    # number of different shape parameters 
    shape_param_ranges = {
        ellipse: list(range(3)), 
        triangle: list(range(5)), 
        rectangle: list(range(3)), 
        trapezoid: list(range(5)), 
        diamond: list(range(3)), 
        tee: list(range(5))
    }        

    # available modifications
    modifications = [scale, rotation, shading, numerosity]

    decorators = {
        scale: defs.scale,
        rotation: defs.rotation,
        shading: defs.shading,
        numerosity: defs.numerosity
    }

    # available modification parameter values
    modification_param_ranges = {
        scale: list(range(0, 4)),
        rotation: list(range(0, 8)),
        shading: list(range(8, 0, -1)),
        numerosity: list(range(1, 5))
    }


def generate_base_elements_as_vectors(lexicon: Lexicon, num_analogies=1):
    """Generate a vector representing the starting shapes corresponding to 
    an element and an analogy"""

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

    # Create a vector like this [1 0 0 0 0 0] for shape1, say, ellipse
    shape_int, *analogy_shape_ints = np.random.choice(lexicon.shapes, 1 + num_analogies, replace=False)

    shape = generate_shape(shape_int)
    shape_param = generate_shape_params(lexicon, shape_int)

    shape_features = np.zeros(4)
    for modification in lexicon.modifications:
        modification_possible_values = lexicon.modification_param_ranges[modification]
        shape_features[modification] = normalized_random_choice(modification_possible_values)
 
    sample = np.concatenate((shape, shape_param, shape_features))

    analogies = []
    for analogy_shape_int in analogy_shape_ints:
        analogy_shape = generate_shape(analogy_shape_int)
        analogy_shape_param = generate_shape_params(lexicon, analogy_shape_int)

        analogy = np.concatenate((analogy_shape, analogy_shape_param, shape_features))
        
        assert (sample >= 0).all()
        assert (analogy <= 1).all()

        analogies.append(analogy)

    # return matrix, sample, transformation, analogy
    return tuple([sample] + analogies)

def generate_all_base_elements_as_vectors(lexicon: Lexicon):
    """Generate a vector representing the starting shapes corresponding to 
    an element and an analogy"""

    # Create a vector like this [1 0 0 0 0 0] for shape1, say, ellipse
    shape_int, analogy_shape_int = np.random.choice(lexicon.shapes, 2, replace=False)

    shape = generate_shape(shape_int)
    shape_param = generate_shape_params(lexicon, shape_int)

    analogy_shape = generate_shape(analogy_shape_int)
    analogy_shape_param = generate_shape_params(lexicon, analogy_shape_int)

    shape_features = np.zeros(4)
    for modification in lexicon.modifications:
        modification_possible_values = lexicon.modification_param_ranges[modification]
        shape_features[modification] = normalized_random_choice(modification_possible_values)
 
    sample = np.concatenate((shape, shape_param, shape_features))
    analogy = np.concatenate((analogy_shape, analogy_shape_param, shape_features))
    
    # return matrix, sample, transformation, analogy
    return sample, analogy


def normalize(x, possible_values):
    """Taxes a parameter and returns a corresponding value [0, 1]"""
    assert x in possible_values
    min_x = min(possible_values)
    max_x = max(possible_values)
    z = (x - min_x) / (max_x - min_x)
    assert z >= 0
    assert z <= 1
    return z


def denormalize(z, possible_values) -> int:
    """Taxes a normalized value between [0, 1] and returns the corresponding denormalized integer."""
    assert z >= 0
    assert z <= 1
    min_x = min(possible_values)
    max_x = max(possible_values)
    x_float = np.round(z * (max_x - min_x) + min_x, 3)
    assert x_float.is_integer()
    x = int(x_float)
    assert x in possible_values
    return x


def normalize_transformation(t: np.array) -> np.array:
    return t / 2 + 0.5


def denormalize_transformation(t: np.array) -> np.array:
    return (t - 0.5) * 2


def smooth_numerical_innaccuracies(x: float):
    attractors = [0., 0.25, 0.5, 0.75, 1.0]
    tol = 1e-15
    for a in attractors:
        x[abs(a - x) < tol] = a


def generate_shape(shape_int):
    shape = np.zeros(6)
    shape[shape_int] = 1.
    return shape


def generate_shape_params(lexicon: Lexicon, shape_int: int):
    shape_param = np.zeros(1)
    shape_param_range = lexicon.shape_param_ranges[shape_int]
    shape_param[0] = normalized_random_choice(shape_param_range) 
    return shape_param


def normalized_random_choice(all_possible_values, excluded_values = []):
    possible_values = list(set(all_possible_values) - set(excluded_values))
    return normalize(np.random.choice(possible_values), all_possible_values)


def generate_transformation_params(lexicon: Lexicon, base_element, num_modification_choices = [0,1,2,3]):
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

    transformation_params = np.zeros(4)

    # the existing shape features are the last four elements
    shape_features = base_element[-4:]

    # make 0-3 modifications
    num_modifications = np.random.choice(num_modification_choices)
    modifications = np.random.choice(range(4), num_modifications, replace=False)

    for modification, feature in enumerate(shape_features):
        if modification in modifications:
            # We choose a new value from the values in the lexicon assign the delta
            # from the current feature value. This ensures the resulting feature
            # value is also in the lexicon. 
            all_possible_values = lexicon.modification_param_ranges[modification]
            denormalized_feature = denormalize(feature, all_possible_values)
            # select any other (i.e., not the same as the current value) from all possible feature values
            transformed_feature = normalized_random_choice(all_possible_values, excluded_values=[denormalized_feature])
            # transformation will be non-zero [-1 to 1] and within the bounds of the existing shape's characteristics
            transformation = transformed_feature - feature
            if modification == rotation:
                # To do this properly, I think I'd have to have two elements for rotation, 
                # one for the sine of the angle of rotation and one for the cosine.
                pass
            assert transformation != 0
            assert transformation >= -1
            assert transformation <= 1
            transformation_params[modification] = transformation
            smooth_numerical_innaccuracies(transformation_params)

    # normalize the transformation vector
    normalized_transformation_params = normalize_transformation(transformation_params)
    assert np.all(normalized_transformation_params >= 0)
    assert np.all(normalized_transformation_params <= 1)
    assert (normalized_transformation_params != 0.5).sum() == num_modifications

    return normalized_transformation_params


def vector_to_element(lexicon: Lexicon, p: np.array) -> elt.Element:
    # shape to routine
    shape_int = np.argmax(p[:6])
    e = elt.BasicElement()
    routine_gen = gen.RoutineGenerator() 
    e.routine = lexicon.routines[shape_int]
    e.params = routine_gen.params[e.routine]
    # shape_param 
    normalized_shape_param = p[6]
    shape_param = denormalize(normalized_shape_param, lexicon.shape_param_ranges[shape_int])
    routine_param_list = list(routine_gen.params[e.routine]['r'].keys())
    e.params['r'] = routine_param_list[shape_param]
    # features
    normalized_features = p[-4:]
    features = [denormalize(f, lexicon.modification_param_ranges[i]) for i, f in enumerate(normalized_features)]
    feature_count = np.count_nonzero(normalized_features)
    if feature_count > 0:
        decorator_gen = gen.DecoratorGenerator()
        e = elt.ModifiedElement(e, *[elt.ElementModifier() for _ in range(feature_count)])
        j = 0
        for i, feature in enumerate(features):
            if normalized_features[i] > 0:
                decorator = lexicon.decorators[i]
                decorator_param = list(list(decorator_gen.params[decorator].values()).pop().keys())[lexicon.modification_param_ranges[i].index(feature) - 1]
                e.modifiers[j].decorator = decorator
                e.modifiers[j].params = decorator_gen.params[decorator]
                e.modifiers[j].params[list(e.modifiers[j].params.keys())[0]] = decorator_param
                j += 1
    return e

def target(p):
    # make a copy
    pattern = np.copy(p)
    assert (p >= 0).all()
    assert (p <= 1).all()
    assert np.array_equal(p, pattern)

    """Desired response function, target(pattern)"""
    shape = p[0:6]
    shape_param = p[6:7]
    shape_features = p[7:11]
    transformation = p[-4:]
    transformation_parameters = denormalize_transformation(transformation)
    assert np.array_equal(p, pattern)
    
    shape_features = np.add(shape_features, transformation_parameters)
    assert np.array_equal(p, pattern)

    smooth_numerical_innaccuracies(shape_features)

    if shape_features[rotation] > 1:
        shape_features[rotation] -= 1 + 1 / 7 # modulo 1 for rotation
    if shape_features[rotation] < 0:
        shape_features[rotation] += 1 + 1 / 7 # modulo 1 for rotation

    assert np.array_equal(p, pattern)
    
    assert np.array_equal(p, pattern)
     
    assert (shape_features >= 0).all()
    assert (shape_features <= 1).all()
    return np.concatenate((shape, shape_param, shape_features))


def generate_rpm_2_by_2_matrix(lexicon: Lexicon, num_modification_choices = [0,1,2,3]):
    p, a = generate_base_elements_as_vectors(lexicon)
    t = generate_transformation_params(lexicon, p, num_modification_choices=num_modification_choices)
    p2, a2 = target(np.concatenate([p, t])), target(np.concatenate([a, t]))
    vectors = [p, a, p2, a2]
    matrix = [vector_to_element(lexicon, v) for v in vectors]
    return matrix, p, t, a


def generate_rpm_2_by_3_matrix(lexicon: Lexicon, num_modification_choices = [1]):
    p, a = generate_base_elements_as_vectors(lexicon)
    t1 = generate_transformation_params(lexicon, p, num_modification_choices=num_modification_choices)
    p2, a2 = target(np.concatenate([p, t1])), target(np.concatenate([a, t1]))
    t2 = generate_transformation_params(lexicon, p2, num_modification_choices=num_modification_choices)
    p3, a3 = target(np.concatenate([p2, t2])), target(np.concatenate([a2, t2]))

    vectors = [p, a, p2, a2, p3, a3]
    matrix = [vector_to_element(lexicon, v) for v in vectors]
    return matrix, p, t1, t2, a


def generate_rpm_3_by_3_matrix(lexicon: Lexicon, num_modification_choices = [1]):
    p, a1, a2 = generate_base_elements_as_vectors(lexicon, num_analogies=2)
    t1 = generate_transformation_params(lexicon, p, num_modification_choices=num_modification_choices)
    p2, a2 = target(np.concatenate([p, t1])), target(np.concatenate([a1, t1]))
    t2 = generate_transformation_params(lexicon, p2, num_modification_choices=num_modification_choices)
    p3, a3 = target(np.concatenate([p2, t2])), target(np.concatenate([a2, t2]))

    vectors = [p, a1, p2, a2, p3, a3]
    matrix = [vector_to_element(lexicon, v) for v in vectors]
    return matrix, p, t1, t2, a1, a2


def display_one_random_2_by_2(lexicon: Lexicon=None, num_modification_choices=[0,1,2,3]):
    if not lexicon:
        lexicon = Lexicon()
    matrix, test, transformation, analogy = generate_rpm_2_by_2_matrix(lexicon, num_modification_choices=num_modification_choices)
    print(f'Test    = {test}')
    print(f'Analogy = {analogy}')
    print(f'Transformation = {np.round(transformation, 3)}')
    test_matrix(matrix, is_correct=True)


def display_one_random_2_by_3(lexicon: Lexicon=None, num_modification_choices=[0,1,2,3]):
    if not lexicon:
        lexicon = Lexicon()
    matrix, test, transformation1, transformation2, analogy = generate_rpm_2_by_3_matrix(lexicon, num_modification_choices=num_modification_choices)
    print(f'Test    = {test}')
    print(f'Analogy = {analogy}')
    print(f'Transformation1 = {np.round(transformation1, 3)}')
    print(f'Transformation2 = {np.round(transformation2, 3)}')
    test_matrix(matrix, is_correct=True)


def display_one_random_3_by_3(lexicon: Lexicon=None, num_modification_choices=[0,1,2,3]):
    if not lexicon:
        lexicon = Lexicon()
    matrix, test, transformation1, transformation2, analogy1, analogy2 = generate_rpm_3_by_3_matrix(lexicon, num_modification_choices=num_modification_choices)
    print(f'Test    = {test}')
    print(f'Analogy1 = {analogy1}')
    print(f'Analogy2 = {analogy2}')
    print(f'Transformation1 = {np.round(transformation1, 3)}')
    print(f'Transformation2 = {np.round(transformation2, 3)}')
    test_matrix(matrix, is_correct=None)

#%%
#display_one_random_2_by_2()
#display_all_sandia_matrices(100, [0])
#print(sum(1 for i in generate_all_sandia_matrices([0], include_shape_variants = False)))
#display_one_random_2_by_3()
display_one_random_3_by_3()

# lexicon = Lexicon()
# p, a = generate_base_elements_as_vector(lexicon)
# t = generate_transformation(lexicon, p, num_modifications = 1)
# print(np.round(p, 3))
# print(np.round(t, 3))
# print(np.round(a, 3))

# np.random.seed(0)
# lexicon = Lexicon()
# for x in range(100):
#     m, p, t, a = generate_rpm_2_by_2_matrix(lexicon, num_modification_choices=[1])
#     test_matrix(m)
