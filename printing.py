#%%
import copy
import os
from dataclasses import dataclass
from math import pi

import cairo
import numpy as np
from IPython.display import SVG, display

import pyRavenMatrices.element as elt
import pyRavenMatrices.lib.sandia.definitions as defs
import pyRavenMatrices.lib.sandia.generators as gen
import pyRavenMatrices.matrix as mat
import pyRavenMatrices.transformation as tfm

# pylint: disable-msg=E1101 
# E1101: Module 'cairo' has no 'foo' member - of course it has! :) 

def is_paperspace():
     # hack for detecting Paperspace Gradient
    return os.path.exists(f'../storage/')

def is_running_from_ipython():
    from IPython import get_ipython
    return get_ipython() is not None

def cell_path(cell):
    return os.path.join('.', cell.id + '.svg')    

def test_element(element, cell_size = 64):
    cell_margin = cell_size // 16

    cell_structure = mat.CellStructure("generated" + str(0), cell_size, cell_size, cell_margin, cell_margin)
    
    svg_width = cell_structure.width
    svg_height = cell_structure.height

    surface = cairo.SVGSurface(cell_path(cell_structure), svg_width, svg_height)
    ctx = cairo.Context(surface)
    # set colour of ink to middle grey
    #ctx.set_source_rgb(0.5, 0.5, 0.5)

    ctx.rectangle(0, 0, svg_width, svg_height)
    set_source_default(ctx)
    ctx.fill()
    ctx.set_source_rgb(0, 0, 0)        

    element.draw_in_context(ctx, cell_structure)

    ctx.stroke()
    surface.finish()
    display(SVG(cell_path(cell_structure)))


def test_matrix(elements, candidates=None, cell_size = 96, selected: int = None, filename_index = 0):
    if elements == None:
        return

    is_correct = None
    if selected:
        is_correct = selected == 0
    cell_margin = cell_size // 16
    padding = 6

    if len(elements) == 1:
        element1 = elements[0]
        cell_structure = mat.CellStructure("generated" + str(filename_index), cell_size, cell_size, cell_margin, cell_margin)

        svg_width = cell_structure.width + 2
        svg_height = cell_structure.height + 2

        surface = cairo.SVGSurface(cell_path(cell_structure), svg_width, svg_height)
        
        ctx = cairo.Context(surface)    
        ctx.rectangle(0, 0, svg_width, svg_height)
        set_source_default(ctx)
        ctx.fill()
        ctx.set_source_rgb(0, 0, 0)        

        ctx.translate(1, 1)    

        draw_element(ctx, cell_structure, element1)
        ctx.stroke()

        surface.finish()

        display(SVG(cell_path(cell_structure)))


    if len(elements) == 2:
        element1 = elements[0]
        element2 = elements[1]
        cell_structure = mat.CellStructure("generated" + str(0), cell_size, cell_size, cell_margin, cell_margin)

        svg_width = cell_structure.width * 2 + padding + 2
        svg_height = cell_structure.height + 2

        surface = cairo.SVGSurface(cell_path(cell_structure), svg_width, svg_height)
        
        ctx = cairo.Context(surface)    
        ctx.rectangle(0, 0, svg_width, svg_height)
        set_source_default(ctx)
        ctx.fill()
        ctx.set_source_rgb(0, 0, 0)        

        ctx.translate(1, 1)    

        draw_element(ctx, cell_structure, element1)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element2)
        ctx.stroke()

        surface.finish()

        display(SVG(cell_path(cell_structure)))

    if len(elements) == 4:
        element1 = elements[0]
        element2 = elements[1]
        element3 = elements[2]
        #element4 = elements[3]

        cell_structure = mat.CellStructure("generated" + str(0), cell_size, cell_size, cell_margin, cell_margin)

        svg_width = cell_structure.width * 2 + padding + 2
        svg_height = cell_structure.height * 2 + padding + 2
        surface = cairo.SVGSurface(cell_path(cell_structure), svg_width, svg_height)
        
        ctx = cairo.Context(surface)    
        ctx.rectangle(0, 0, svg_width, svg_height)
        set_source_default(ctx)
        ctx.fill()
        ctx.set_source_rgb(0, 0, 0)        

        ctx.translate(1, 1)    

        draw_element(ctx, cell_structure, element1)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element2)
        ctx.translate(-(cell_structure.width + padding), cell_structure.height + padding)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element3)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_element(ctx, cell_structure, elt.EmptyElement())
        ctx.stroke()

        surface.finish()

        display(SVG(cell_path(cell_structure)))

    if len(elements) == 6:
        element1 = elements[0]
        element2 = elements[1]
        element3 = elements[2]
        element4 = elements[3]
        element5 = elements[4]
        #element6 = elements[5]

        cell_structure = mat.CellStructure("generated" + str(0), cell_size, cell_size, cell_margin, cell_margin)

        svg_width = cell_structure.width * 3 + padding * 2 + 2
        svg_height = cell_structure.height * 2 + padding + 2
        surface = cairo.SVGSurface(cell_path(cell_structure), svg_width, svg_height)

        ctx = cairo.Context(surface)    
        ctx.rectangle(0, 0, svg_width, svg_height)
        set_source_default(ctx)
        ctx.fill()
        ctx.set_source_rgb(0, 0, 0)        

        ctx.translate(1, 1)    
        
        draw_element(ctx, cell_structure, element1)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element2)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element3)
        ctx.translate(-2 * (cell_structure.width + padding), cell_structure.height + padding)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element4)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element5)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_element(ctx, cell_structure, elt.EmptyElement())
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
        #element9 = elements[8]

        cell_structure = mat.CellStructure("generated" + str(0), cell_size, cell_size, cell_margin, cell_margin)

        svg_width = cell_structure.width * 3 + padding * 2 + 2
        svg_height = cell_structure.height * 3 + padding * 2 + 2
        surface = cairo.SVGSurface(cell_path(cell_structure), svg_width, svg_height)
        
        ctx = cairo.Context(surface)    
        ctx.rectangle(0, 0, svg_width, svg_height)
        set_source_default(ctx)
        ctx.fill()
        ctx.set_source_rgb(0, 0, 0)        

        ctx.translate(1, 1)    

        draw_element(ctx, cell_structure, element1)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element2)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element3)
        ctx.translate(-2 * (cell_structure.width + padding), cell_structure.height + padding)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element4)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element5)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element6)
        ctx.translate(-2 * (cell_structure.width + padding), cell_structure.height + padding)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element7)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_element(ctx, cell_structure, element8)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_element(ctx, cell_structure, elt.EmptyElement())
        ctx.stroke()

        surface.finish()

        display(SVG(cell_path(cell_structure)))

    if candidates:
        element0 = candidates[0]
        element1 = candidates[1]
        element2 = candidates[2]
        element3 = candidates[3]
        element4 = candidates[4]
        element5 = candidates[5]
        element6 = candidates[6]
        element7 = candidates[7]

        padding = 6
        v_padding = 18
        cell_structure = mat.CellStructure("candidates" + str(0), cell_size, cell_size, cell_margin, cell_margin)

        svg_width = cell_structure.width * 4 + padding * 3 + 2
        svg_height = cell_structure.height * 2 + v_padding * 3 + 2
        surface = cairo.SVGSurface(cell_path(cell_structure), svg_width, svg_height)
        
        ctx = cairo.Context(surface)    
        ctx.rectangle(0, 0, svg_width, svg_height)
        set_source_default(ctx)
        ctx.fill()

        ctx.translate(1, 1)    
        draw_candidate_cell(ctx, cell_structure, selected, 0)
        element0.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_candidate_cell(ctx, cell_structure, selected, 1)
        element1.draw_in_context(ctx, cell_structure)    
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_candidate_cell(ctx, cell_structure, selected, 2)
        element2.draw_in_context(ctx, cell_structure)    
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_candidate_cell(ctx, cell_structure, selected, 3)
        element3.draw_in_context(ctx, cell_structure)
        ctx.translate(-3 * (cell_structure.width + padding), cell_structure.height + v_padding)    
        ctx.stroke()

        ctx.translate(0, v_padding)    
        draw_candidate_cell(ctx, cell_structure, selected, 4)
        element4.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_candidate_cell(ctx, cell_structure, selected, 5)
        element5.draw_in_context(ctx, cell_structure)    
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_candidate_cell(ctx, cell_structure, selected, 6)
        element6.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width + padding, 0)    
        ctx.stroke()

        draw_candidate_cell(ctx, cell_structure, selected, 7)
        element7.draw_in_context(ctx, cell_structure)
        ctx.translate(cell_structure.width + padding, cell_structure.height + v_padding)    
        ctx.stroke()

        surface.finish()

        display(SVG(cell_path(cell_structure)))


def test_base_elements(elements, cell_size = 96):
    cell_margin = cell_size // 16
    if elements == None:
        return

    element0 = elements[0]
    element1 = elements[1]
    element2 = elements[2]
    element3 = elements[3]
    element4 = elements[4]
    element5 = elements[5]
    element6 = elements[6]
    element7 = elements[7]
    element8 = elements[8]
    element9 = elements[9]
    element10 = elements[10]
    element11 = elements[11]
    element12 = elements[12]
    element13 = elements[13]
    element14 = elements[14]
    element15 = elements[15]
    element16 = elements[16]
    element17 = elements[17]

    cell_structure = mat.CellStructure("generated" + str(0), cell_size, cell_size, cell_margin, cell_margin)

    surface = cairo.SVGSurface(cell_path(cell_structure), cell_structure.width * 4, cell_structure.height * 6)
    
    ctx = cairo.Context(surface)    

    selected = -1

    # ellipse
    draw_candidate_cell(ctx, cell_structure, selected, 0)
    element0.draw_in_context(ctx, cell_structure)
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 1)
    element1.draw_in_context(ctx, cell_structure)    
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 2)
    elt.EmptyElement().draw_in_context(ctx, cell_structure)    
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 3)
    elt.EmptyElement().draw_in_context(ctx, cell_structure)    
    ctx.translate(-3 * cell_structure.width, cell_structure.height)    
    ctx.stroke()

    # rectangle
    draw_candidate_cell(ctx, cell_structure, selected, 8)
    element6.draw_in_context(ctx, cell_structure)
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 9)
    element7.draw_in_context(ctx, cell_structure)
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 10)
    elt.EmptyElement().draw_in_context(ctx, cell_structure)    
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 11)
    elt.EmptyElement().draw_in_context(ctx, cell_structure)    
    ctx.translate(-3 * cell_structure.width, cell_structure.height)    
    ctx.stroke()

    # diamond
    draw_candidate_cell(ctx, cell_structure, selected, 16)
    element12.draw_in_context(ctx, cell_structure)
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 17)
    element13.draw_in_context(ctx, cell_structure)    
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 18)
    elt.EmptyElement().draw_in_context(ctx, cell_structure)    
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 19)
    elt.EmptyElement().draw_in_context(ctx, cell_structure)    
    ctx.translate(-3 * cell_structure.width, cell_structure.height)    
    ctx.stroke()

    # triangle
    draw_candidate_cell(ctx, cell_structure, selected, 4)
    element2.draw_in_context(ctx, cell_structure)    
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 5)
    element3.draw_in_context(ctx, cell_structure)
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 6)
    element4.draw_in_context(ctx, cell_structure)
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 7)
    element5.draw_in_context(ctx, cell_structure)    
    ctx.translate(-3 * cell_structure.width, cell_structure.height)    
    ctx.stroke()

    # trapezoid
    draw_candidate_cell(ctx, cell_structure, selected, 12)
    element8.draw_in_context(ctx, cell_structure)    
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 13)
    element9.draw_in_context(ctx, cell_structure)
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 14)
    element10.draw_in_context(ctx, cell_structure)
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 15)
    element11.draw_in_context(ctx, cell_structure)    
    ctx.translate(-3 * cell_structure.width, cell_structure.height)    
    ctx.stroke()

    # tee
    draw_candidate_cell(ctx, cell_structure, selected, 20)
    element14.draw_in_context(ctx, cell_structure)    
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 21)
    element15.draw_in_context(ctx, cell_structure)
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 22)
    element16.draw_in_context(ctx, cell_structure)
    ctx.translate(cell_structure.width, 0)    
    ctx.stroke()

    draw_candidate_cell(ctx, cell_structure, selected, 23)
    element17.draw_in_context(ctx, cell_structure)    
    ctx.translate(-3 * cell_structure.width, cell_structure.height)    
    ctx.stroke()

    surface.finish()

    display(SVG(cell_path(cell_structure)))

def set_source_default(ctx):
    ctx.set_source_rgb(1., 1., 1.)

def draw_element_cell(ctx, cell_structure):
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(1)
    ctx.rectangle(0, 0, cell_structure.width, cell_structure.height)
    ctx.stroke()    
    ctx.set_line_width(2)

def draw_candidate_cell(ctx, cell_structure, selected, cell_number):
    draw_element_cell(ctx, cell_structure)
    if selected == cell_number:
        if selected == 0:    
            ctx.set_source_rgb(0.9, 1.0, 0.9)
        else:
            ctx.set_source_rgb(1.0, 0.9, 0.9)
    else:
        set_source_default(ctx)
    ctx.rectangle(0, 0, cell_structure.width, cell_structure.height)
    ctx.fill()
    ctx.set_source_rgb(0, 0, 0)
    
    # Label
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    label = labels[cell_number]
    
    font = ctx.get_font_face()
    font_family = font.get_family()
    ctx.select_font_face(font_family, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(13)

    (x, y, width, height, dx, dy) = ctx.text_extents(label)
    x_shift = (cell_structure.width - width) / 2
    y_shift = (cell_structure.height + 13)

    ctx.translate(x_shift, y_shift)
    ctx.show_text(label)
    ctx.translate(-x_shift, -y_shift)

def draw_element(ctx, cell_structure, element1):
    draw_element_cell(ctx, cell_structure)
    element1.draw_in_context(ctx, cell_structure)



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
        ellipse: [0, 1], 
        triangle: [0, 1, 3, 4], 
        rectangle: [0, 1], 
        trapezoid: [0, 1, 3, 4], 
        diamond: [0, 1], 
        tee: [0, 1, 3, 4]
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
        shading: list(range(0, 8)),
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
    for shape_int in lexicon.shapes:
        shape = generate_shape(shape_int)

        shape_param = np.zeros(1)
        shape_param_range = lexicon.shape_param_ranges[shape_int]
        for choice in shape_param_range:
            shape_param[0] = normalize(choice, shape_param_range) 

            shape_features = np.zeros(4)  

            # shading = 6
            modification_possible_values = lexicon.modification_param_ranges[shading]
            shape_features[shading] = normalize(6, modification_possible_values)

            sample = np.concatenate((shape, shape_param, shape_features))
            
            # return matrix, sample, transformation, analogy
            yield sample


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


def random_choice(all_possible_values, excluded_values=[]):
    possible_values = list(set(all_possible_values) - set(excluded_values))
    return np.random.choice(possible_values)


def normalized_random_choice(all_possible_values, excluded_values = []):
    choice = random_choice(all_possible_values, excluded_values=excluded_values)
    return normalize(choice, all_possible_values)


def generate_transformation_params(lexicon: Lexicon, base_element, num_modification_choices=[0,1,2,3]):
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

    # the existing shape features are the seventh to the eleventh elements
    shape_features = base_element[7:11]

    # make 0-3 modifications
    num_modifications = np.random.choice(num_modification_choices)
    modifications = np.random.choice([scale, rotation, shading, numerosity], num_modifications, replace=False)

    for modification, feature in enumerate(shape_features):
        if modification in modifications:
            # We choose a new value from the values in the lexicon assign the delta
            # from the current feature value. This ensures the resulting feature
            # value is also in the lexicon. 
            transformation = generate_transformation_for_modification_type(lexicon, modification, feature)
            transformation_params[modification] = transformation
            smooth_numerical_innaccuracies(transformation_params)

    # normalize the transformation vector
    normalized_transformation_params = normalize_transformation(transformation_params)
    assert np.all(normalized_transformation_params >= 0)
    assert np.all(normalized_transformation_params <= 1)
    assert (normalized_transformation_params != 0.5).sum() == num_modifications

    return normalized_transformation_params

def generate_transformation_for_modification_type(lexicon, modification, current_feature, excluded_features = []):
    # We choose a new value from the values in the lexicon assign the delta
    # from the current feature value. This ensures the resulting feature
    # value is also in the lexicon. 
    all_possible_values = lexicon.modification_param_ranges[modification]
    denormalized_feature = denormalize(current_feature, all_possible_values)
    denormalized_excluded_features = [denormalize(x, all_possible_values) for x in excluded_features]
    # union of excluded feature values
    all_excluded_values = list(set(denormalized_excluded_features) | set([denormalized_feature]))
    while True:
        # select any other (i.e., not the same as the current value) from all possible feature values
        transformed_feature = normalized_random_choice(all_possible_values, excluded_values=all_excluded_values)
        # transformation will be non-zero [-1 to 1] and within the bounds of the existing shape's characteristics
        transformation = transformed_feature - current_feature
        if modification == rotation:
            # To do this properly, I think I'd have to have two elements for rotation, 
            # one for the sine of the angle of rotation and one for the cosine.
            pass
        if excluded_features == [] or modification != shading or abs(transformation) > 1/7:
            # Make sure shading difference is above threshold
            break


    assert transformation != 0
    assert transformation >= -1
    assert transformation <= 1
    return transformation


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
    normalized_features = p[7:11]
    features = [denormalize(f, lexicon.modification_param_ranges[i]) for i, f in enumerate(normalized_features)]
    feature_count = np.count_nonzero(normalized_features)
    if feature_count > 0:
        # Adjust scale output
        scale_params = {
            'factor': {
                (3 / 4): 1 / 3,
                (2 / 4): 1 / 3,
                (1 / 3): 1 / 3,
            } 
        }        
        decorator_gen = gen.DecoratorGenerator(scale_params=scale_params)
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

def target_simple(p):
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

    shape_features = np.clip(shape_features, 0., 1.)
    
    assert np.array_equal(p, pattern)     
    assert (shape_features >= 0).all()
    assert (shape_features <= 1).all()

    return np.concatenate((shape, shape_param, shape_features))


def target_complex(p):
    """Target for experiments with CA(t2) output"""
    transformation = p[-4:]
    output = target_simple(p)
    return np.concatenate((output, transformation))

def target(p):
    return target_complex(p)

def generate_candidates(lexicon, t, a, a2, d1, d2):
    """Generate candidates for analogy completion."""
    d1 = np.copy(d1)
    d2 = np.copy(d2)
    # make sure the range of possible transformations on 
    # the distractors is the same as those for the analogy 
    d1[7:11] = a[7:11]
    d2[7:11] = a[7:11]

    # Pick two of the existing transformations (right transformations) 
    # and two of the zero transformations (wrong transformations).
    # If there are not enough, pick at random.
    num_existing_transformations = np.count_nonzero(t != 0.5)
    no_modifications = np.where(t == 0.5)[0]
    yes_modifications = np.where(t != 0.5)[0]
    if num_existing_transformations == 0:
        right_mod1, wrong_mod1, wrong_mod2 = np.random.choice(4, 3, replace=False)
    elif num_existing_transformations == 1:
        right_mod1 = np.random.choice(yes_modifications)
        wrong_mod1, wrong_mod2 = np.random.choice(no_modifications, 2, replace=False)
    elif num_existing_transformations == 2:
        right_mod1 = np.random.choice(yes_modifications)
        wrong_mod1, wrong_mod2 = np.random.choice(no_modifications, 2, replace=False)
    elif num_existing_transformations == 3:
        right_mod1, wrong_mod1 = np.random.choice(yes_modifications, 2, replace=False)
        wrong_mod2 = np.random.choice(no_modifications)

    current_features = a[7:11]
    transformed_features = a2[7:11]

    # 1 correct answer
    c0 = a2

    # 1 x right base element, right transformation, wrong parameter (wrong direction)
    # - for a random nonzero element in t, generate an alternative
    modified_t = generate_distractor_transformation(lexicon, t, right_mod1, current_features, transformed_features)    
    c1 = target(np.concatenate([a, modified_t]))

    # 2 x right base element, wrong transformation
    # - for a random nonzero element in t, replace a zero element with a different modification type
    # e.g., [0, 0.5, 0, 0] might be [0, 0, 0.25, 0]
    # e.g., [0.7, 0.7, 0, 0.7] might be [0.7, 0.7, 0.2, 0]
    d_tf1 = generate_distractor_transformation(lexicon, t, wrong_mod1, current_features, transformed_features)
    c2 = target(np.concatenate([a, d_tf1]))
    d_tf2 = generate_distractor_transformation(lexicon, t, wrong_mod2, current_features, transformed_features)
    c3 = target(np.concatenate([a, d_tf2]))

    # 1 x exemplar element, i.e., wrong element, no transformation
    c4 = d1

    # 1 x exemplar target, i.e., wrong element, right transformation
    if (t == 0.5).all():   
        modified_t = generate_distractor_transformation(lexicon, t, np.random.choice(4), current_features, transformed_features)     
        c5 = target(np.concatenate([d1, modified_t]))
    else:
        c5 = target(np.concatenate([d1, t]))

    # 1 x wrong base element, wrong transformation
    c6 = target(np.concatenate([d1, d_tf1]))
    c7 = target(np.concatenate([d2, d_tf2]))

    candidates = [c0[:11], c1[:11], c2[:11], c3[:11], c4[:11], c5[:11], c6[:11], c7[:11]]
    return candidates


def generate_distractor_transformation(lexicon, current_transformation, modification_type, current_features, transformed_features):
    new_value = generate_transformation_for_modification_type(lexicon, modification_type, current_features[modification_type], [transformed_features[modification_type]])
    normalized_new_value = normalize_transformation(new_value)
    distractor_transformation = np.copy(current_transformation)
    distractor_transformation[modification_type] = normalized_new_value
    return distractor_transformation


def generate_rpm_2_by_2_matrix(lexicon: Lexicon, num_modification_choices = [0,1,2,3]):
    p, a, d1, d2 = generate_base_elements_as_vectors(lexicon, 3)
    t = generate_transformation_params(lexicon, p, num_modification_choices=num_modification_choices)
    p2, a2 = target(np.concatenate([p, t])), target(np.concatenate([a, t]))
    vectors = [p, p2, a, a2]

    # generate candidates

    # Pick two of the existing transformations (right transformations) 
    # and two of the zero transformations (wrong transformations).
    # If there are not enough, pick at random.
    candidates = generate_candidates(lexicon, t, a, a2, p, d2)

    matrix = [[vector_to_element(lexicon, v) for v in vectors], [vector_to_element(lexicon, c) for c in candidates]]
    return matrix, candidates, p, t, a


def generate_rpm_2_by_3_matrix(lexicon: Lexicon, num_modification_choices=[1]):
    # Matrix is as follows
    # -------------------
    # | p11 | p12 | p13 |
    # -------------------
    # | a21 | a22 | a23 |
    # -------------------
    p1, a1, d1, d2 = generate_base_elements_as_vectors(lexicon, 3)
    t1 = generate_transformation_params(lexicon, p1, num_modification_choices=num_modification_choices)
    p11 = np.concatenate([p1, t1])
    a21 = np.concatenate([a1, t1])

    p2, a2 = target(p11)[:11], target(a21)[:11]
    t2 = generate_transformation_params(lexicon, p2, num_modification_choices=num_modification_choices)
    p12 = np.concatenate([p2, t2])
    a22 = np.concatenate([a2, t2])
    
    p3, a3 = target(p12)[:11], target(a22)[:11]

    vectors = [p1, p2, p3, a1, a2, a3]

    candidates = generate_candidates(lexicon, t2, a2, a3, p1, d2)

    matrix = [[vector_to_element(lexicon, v) for v in vectors], [vector_to_element(lexicon, c) for c in candidates]]

    return matrix, candidates, p11, t1, t2, a21


def generate_rpm_3_by_3_matrix(lexicon: Lexicon, num_modification_choices = [1], distribution_styles = ['none', '3-left', '3-right']):
    # Matrix is as follows
    # -------------------
    # | p11 | p12 | p13 |
    # -------------------
    # | a21 | a22 | a23 |
    # -------------------
    # | a31 | a32 | a33 |
    # -------------------
    
    p1, a1, b1, d1, d2 = generate_base_elements_as_vectors(lexicon, num_analogies=4)
    t1 = generate_transformation_params(lexicon, p1, num_modification_choices=num_modification_choices)
    p11 = np.concatenate([p1, t1])
    a21 = np.concatenate([a1, t1])
    a31 = np.concatenate([b1, t1])

    p2, a2, b2 = target(p11)[:11], target(a21)[:11], target(a31)[:11]
    t2 = generate_transformation_params(lexicon, p2, num_modification_choices=num_modification_choices)
    p12 = np.concatenate([p2, t2])
    a22 = np.concatenate([a2, t2])
    a32 = np.concatenate([b2, t2])

    p3, a3, b3 = target(p12)[:11], target(a22)[:11], target(a32)[:11]

    vectors = [p1, p2, p3, a1, a2, a3, b1, b2, b3]

    candidates = generate_candidates(lexicon, t2, b2, b3, p11, a2)

    distribution_style = np.random.choice(distribution_styles)
    if distribution_style == 'none':
        vectors = [p1, p2, p3, a1, a2, a3, b1, b2, b3]
    elif distribution_style == '3-left':
        # Left shift row 1 of m twice
        # Left shift row 2 of m once
        vectors = [p3, p1, p2, a2, a3, a1, b1, b2, b3]
    elif distribution_style == '3-right':
        # Right shift row 1 of m twice
        # Right shift row 2 of m once
        vectors = [p2, p3, p1, a3, a1, a2, b1, b2, b3]
    # TODO: '2-left', '2-right'

    matrix = [[vector_to_element(lexicon, v) for v in vectors], [vector_to_element(lexicon, c) for c in candidates]]

    return matrix, candidates, p11, t1, t2, a21, a31, distribution_style


def display_one_random_training_pattern(lexicon: Lexicon=None, num_modification_choices=[0,1,2,3], selected = None):
    if not lexicon:
        lexicon = Lexicon()
    m, _candidates, test, transformation, analogy = generate_rpm_2_by_2_matrix(lexicon, num_modification_choices=num_modification_choices)
    print(f'Test    = {test}')
    print(f'Analogy = {analogy}')
    print(f'Transformation = {np.round(transformation, 3)}')

    test_matrix(m[0][0:2], None, selected=selected)


def display_one_random_2_by_2(lexicon: Lexicon=None, num_modification_choices=[0,1,2,3], selected = None):
    if not lexicon:
        lexicon = Lexicon()
    m, _candidates, test, transformation, analogy = generate_rpm_2_by_2_matrix(lexicon, num_modification_choices=num_modification_choices)
    print(f'Test    = {test}')
    print(f'Analogy = {analogy}')
    print(f'Transformation = {np.round(transformation, 3)}')
    
    test_matrix(m[0], m[1], selected=selected)


def display_one_random_2_by_3(lexicon: Lexicon=None, num_modification_choices=[1,2], selected = None):
    if not lexicon:
        lexicon = Lexicon()
    m, _candidates, test, transformation1, transformation2, analogy = generate_rpm_2_by_3_matrix(lexicon, num_modification_choices=num_modification_choices)
    print(f'Test    = {test}')
    print(f'Analogy = {analogy}')
    print(f'Transformation1 = {np.round(transformation1, 3)}')
    print(f'Transformation2 = {np.round(transformation2, 3)}')

    test_matrix(m[0], m[1], selected=selected)


def display_one_random_3_by_3(lexicon: Lexicon=None, num_modification_choices=[1,2], selected = None):
    if not lexicon:
        lexicon = Lexicon()
    m, _candidates, test, transformation1, transformation2, analogy1, analogy2 = generate_rpm_3_by_3_matrix(lexicon, num_modification_choices=num_modification_choices)
    print(f'Test    = {test}')
    print(f'Analogy1 = {analogy1}')
    print(f'Analogy2 = {analogy2}')
    print(f'Transformation1 = {np.round(transformation1, 3)}')
    print(f'Transformation2 = {np.round(transformation2, 3)}')

    test_matrix(m[0], m[1], selected=selected)


def display_one_random_distribution_of_3_by_3(lexicon: Lexicon=None, num_modification_choices=[1,2], selected = None):
    if not lexicon:
        lexicon = Lexicon()
    m, _candidates, test, transformation1, transformation2, analogy1, analogy2, distribution_style = generate_rpm_3_by_3_matrix(lexicon, num_modification_choices=num_modification_choices)

    print(f'Test    = {test}')
    print(f'Analogy1 = {analogy1}')
    print(f'Analogy2 = {analogy2}')
    print(f'Transformation1 = {np.round(transformation1, 3)}')
    print(f'Transformation2 = {np.round(transformation2, 3)}')
    print(f'Distribution style = {distribution_style}')

    test_matrix(m[0], m[1], selected=selected)
    
def display_all_base_elements(lexicon: Lexicon=None):
    if not lexicon:
        lexicon = Lexicon()
    elements = [vector_to_element(lexicon, v) for v in generate_all_base_elements_as_vectors(lexicon)]
    test_base_elements(elements)

def display_one_basic_3_by_3():
    lexicon = Lexicon()

    p1 = [0., 1., 0., 0., 0., 0., 3/4, 1., 0., 7/7, 0.]
    a1 = [0., 0., 0., 0., 1., 0., 0., 1., 0., 3/7, 0.]
    b1 = [0., 0., 0., 0., 0., 1., 3/4, 1., 0., 0/7, 0.]

    tf = np.asarray([1/3, 4/7, 0.5, 0.5]) 
    p2 = target(np.concatenate((p1[:11], tf)))
    p3 = target(np.concatenate((p2[:11], tf)))

    a2 = target(np.concatenate((a1[:11], tf)))
    a3 = target(np.concatenate((a2[:11], tf)))

    b2 = target(np.concatenate((b1[:11], tf)))
    b3 = target(np.concatenate((b2[:11], tf)))

    vectors = [p1,p2,p3,a1,a2,a3,b1,b2,b3]
    candidate_vectors = generate_candidates(lexicon, tf, b2, b3, a2, p2)

    elements, candidates = [vector_to_element(lexicon, p=v) for v in vectors], [vector_to_element(lexicon, p=c) for c in candidate_vectors]

    #for i, element in enumerate(elements):
    #    test_matrix([element], filename_index=i+1)

    test_matrix(elements, candidates=candidates)

def display_one_basic_2_by_2():
    lexicon = Lexicon()

    p1 = [0., 1., 0., 0., 0., 0., 3/4, 1/3, 0., 5/7, 0.]
    a1 = [0., 0., 0., 0., 1., 0., 0., 1/3, 0., 5/7, 0.]

    tf = np.asarray([0.5, 3/7, 1/7, 0.5]) 
    p2 = target(np.concatenate((p1[:11], tf)))
    a2 = target(np.concatenate((a1[:11], tf)))

    vectors = [p1,p2,a1,a2]
    candidate_vectors = generate_candidates(lexicon, tf, a1, a2, p1, p2)

    elements, candidates = [vector_to_element(lexicon, p=v) for v in vectors], [vector_to_element(lexicon, p=c) for c in candidate_vectors]

    #elements = [vector_to_element(lexicon, a2)]
    #test_matrix(elements)

    test_matrix(elements, candidates=candidates)

def display_one_composite_3_by_3():
    e1 = vector_to_element(Lexicon(), [0., 0., 1., 0., 0., 0., 0., 0., 2/7, 0., 0.])
    e2 = vector_to_element(Lexicon(), [1., 0., 0., 0., 0., 0., 0., 1/3, 2/7, 0., 0.])
    element1 = elt.CompositeElement(e1, e2)

    e2 = vector_to_element(Lexicon(), [0., 1., 0., 0., 0., 0., 1/4, 1., 0., 0., 0.])
    element2 = elt.CompositeElement(e1, e2)

    element3 = elt.CompositeElement(element1, element2)
    element4 = element2
    element5 = e2
    element6 = element2
    element7 = element3
    element8 = element2
    element9 = elt.EmptyElement()

    e3 = vector_to_element(Lexicon(), [1., 0., 0., 0., 0., 0., 0., 1., 2/7, 0., 0.])
    e4 = vector_to_element(Lexicon(), [1., 0., 0., 0., 0., 0., 1., 1/3, 2/7, 0., 0.])
    e5 = vector_to_element(Lexicon(), [0., 1., 0., 0., 0., 0., 1/4, 2/3, 0., 0., 0.])

    candidate1 = element1
    candidate2 = elt.CompositeElement(element1, e2)
    candidate3 = element2
    candidate4 = elt.CompositeElement(element1, e3)
    candidate5 = element3
    candidate6 = elt.CompositeElement(e3, e4)
    candidate7 = elt.CompositeElement(e1, e3)
    candidate8 = elt.EmptyElement()

    candidates = [candidate1, candidate2, candidate3, candidate4, candidate5, candidate6, candidate7, candidate8]

    test_matrix([element1, element2, element3, element4, element5, element6, element7, element8, element9], candidates=candidates)


def display_one_progressive_3_by_3():
    e1 = vector_to_element(Lexicon(), [0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.])
    e2 = vector_to_element(Lexicon(), [0., 0., 0., 0., 1., 0., 1., 0., 0., 1/7, 0.])
    e3 = vector_to_element(Lexicon(), [0., 0., 0., 0., 1., 0., 1., 0., 0., 3/7, 0.])
    e4 = vector_to_element(Lexicon(), [0., 0., 0., 0., 1., 0., 1., 0., 0., 5/7, 0.])
    e5 = vector_to_element(Lexicon(), [0., 0., 0., 0., 1., 0., 1., 0., 0., 7/7, 0.])

    element1 = e1
    element2 = e2
    element3 = e3
    element4 = e2
    element5 = e3
    element6 = e4
    element7 = e3
    element8 = e4
    element9 = elt.EmptyElement()

    c1 = vector_to_element(Lexicon(), [0., 0., 0., 0., 1., 0., 1., 2/3, 0., 3/7, 0.])
    c2 = vector_to_element(Lexicon(), [0., 0., 0., 0., 1., 0., 1., 2/3, 0., 7/7, 0.])
    c3 = vector_to_element(Lexicon(), [0., 0., 0., 0., 1., 0., 1., 2/3, 0., 5/7, 0.])

    candidate1 = e5
    candidate2 = e1
    candidate3 = c3
    candidate4 = c1
    candidate5 = e3
    candidate6 = c2
    candidate7 = e2
    candidate8 = elt.EmptyElement()


    candidates = [candidate1, candidate2, candidate3, candidate4, candidate5, candidate6, candidate7, candidate8]


    test_matrix([element1, element2, element3, element4, element5, element6, element7, element8, element9], candidates=candidates)


#%%

# np.random.seed(0)
# lexicon = Lexicon()
# for x in range(10):
#     display_one_random_3_by_3()

#display_all_base_elements()
#display_one_random_training_pattern(num_modification_choices=[3])
#display_one_random_2_by_2(num_modification_choices=[2])
#display_one_random_2_by_3()
#display_one_random_distribution_of_3_by_3(num_modification_choices=[3])


# p1 = np.asarray([0., 0., 0., 1., 0., 0., 0.75, 1/3, 0/7, 1., 0., 0.5, 5/7, 2/7, 1.])
# a1 = np.asarray([0., 0., 0., 0., 1., 0., 1., 1/3, 7/7, 1., 0., 0.5, 5/7, 2/7, 1.])
# b1 = np.asarray([0., 0., 0., 0., 0., 1., 0.75, 1/3, 6/7, 1., 0., 0.5, 5/7, 2/7, 1.])

# tf = np.asarray([0.5, 4/7, 2/7, 0])
# tf2 = np.asarray([0.5, 4/7, 2/7, 0])

# p2 = target(np.concatenate((p1, tf)))[:11]
# a2 = target(np.concatenate((a1, tf)))[:11]
# b2 = target(np.concatenate((b1, tf)))[:11]

# p3 = target(np.concatenate((p2, tf2)))[:11]
# a3 = target(np.concatenate((a2, tf2)))[:11]
# b3 = target(np.concatenate((b2, tf2)))[:11]

# vectors = [p1,p2,p3,a1,a2,a3,b1,b2,b3]
# elements = [vector_to_element(Lexicon(), v) for v in vectors]

# for i, element in enumerate(elements):
#     test_matrix([element], filename_index=i+1)

# test_matrix(elements)

#display_one_composite_3_by_3()
#display_one_progressive_3_by_3()

#display_one_basic_3_by_3()
#a = target(np.asarray([0.00,0.00,0.00,0.00,0.00,1.00,0.75,0.00,1/7,5/7,0.00,2/3,4/7,2/7,1/2]))
#test_matrix([vector_to_element(Lexicon(),a)])
#display_one_basic_3_by_3()



#%%
