#%%
import cairo
#import dataclasses
import numpy as np
import os
import copy
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

def test_matrix(elements, cell_size = 64):
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

def generate_sandia_matrix():
    structure_gen = gen.StructureGenerator(
        branch = {
            'basic': 0.,
            'composite': 0.,
            'modified': 1.
        },
        modifier_num = {
            1: 1.,
            2: 0.,
            3: 0.
        }
    )
    routine_gen = gen.RoutineGenerator()
    decorator_gen = gen.DecoratorGenerator()

    # Generate an element. For now this will be a modified element with one modification.
    element = gen.generate_sandia_figure(structure_gen, routine_gen, decorator_gen)

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
    shape[shape_index] = 1
    shape_params = np.array([basic_element.params['r'] / 8])
    
    analogy_shape_index = list(routine_gen.routines.keys()).index(basic_analogy_element.routine)
    analogy_shape = np.zeros(6)
    analogy_shape[analogy_shape_index] = 1
    analogy_shape_params = np.array([basic_analogy_element.params['r'] / 8])

    initial_decoration = np.zeros(4)

    decorator = np.zeros(4)
    decorator_params = np.zeros(4)

    # Get the modification - TO DO: foreach modification
    for target in targets[1:]:
        modification = target(element)

        decorator_index = list(decorator_gen.decorators.keys()).index(modification.decorator)
        decorator[decorator_index] = 1
        decorator_params[decorator_index] = list(modification.params.values())[0]
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
    return matrix, test, transformation, analogy

matrix, test, transformation, analogy = generate_sandia_matrix()
print(f'Test    = {test}')
print(f'Analogy = {analogy}')
print(f'Transformation = {np.round(transformation, 3)}')
test_matrix(matrix)


#%%


#%%
