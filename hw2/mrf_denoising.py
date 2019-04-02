# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:35:26 2017
@author: carmonda

Updated on Apr 2
@authors: Shahar Azulay, 039764063; Guy Oren, 302764956; Eitan-Hai Mashiah, 206349045
"""

import sys
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import argparse 

############## CONSTANTS ################
# Program variables
POSSIBLE_VALUES = [-1, 1]
PLOT = False

# LBP variables
CONVERGENCE_ATOL = 10**(-15)
MAX_ITERATIONS = 100
ALPHA = 0.1
BETA = 0.5
##########################################


class Vertex(object):
    
    def __init__(self, name='', y=None, neighbors=None, messages=None):
        """
        Params:
        --------
        name: string, the name of the vertex
        y: int (1 or -1), the observed value of the vertex (default: None)
        neighbors: set, of neighbouring vertices (default: None)
        messages: dict, mapping neighbouring vertices to their input messages (default: None)
        """
        self.name = name
        self._y = y
        
        if neighbors is None: 
            neighbors = set()
        
        if messages is None:
            messages = {}
            
        self._neighbors = neighbors
        self._messages = messages
        
    def add_neighbor(self, vertex):
        self._neighbors.add(vertex)
        
    def remove_neighbor(self, vertex):
        self._neighbors.remove(vertex)
    
    def update_message(self, msg, neighbor, atol=None):
        """ 
        Update a message from a neighbouring vertex.
        
        Params:
        --------
        msg: dict, represeting the messages (for every possible believed value)
        neighbor: Vertex, the source neighbouring vertex.
        atol: float, absolute tolerance under which no update is made (default: None)
        """
        should_update = True
        
        before_msg = self._messages[neighbor] if neighbor in self._messages else None
        
        if atol and before_msg:
            for x_i in POSSIBLE_VALUES:
                should_update = (abs(msg[x_i] - before_msg[x_i]) > atol)
            
        if should_update:
            self._messages[neighbor] = msg
        
        return should_update
        
    def get_belief(self):
        """
        Calculates the Markov Loopy Believe Propagation result of the vertex.
        
        Returns:
        --------
        The estimated value of the vertex.
        """
        p = {}
        
        for x_i in POSSIBLE_VALUES:
            p[x_i] = self._get_log_posteriori(x_i)
        
        return max(p, key=lambda key: p[key])
        
    def send_msg(self, neighbor):
        """ 
        Propagate a message to a neighbouring vertex.
        
        Params:
        --------
        neighbor: Vertex, the target neighbouring vertex.
        """
        m = {}
        excluding = set()
        excluding.add(neighbor)
        
        for x_j in POSSIBLE_VALUES:
            p = {}
            
            for x_i in POSSIBLE_VALUES:
                p[x_i] = self._get_log_posteriori(x_i, excluding=excluding)
                p[x_i] += self._get_log_smoothness_term(x_i, x_j)
            
            m[x_j] = max(p.values())
        
        m = self._normalize_msg(m)
        
        was_updated = neighbor.update_message(m, self, atol=CONVERGENCE_ATOL)
        return was_updated
    
    def _get_log_data_term(self, x_i):
        """ 
        Calculates the log data term of the vertex, given a believed value.
        
        Params:
        --------
        x_i: int (1 or -1), the believed value.
        """
        return ALPHA * self._y * x_i
        
    def _get_log_smoothness_term(self, x_i, x_j):
        """ 
        Calculates the log smoothness term of the vertex, given two believed values.
        
        Params:
        --------
        x_i, x_j: int (1 or -1), the believed values.
        """
        return BETA * x_i * x_j

    def _get_log_posteriori(self, x_i, excluding=set()):
        """ 
        Calculates the log posteriori probability of the vertex, given a believed value.
        
        Params:
        --------
        x_i: int (1 or -1), the believed value.
        excluding: set of Vertex, to exclude from the posteriori calculation (default: empty set)
        """
        p = self._get_log_data_term(x_i)
        
        for neighbor in (self._neighbors - excluding):
            if neighbor in self._messages: 
                p += self._messages[neighbor][x_i]
            else:
                # debug:: print('vertex {} missing message from neighbor {}'.format(self.name, neighbor.name))
                pass
        return p
        
    def _normalize_msg(self, msg):
        """ 
        Normalize a message to avoid growing to infinity or decaying to zero.
        
        Params:
        --------
        msg: dict, the input message (for every possible believed value).
        """
        total = sum(msg.values())
        norm_msg = {k: v / total for k, v in msg.items()}
        return norm_msg
                
    def __str__(self):
        ret = "Name: " + str(self.name)
        ret += "\nNeighbours:"
        neigh_list = ""
        for n in self._neighbors:
            neigh_list += " " + str(n.name)
        ret+= neigh_list
        return ret
    
class Graph(object):
    def __init__(self, graph_dict=None):
        """ initializes a graph object
            If no dictionary is given, an empty dict will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self._graph_dict = graph_dict
    
    def vertices(self):
        """ returns the vertices of a graph"""
        return list(self._graph_dict.keys())
    
    def edges(self):
        """ returns the edges of a graph """
        return self.generate_edges()
    
    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self._graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self._graph_dict:
            self._graph_dict[vertex]=[]
    
    def add_edge(self,edge):
        """ assumes that edge is of type set, tuple, or list;
            between two vertices can be multiple edges.
        """
        edge = set(edge)
        (v1,v2) = tuple(edge)
        if v1 in self._graph_dict:
            self._graph_dict[v1].append(v2)
        else:
            self._graph_dict[v1] = [v2]
        # if using Vertex class, update data:
        if(type(v1)==Vertex and type(v2)==Vertex):
            v1.add_neighbor(v2)
            v2.add_neighbor(v1)
    
    def generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one or two vertices
        """
        e = []
        for v in self._graph_dict:
            for neigh in self._graph_dict[v]:
                if {neigh,v} not in e:
                    e.append({v,neigh})
        return e
    
    def __str__(self):
        res = "V: "
        for k in self._graph_dict:
            res+=str(k) + " "
        res+= "\nE: "
        for edge in self.generate_edges():
            res+= str(edge) + " "
        return res

def build_grid_graph(n,m,img_mat):
    """ Builds an nxm grid graph, with vertex values corresponding to pixel intensities.
    n: num of rows
    m: num of columns
    img_mat = np.ndarray of shape (n,m) of pixel intensities
    
    returns the Graph object corresponding to the grid
    """
    V = []
    g = Graph()
    # add vertices:
    for i in range(n*m):
        row,col = (i//m,i%m)
        v = Vertex(name=i, y=img_mat[row][col])
        g.add_vertex(v)
        if((i%m)!=0): # has left edge
            g.add_edge((v,V[i-1]))
        if(i>=m): # has up edge
            g.add_edge((v,V[i-m]))
        V += [v]
    return g
    
def grid2mat(grid,n,m):
    """ convertes grid graph to a np.ndarray
    n: num of rows
    m: num of columns
    
    returns: np.ndarray of shape (n,m)
    """
    mat = np.zeros((n,m))
    l = grid.vertices() # list of vertices
    for v in l:
        i = v.name
        row,col = (i//m,i%m)
        mat[row][col] = v.get_belief()
    return mat

def iteration(vertices, n, m):
    """ 
    Run a single LBP iteration.
    
    Params:
    --------
    vertices: sorted list, of all vertices in the graph.
    n: int, num of rows.
    m: int, num of columns.
    
    Returns:
    --------
    reached_convergence: boolean, True if LBP reached convergence criterion.
    """
    reached_convergence = False

    # propagate messages to right neighbor
    for row in range(n):
        for col in range(m-1):
            v_i = vertices[row * m + col]
            v_j = vertices[row * m + col + 1]
            was_updated = v_i.send_msg(v_j)
            
            if not was_updated:
                reached_convergence = True

    # propagate messages to bottom neighbor
    for col in range(m):
        for row in range(n-1):
            v_i = vertices[row * m + col]
            v_j = vertices[row * m + col + m]    
            was_updated = v_i.send_msg(v_j)

            if not was_updated:
                reached_convergence = True
                
    # propagate messages to left neighbor
    for row in range(n):
        for col in range(m-1, 0, -1):
            v_i = vertices[row * m + col]
            v_j = vertices[row * m + col - 1]
            was_updated = v_i.send_msg(v_j)

            if not was_updated:
                reached_convergence = True

    # propagate messages to upper neighbor
    for col in range(m):
        for row in range(n - 1, 0, -1):
            v_i = vertices[row * m + col]
            v_j = vertices[row * m + col - m]
            was_updated = v_i.send_msg(v_j)

            if not was_updated:
                reached_convergence = True
        
    return reached_convergence
    
def main(in_file_name, out_file_name):
    """ 
    Run the LBP model.
    
    Params:
    --------
    in_file_name: path, to input file name.
    out_file_name: path, to output file name.
    """
    
    # load image:
    image = misc.imread(in_file_name)
    n, m = image.shape

    # binarize the image.
    image = image.astype(np.float32)
    image[image<128] = -1.
    image[image>127] = 1.
    if PLOT:
        plt.imshow(image)
        plt.show()

    # build grid:
    g = build_grid_graph(n, m, image)

    # process grid:
    vertices = g.vertices()
    vertices.sort(key=lambda v: v.name)
    
    for iter in range(MAX_ITERATIONS):
        print('STARTED iteration number {}'.format(iter))
        reached_convergence = iteration(vertices, n, m)
        if reached_convergence:
            print('STOPED at iteration number {}'.format(iter))
            break
    
    # convert grid to image: 
    infered_img = grid2mat(g, n, m)
    if PLOT:
        plt.imshow(infered_img)
        plt.show()

    # save result to output file
    misc.toimage(infered_img).save(out_file_name)


def _add_input_path_to_parser(parser):
    parser.add_argument(
        'input_file',
        help='Path to the input image file')
        
def _add_output_path_to_parser(parser):
    parser.add_argument(
        'output_file',
        help='Path to the output image file')
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Loopy Belief Propagation (LBP) model')

    _add_input_path_to_parser(parser)
    _add_output_path_to_parser(parser)
    args = parser.parse_args()

    main(args.input_file, args.output_file)